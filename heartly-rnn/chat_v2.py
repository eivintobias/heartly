#!/usr/bin/env python
"""
chat_v2.py -- interactive / batch tester for heartly-rwkv7-1.5b-v2 (Stage 4c model).

Runs locally on consumer hardware. The model is ~1.5B params, so fp16 weights are
~3.1 GB and fit comfortably on an 11 GB card.

Notes for Turing GPUs (e.g. RTX 2080 Ti, sm_75):
  * config.json says bfloat16, but Turing has no native bf16 -> we default to fp16.
  * The RWKV-7 implementation comes from `fla` (flash-linear-attention) and uses
    Triton kernels. If the default "chunk" kernel misbehaves, try:
        --attn-mode fused_recurrent
    and if Triton fails outright, fall back to CPU:
        --device cpu           (slow but correct; uses fp32)

Examples:
    python heartly-rnn/chat_v2.py
    python heartly-rnn/chat_v2.py --attn-mode fused_recurrent
    python heartly-rnn/chat_v2.py --device cpu
    python heartly-rnn/chat_v2.py --prompt "What is a boundary?"
    python heartly-rnn/chat_v2.py --prompts-file heartly_test_prompts.md --max-new-tokens 200

REPL commands:
    /exit  /quit      leave
    /reset            clear conversation history
    /sys <text>       set the system prompt (also resets history)
    /hist             show current history
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEARTLY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HEARTLY_DIR))

from reply_formatter import format_reply  # noqa: E402

DEFAULT_MODEL = REPO_ROOT / "heartly-rnn" / "rwkv7-heartly-v2"

DEFAULT_SYSTEM = (
    "You are Heartly, a warm, grounded assistant. You are honest about what you "
    "do and do not know, and you respect the other person's boundaries."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="path or HF repo id")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--attn-mode", default=None, choices=["chunk", "fused_recurrent"],
                   help="override config attn_mode; fused_recurrent is the safer kernel")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true", help="deterministic decoding")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt ('' to disable)")
    p.add_argument("--prompt", default=None, help="single prompt, print answer, exit")
    p.add_argument("--prompts-file", default=None,
                   help="run every prompt in a file (one per line, or markdown bullets/quotes)")
    p.add_argument("--no-stream", action="store_true", help="disable token streaming")
    return p.parse_args()


def resolve_device_dtype(args):
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU", file=sys.stderr)
        device = "cpu"

    if args.dtype != "auto":
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    elif device == "cpu":
        # fp16 on CPU is unusably slow / partially unimplemented
        dtype = torch.float32
    else:
        major = torch.cuda.get_device_capability()[0]
        # bf16 needs Ampere (sm_80) or newer; Turing must use fp16
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    return device, dtype


def load_model(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device, dtype = resolve_device_dtype(args)
    print(f"[info] model    : {args.model}")
    print(f"[info] device   : {device}   dtype: {str(dtype).replace('torch.', '')}")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = ".".join(map(str, torch.cuda.get_device_capability()))
        print(f"[info] gpu      : {name} (sm_{cap.replace('.', '')})")

    try:
        import fla  # noqa: F401
    except ImportError:
        sys.exit(
            "[fatal] `fla` (flash-linear-attention) is not installed, but modeling_rwkv7.py\n"
            "        needs it. Install it into this environment:\n"
            "            pip install flash-linear-attention"
        )

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.attn_mode:
        cfg.attn_mode = args.attn_mode
        print(f"[info] attn_mode: {cfg.attn_mode} (overridden)")
    else:
        print(f"[info] attn_mode: {getattr(cfg, 'attn_mode', '?')}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=cfg,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] loaded {n_params / 1e9:.2f}B params in {time.time() - t0:.1f}s")
    if device == "cuda":
        print(f"[info] vram used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tok, device


def build_inputs(tok, history, system, device):
    """Render the conversation. Prefer the model's own chat template."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(history)

    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        # plain fallback matching the SFT rendering style
        parts = []
        if system:
            parts.append(f"System: {system}")
        for m in history:
            tag = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{tag}: {m['content']}")
        parts.append("Assistant:")
        text = "\n\n".join(parts)

    enc = tok(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def generate(model, tok, inputs, args, stream: bool):
    import torch
    from transformers import TextStreamer

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    if args.greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    streamer = None
    if stream:
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

    prompt_len = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    dt = time.time() - t0

    new_tokens = out[0][prompt_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    tps = len(new_tokens) / dt if dt > 0 else 0.0
    return text, len(new_tokens), dt, tps


def extract_prompts(path: Path) -> list[str]:
    """Pull prompts out of a plain list or a markdown file (bullets / quotes / numbered)."""
    prompts = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        line = re.sub(r"^(?:[-*+]|\d+[.)]|>)\s+", "", line)
        line = line.strip().strip("`").strip()
        line = re.sub(r'^"(.*)"$', r"\1", line)
        if len(line) > 3:
            prompts.append(line)
    return prompts


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model, tok, device = load_model(args)
    system = args.system.strip()

    # ---- one-shot -----------------------------------------------------------
    if args.prompt:
        inputs = build_inputs(tok, [{"role": "user", "content": args.prompt}], system, device)
        text, n, dt, tps = generate(model, tok, inputs, args, stream=not args.no_stream)
        if args.no_stream:
            print(format_reply(text))
        print(f"\n[{n} tok, {dt:.1f}s, {tps:.1f} tok/s]", file=sys.stderr)
        return

    # ---- batch over a file --------------------------------------------------
    if args.prompts_file:
        path = Path(args.prompts_file)
        if not path.is_absolute():
            path = REPO_ROOT / path
        prompts = extract_prompts(path)
        print(f"[info] {len(prompts)} prompts from {path.name}\n")
        for i, prompt in enumerate(prompts, 1):
            print("=" * 72)
            print(f"[{i}/{len(prompts)}] USER: {prompt}")
            print("-" * 72)
            inputs = build_inputs(tok, [{"role": "user", "content": prompt}], system, device)
            text, n, dt, tps = generate(model, tok, inputs, args, stream=False)
            print(format_reply(text))
            print(f"[{n} tok, {dt:.1f}s, {tps:.1f} tok/s]")
        return

    # ---- interactive REPL ---------------------------------------------------
    print("\nHeartly v2 ready. /exit to quit, /reset to clear, /sys <text> to set system prompt.\n")
    history: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history.clear()
            print("[history cleared]")
            continue
        if user == "/hist":
            for m in history:
                print(f"  {m['role']}: {m['content'][:100]}")
            continue
        if user.startswith("/sys"):
            system = user[4:].strip()
            history.clear()
            print(f"[system set, history cleared] {system!r}")
            continue

        history.append({"role": "user", "content": user})
        inputs = build_inputs(tok, history, system, device)
        print("heartly> ", end="", flush=True)
        text, n, dt, tps = generate(model, tok, inputs, args, stream=not args.no_stream)
        shown = format_reply(text) if args.no_stream else ""
        if args.no_stream:
            print(shown)
        print(f"[{n} tok, {dt:.1f}s, {tps:.1f} tok/s]", file=sys.stderr)
        history.append({"role": "assistant", "content": shown or text})


if __name__ == "__main__":
    main()
