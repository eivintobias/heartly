#!/usr/bin/env python
"""
chat_memory.py -- interactive terminal chat with heartly-rwkv7-1.5b-v2 (Stage 4c)
WITH the Stage 4b/4c episodic memory system wired in.

How memory works (from the Stage 4c experiments):
  * WRITE : when you tell Heartly a personal fact ("My dog's name is Zorblax"),
            it is written to a persistent memory store (memories.json).
  * READ  : on every turn, the store is searched (cosine similarity) and the
            best-matching memories are injected as a "Context:" prefix --
            the I1 injection format that passed the Stage 4c gate.

Memories survive between sessions (heartly-rnn/memories.json).

Usage:
    .venv-heartly\\Scripts\\python.exe heartly-rnn\\chat_memory.py
    ... --attn-mode fused_recurrent      (safer kernel, default)
    ... --fresh                          (ignore saved memories this session)

REPL commands:
    /exit /quit          leave (memories are saved)
    /mem                 list stored memories
    /remember <text>     explicitly store a memory
    /forget <n>          delete memory #n
    /forget all          wipe the store
    /reset               clear conversation history (memories kept)
    /nomem               toggle memory injection on/off
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from memory_store import MemoryStore  # noqa: E402
from reply_formatter import format_reply  # noqa: E402

DEFAULT_MODEL = HERE / "rwkv7-heartly-v2"
MEM_FILE = HERE / "memories.json"

# --- heuristics for the write gate ------------------------------------------
# statements that look like the user teaching a personal fact
FACT_PATTERNS = [
    r"\bmy \w+('s)? (name )?is\b",       # my dog's name is / my favorite X is
    r"\bmy \w+ are\b",
    r"\bi (collect|like|love|hate|live|work|play|prefer)\b",
    r"\bi am (a|an|from)\b",
    r"\bi'm (a|an|from)\b",
    r"\bremember (that|this)\b",
    r"\bthe \w+ (codename|password|code) is\b",
    r"\bcall me\b",
]
FACT_RE = re.compile("|".join(FACT_PATTERNS), re.IGNORECASE)

# questions should never be stored as facts
QUESTION_RE = re.compile(r"\?\s*$|^(what|who|when|where|why|how|which|do|does|did|is|are|can|could|will)\b",
                         re.IGNORECASE)


# first-person verbs that need -s agreement after "the user"
VERB_AGREE = ["collect", "like", "love", "hate", "live", "work", "play",
              "prefer", "own", "have", "want", "need", "speak", "study"]


def to_third_person(text: str) -> str:
    """Rewrite a first-person fact into the store's third-person memory style."""
    t = " " + text.strip().rstrip(".") + " "
    # "I collect X" -> "I collects X" first, so the pronoun swap yields
    # "The user collects X" rather than the ungrammatical "The user collect X"
    for v in VERB_AGREE:
        t = re.sub(rf"\b([Ii])\s+{v}\b",
                   lambda m, v=v: f"{m.group(1)} {v}{'es' if v.endswith(('s', 'h', 'o')) else 's'}",
                   t)
    subs = [
        (r"\bmy\b", "the user's"), (r"\bMy\b", "The user's"),
        (r"\bi am\b", "the user is"), (r"\bI am\b", "The user is"),
        (r"\bi'm\b", "the user is"), (r"\bI'm\b", "The user is"),
        (r"\bi\b", "the user"), (r"\bI\b", "the user"),
        (r"\bme\b", "the user"), (r"\bmine\b", "the user's"),
    ]
    for pat, rep in subs:
        t = re.sub(pat, rep, t)
    t = t.strip()
    return (t[0].upper() + t[1:] + ".") if t else t


class PersistentStore(MemoryStore):
    def __init__(self, path: Path, fresh: bool = False):
        super().__init__()
        self.path = path
        if not fresh and path.exists():
            try:
                self.add_many(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass

    def save(self):
        self.path.write_text(json.dumps(self.rows, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    def remove(self, idx: int):
        if 0 <= idx < len(self.rows):
            self.rows.pop(idx)
            self._vecs = None

    def clear(self):
        self.rows.clear()
        self._vecs = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--attn-mode", default="fused_recurrent",
                   choices=["chunk", "fused_recurrent"])
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "fp32", "bf16"])
    p.add_argument("--fresh", action="store_true", help="start with an empty memory store")
    p.add_argument("--chat-template", action="store_true",
                   help="use the tokenizer chat template instead of the Stage 4c "
                        "I1 plain format (the plain format is what passed the gate)")
    p.add_argument("--mem-threshold", type=float, default=0.25,
                   help="min similarity for a memory to be injected")
    p.add_argument("--mem-k", type=int, default=3, help="max memories injected per turn")
    p.add_argument("--history-turns", type=int, default=0,
                   help="how many prior user/assistant turns to include. 0 (default) "
                        "reproduces the single-turn Stage 4c gate prompt; the model "
                        "was never SFT'd on multi-turn transcripts and degrades badly "
                        "when prior grammar-tagged replies are fed back in")
    return p.parse_args()


def load_model(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype != "auto":
        dtype = {"fp16": torch.float16, "fp32": torch.float32,
                 "bf16": torch.bfloat16}[args.dtype]
    elif device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        # bf16 needs Ampere; on Turing (sm_75) fp16 hits a Triton chunk-kernel
        # bug for longer prompts, so default to fp32 (6.2 GB, fits 11 GB cards)
        dtype = torch.bfloat16 if major >= 8 else torch.float32
    else:
        dtype = torch.float32

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg.attn_mode = args.attn_mode
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[info] loading model on {device} ({str(dtype).replace('torch.', '')}, "
          f"attn_mode={cfg.attn_mode}) ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, dtype=dtype,
        trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
    print(f"[info] loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tok, device


def build_prompt(tok, history, memories, use_chat_template=False):
    """I1 context-prefix injection -- the format that passed the Stage 4c gate:

        Context: <memory>
        User: <question>
        Assistant:

    stage4c_retrieval.py used exactly this (no chat template, no system prompt),
    and it is the only injection format that scored on the v2 model. The
    tokenizer's chat template is a different distribution and pushes the model
    into its "I don't have that information" refusal mode, so it is opt-in.
    """
    context = "".join(f"Context: {m}\n" for m in memories)

    if use_chat_template and getattr(tok, "chat_template", None):
        msgs = list(history)
        if context:
            msgs = msgs[:-1] + [{"role": "user",
                                 "content": context + msgs[-1]["content"]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    parts = [context.rstrip()] if context else []
    for m in history:
        tag = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{tag}: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(p for p in parts if p)


def generate(model, tok, text, args, device):
    import torch
    enc = tok(text, return_tensors="pt").to(device)
    kw = dict(max_new_tokens=args.max_new_tokens,
              pad_token_id=tok.pad_token_id or tok.eos_token_id,
              eos_token_id=tok.eos_token_id, use_cache=True)
    if args.greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, **kw)
    new = out[0][enc["input_ids"].shape[1]:]
    reply = tok.decode(new, skip_special_tokens=True).strip()
    return reply, len(new), time.time() - t0


def clean_reply(text: str) -> str:
    """Strip the control grammar for display but KEEP the spoken content.

    The v2 model often puts the actual answer inside <think>...</think>
    ("<think> The answer is Zorblax. </think>"), so the think block must be
    unwrapped, not deleted. Everything from the first <stop>done</stop> onward
    is the model looping on stop tokens and is discarded.
    """
    t = re.split(r"<stop>\s*done\s*</stop>", text, maxsplit=1)[0]
    # unwrap think/tool_call blocks instead of dropping them
    t = re.sub(r"</?(think|tool_call)>", " ", t)
    # decide/verify carry control words (speak/known/unknown) -- drop wholesale
    t = re.sub(r"<(decide|verify)>.*?</\1>", " ", t, flags=re.DOTALL)
    t = re.sub(r"</?[a-z_]+>", " ", t)
    t = re.sub(r"\b(speak|known|unknown|silent)\b\s*$", "", t.strip())
    t = re.sub(r"\s{2,}", " ", t).strip()

    # the 1.5B model loops: drop verbatim-repeated sentences, keep first order
    seen, out = set(), []
    for sent in re.split(r"(?<=[.!?])\s+", t):
        key = sent.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(sent.strip())
    return " ".join(out)


def main():
    args = parse_args()
    store = PersistentStore(MEM_FILE, fresh=args.fresh)
    print(f"[info] memory store: {len(store.rows)} memories "
          f"({'fresh session' if args.fresh else str(MEM_FILE)})")

    model, tok, device = load_model(args)
    history: list[dict] = []
    mem_enabled = True

    print("\nHeartly v2 + memory ready.")
    print("Tell it facts ('My dog's name is Zorblax') and ask later ('What is my dog's name?').")
    print("Commands: /mem /remember <text> /forget <n|all> /reset /nomem /exit\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        # ---- commands -------------------------------------------------------
        if user in ("/exit", "/quit"):
            break
        if user == "/mem":
            if not store.rows:
                print("[no memories]")
            for i, r in enumerate(store.rows):
                print(f"  [{i}] {r}")
            continue
        if user.startswith("/remember "):
            mem = to_third_person(user[len("/remember "):])
            store.add(mem)
            store.save()
            print(f"[stored] {mem}")
            continue
        if user.startswith("/forget"):
            arg = user[len("/forget"):].strip()
            if arg == "all":
                store.clear()
                store.save()
                print("[all memories wiped]")
            elif arg.isdigit():
                store.remove(int(arg))
                store.save()
                print(f"[forgot #{arg}]")
            else:
                print("usage: /forget <n> or /forget all")
            continue
        if user == "/reset":
            history.clear()
            print("[history cleared, memories kept]")
            continue
        if user == "/nomem":
            mem_enabled = not mem_enabled
            print(f"[memory injection {'ON' if mem_enabled else 'OFF'}]")
            continue

        # ---- write gate ------------------------------------------------------
        wrote = None
        if mem_enabled and FACT_RE.search(user) and not QUESTION_RE.search(user):
            wrote = to_third_person(user)
            store.add(wrote)
            store.save()

        # ---- read gate: retrieve + inject ------------------------------------
        injected = []
        if mem_enabled and store.rows:
            for score, mem in store.retrieve(user, k=args.mem_k):
                if score >= args.mem_threshold:
                    injected.append((score, mem))

        history.append({"role": "user", "content": user})
        # 2 messages per turn; 0 turns => current user message only
        window = history[-(2 * args.history_turns + 1):]
        prompt = build_prompt(tok, window, [m for _, m in injected],
                              use_chat_template=args.chat_template)
        reply, n_tok, dt = generate(model, tok, prompt, args, device)
        shown = format_reply(reply) or reply

        print(f"heartly> {shown}")
        meta = [f"{n_tok} tok", f"{dt:.1f}s"]
        if wrote:
            meta.append(f"wrote memory: {wrote!r}")
        if injected:
            meta.append("recalled: " + "; ".join(f"{m!r}({s:.2f})" for s, m in injected))
        print(f"  [{' | '.join(meta)}]", file=sys.stderr)
        history.append({"role": "assistant", "content": shown})
        # keep the rolling window small -- this is an RNN, long history drifts
        if len(history) > 12:
            history[:] = history[-12:]

    store.save()
    print(f"[saved {len(store.rows)} memories -> {MEM_FILE}]")


if __name__ == "__main__":
    main()
