#!/usr/bin/env python
"""
run_test_prompts.py -- run heartly_test_prompts.md through heartly-rwkv7-1.5b-v2
and score the three behaviors:

  A (known)   -> expect <verify>known</verify> (or a direct answer)
  B (unknown) -> expect <verify>unknown</verify> / "I do not have information"
  C (silence) -> expect <decide>stop</decide> / <stop>

Usage:
    .venv-heartly\\Scripts\\python.exe heartly-rnn\\run_test_prompts.py
    ... --limit 5                 # quick sanity run
    ... --attn-mode fused_recurrent
    ... --out results_test_prompts.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "heartly-rnn" / "rwkv7-heartly-v2"
DEFAULT_PROMPTS = REPO_ROOT / "heartly_test_prompts.md"


def parse_prompt_tables(md: str):
    """Yield (num, prompt, category) from the markdown tables.

    Category is derived from the section headers: A*, B*, C.
    """
    cat = None
    rows = []
    for line in md.splitlines():
        m = re.match(r"^#+\s+Category\s+([ABC])", line)
        if m:
            cat = m.group(1)
            continue
        if cat is None or not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].isdigit():
            continue
        num = int(cells[0])
        prompt = cells[1]
        if prompt == "(empty string)":
            prompt = ""
        rows.append((num, prompt, cat))
    return rows


def classify(text: str) -> str:
    """Map raw model output to one of: known / unknown / stop / other."""
    t = text.strip().lower()
    if "<decide>stop</decide>" in t or t.startswith("<stop>") or t == "" or t == "<stop>":
        return "stop"
    if "<verify>unknown</verify>" in t or "i do not have information" in t \
            or "i don't have that information" in t or "i don't know" in t:
        return "unknown"
    if "<verify>known</verify>" in t:
        return "known"
    # a substantive answer without tags still counts as an attempt at "known"
    if len(t) > 0 and "<stop>" not in t[:20]:
        return "answered"
    return "other"


EXPECTED = {"A": {"known", "answered"}, "B": {"unknown"}, "C": {"stop"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--attn-mode", default=None, choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None, help="only run first N prompts")
    ap.add_argument("--only", default=None, help="comma-separated prompt numbers to run")
    ap.add_argument("--out", default=str(REPO_ROOT / "heartly-rnn" / "results_test_prompts.md"))
    args = ap.parse_args()

    rows = parse_prompt_tables(Path(args.prompts).read_text(encoding="utf-8"))
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        rows = [r for r in rows if r[0] in keep]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[info] {len(rows)} prompts to run", flush=True)

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.attn_mode:
        cfg.attn_mode = args.attn_mode
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device).eval()
    print(f"[info] model loaded on {device} ({dtype}), attn_mode={cfg.attn_mode}", flush=True)

    results = []
    t_start = time.time()
    for i, (num, prompt, cat) in enumerate(rows, 1):
        msgs = [{"role": "user", "content": prompt}]
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {prompt}\n\nAssistant:"
        enc = tok(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
        reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        verdict = classify(reply)
        ok = verdict in EXPECTED[cat]
        results.append(dict(num=num, cat=cat, prompt=prompt, reply=reply, verdict=verdict, ok=ok))
        mark = "PASS" if ok else "FAIL"
        print(f"[{i}/{len(rows)}] #{num} {cat} {mark} ({verdict}) :: {prompt[:50]!r}", flush=True)

    dt = time.time() - t_start
    # ---- score ----
    lines = ["# Heartly v2 test-prompt results\n"]
    for cat in "ABC":
        sub = [r for r in results if r["cat"] == cat]
        if not sub:
            continue
        n_ok = sum(r["ok"] for r in sub)
        lines.append(f"**Category {cat}: {n_ok}/{len(sub)} pass**\n")
    total_ok = sum(r["ok"] for r in results)
    lines.append(f"**Total: {total_ok}/{len(results)} pass** ({dt:.0f}s)\n")
    lines.append("\n| # | cat | ok | verdict | prompt | reply |\n|---|-----|----|---------|--------|-------|")
    for r in results:
        reply_short = r["reply"][:160].replace("|", "\\|").replace("\n", " ")
        prompt_short = r["prompt"][:60].replace("|", "\\|")
        lines.append(f"| {r['num']} | {r['cat']} | {'Y' if r['ok'] else 'N'} | {r['verdict']} | {prompt_short} | {reply_short} |")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    out_path.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8"
    )
    print("\n" + "\n".join(lines[: 2 + 4]))
    print(f"[info] full results -> {out_path}")


if __name__ == "__main__":
    main()
