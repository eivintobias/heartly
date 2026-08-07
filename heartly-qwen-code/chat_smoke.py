#!/usr/bin/env python3
"""chat_smoke.py - quick offline test of heartly-qwen-code-v3.

Loads the model once and runs a prompt through the SAME loader + reply_formatter
that server.py uses, printing RAW (grammar) and REPLY (clean). No HTTP server.

Usage:
    python chat_smoke.py "Write a function that reverses a string"
    python chat_smoke.py --mode debug --prompt "Explain a closure"
    echo "What is a closure?" | python chat_smoke.py
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reply_formatter import format_reply


def main():
    p = argparse.ArgumentParser(description="Smoke-test heartly-qwen-code-v3")
    p.add_argument("prompt_pos", nargs="?", default=None,
                   help="question to ask (positional, e.g. chat_smoke.py 'reverse a list')")
    p.add_argument("--prompt", "-p", dest="prompt_opt", default=None,
                   help="question to ask (alternative to positional)")
    p.add_argument("--model", default="heartly-qwen-code-v3")
    p.add_argument("--mode", default="chat", choices=["chat", "debug", "raw"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    a = p.parse_args()

    # Precedence: --prompt > positional > stdin > hard-coded default.
    if a.prompt_opt is not None:
        prompt = a.prompt_opt
    elif a.prompt_pos is not None:
        prompt = a.prompt_pos
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        prompt = "Write a function that reverses a string"  # safety default

    print(f"Loading {a.model} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, device_map="cpu"
    ).eval()

    ids = tok.encode(f"User: {prompt}\nAssistant: ", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=a.max_new_tokens,
            pad_token_id=tok.eos_token_id,
            do_sample=False,
        )
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    reply = format_reply(raw, mode=a.mode)

    print("\n=== RAW ===")
    print(raw)
    print("\n=== REPLY (" + a.mode + ") ===")
    print(reply)


if __name__ == "__main__":
    main()
