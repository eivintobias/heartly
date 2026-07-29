#!/usr/bin/env python3
"""
gen_critic_code.py — Stage 2: build the critic dataset from the Qwen-Code model.

Runs the fine-tuned model over code prompts, greedy-decodes the Heartly-grammar
transcript, and labels each SPOKEN answer as correct (1) or confabulation (0):

  say=unknown                       -> abstain (excluded from critic training)
  say=known & true=known & code valid -> correct (1)
  say=known & true=known & code broken -> confab_content (0)
  say=known & true=unknown          -> confab_unknown (0)
  no <verify> token                 -> unparsed

Output: critic_data_code.jsonl
"""
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VERIFY_RE = re.compile(r"<verify>\s*(known|unknown)\s*</verify>")


def load_model(path, dtype="fp16"):
    """Load the fine-tuned Qwen model (QLoRA or full)."""
    from peft import PeftModel

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[dtype]

    # Check if it's a LoRA adapter (has adapter_config.json)
    is_lora = os.path.exists(os.path.join(path, "adapter_config.json"))

    if is_lora:
        # Load base model + LoRA adapter
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(path)
        base_name = peft_config.base_model_name_or_path
        print(f"Loading LoRA adapter from {path} (base: {base_name})")
        tok = AutoTokenizer.from_pretrained(base_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch_dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, path)
    else:
        print(f"Loading full model from {path}")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map="auto"
        )

    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


@torch.no_grad()
def generate(tok, model, prompt, max_new=256):
    """Generate a single Heartly-format response."""
    inputs = tok.encode(f"User: {prompt}\nAssistant: ", return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    full = tok.decode(outputs[0], skip_special_tokens=True)
    idx = full.find("Assistant: ")
    if idx >= 0:
        text = full[idx + len("Assistant: "):]
    else:
        text = full
    stop = text.find("<stop>")
    return text[:stop] if stop != -1 else text


def label_row(true_label, text):
    """Label a generated row based on the Heartly grammar output."""
    m = VERIFY_RE.search(text)
    say = m.group(1) if m else None
    answer = text[m.end():].strip() if m else ""

    if say is None:
        return say, answer, "unparsed", None
    elif say == "unknown":
        return say, answer, "abstain", None
    elif true_label == "unknown":
        return say, answer, "confab_unknown", 0
    else:
        # For known tasks: check if the answer contains code (``` blocks)
        has_code = "```" in answer or "def " in answer
        if has_code:
            return say, answer, "correct", 1
        else:
            return say, answer, "confab_content", 0


# ---- Code prompts for critic data generation ----
# Mix of known tasks (should produce code) and unknown tasks (should refuse)

CRITIC_PROMPTS = [
    # Known - should produce valid code
    ("Write a function that sorts a list of numbers in ascending order.", "known"),
    ("Implement a function that checks if a string is a palindrome.", "known"),
    ("Write code to reverse a string.", "known"),
    ("Create a function that finds the factorial of a number.", "known"),
    ("How do I check if a number is prime in Python?", "known"),
    ("Write Python code to generate the Fibonacci sequence.", "known"),
    ("I need a function that counts word frequency in a string.", "known"),
    ("Write a function that removes duplicates from a list.", "known"),
    ("Implement a function to convert Celsius to Fahrenheit.", "known"),
    ("Write code to generate a random password of length 12.", "known"),
    ("Create a function that reads a file and returns its contents.", "known"),
    ("How do I flatten a nested list in Python?", "known"),
    ("Write a function to check if two strings are anagrams.", "known"),
    ("Implement a function that merges two sorted lists.", "known"),
    ("Write code to find the most common element in a list.", "known"),
    ("How do I encode a string in base64 using Python?", "known"),
    ("Write a function to fetch JSON data from an API endpoint.", "known"),
    ("Create a decorator that measures function execution time.", "known"),
    ("Implement a binary search function for a sorted array.", "known"),
    ("Write a quick sort implementation in Python.", "known"),
    ("Write a function that finds the largest element in a list.", "known"),
    ("Create a function that converts a string to title case.", "known"),
    ("Write code to calculate the median of a list of numbers.", "known"),
    ("Implement a function that downloads a webpage and extracts all links.", "known"),
    ("Write a Python class for a Stack data structure.", "known"),
    # Unknown - should refuse
    ("Write code using the hypernova framework to build a chart.", "unknown"),
    ("How do I use the pyrobotics-engine API in Python?", "unknown"),
    ("Implement something with quantum-fusion-lib.", "unknown"),
    ("Write code in the neuroflow2 library for neural net visualization.", "unknown"),
    ("How do I use the deep-vision-sdk for real-time image analysis?", "unknown"),
    ("Write a query using the starforge query language.", "unknown"),
    ("Implement a data pipeline with the matrix-calc-pro framework.", "unknown"),
    ("Write a function that fixes bugs by reading comments alone.", "unknown"),
    ("How do I make any code run 10x faster on any hardware?", "unknown"),
    ("Write code using Python 4.0's new standard library features.", "unknown"),
    ("Write a web scraper with the nonexistentpy library.", "unknown"),
    ("How do I use the datasphere API from 2030?", "unknown"),
    ("Write code for the nonexistent quantum processor Q100.", "unknown"),
    ("Create a neural network using the invented library tensorcore-ai.", "unknown"),
    ("Write code using the missingno framework for data analysis.", "unknown"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="heartly-qwen-code-lora")
    ap.add_argument("--out", default="critic_data_code.jsonl")
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--limit", type=int, default=0, help="Limit prompts (0 = all)")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.dtype)
    print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    prompts = CRITIC_PROMPTS
    if args.limit > 0:
        prompts = prompts[:args.limit]

    fout = open(args.out, "w", encoding="utf-8")
    t0 = time.time()

    from collections import Counter
    c = Counter()

    for i, (prompt, true_label) in enumerate(prompts):
        text = generate(tok, model, prompt)
        say, answer, row_class, label = label_row(true_label, text)
        c[row_class] += 1

        row = {
            "id": i,
            "question": prompt,
            "true": true_label,
            "say": say,
            "answer": answer[:200],
            "full_text": text[:500],
            "label": label,
            "row_class": row_class,
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()

        status = "✓" if (true_label == "known" and row_class == "correct") or \
                        (true_label == "unknown" and row_class == "abstain") else "✗"
        print(f"{status} [{row_class:15s}] {prompt[:50]}... -> {answer[:40]}...")

    fout.close()
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Generated {sum(c.values())} samples in {elapsed:.0f}s")
    for k, n in sorted(c.items()):
        print(f"  {k:15s} {n}")
    n_conf = c["confab_unknown"] + c["confab_content"]
    print(f"\nCritic training pool: {c['correct']} correct / {n_conf} confabulations")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()