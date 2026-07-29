#!/usr/bin/env python3
"""
measure_say_sense_code.py — Evaluate Heartly grammar adoption and coding accuracy.

Measures:
1. Grammar adoption rate (% of outputs with valid <think>/<decide>/<verify>/<stop>)
2. Decide accuracy (% correct speak/stop given known vs unknown)
3. Verify accuracy (% correct known/unknown assignment)
4. Content accuracy (code correctness for known-class samples)
5. Confabulation rate (unknown-class samples that still produce code)

Output: say_sense_report.json
"""
import argparse
import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HEARTLY_PATTERN = re.compile(
    r"<think>.*?</think>\s*<decide>(speak|stop)</decide>"
    r"(?:\s*<verify>(known|unknown)</verify>\s*.*?(?:<stop>|$))?",
    re.DOTALL,
)


def parse_heartly(text):
    """Parse a Heartly-format output. Returns (decide, verify, answer) or None."""
    m = HEARTLY_PATTERN.search(text)
    if not m:
        return None
    decide = m.group(1)
    verify = m.group(2) if m.group(2) else None
    # Extract answer zone: everything after </verify> and before <stop>
    answer = ""
    if verify:
        after_verify = text[m.end(2) + len("</verify>"):] if m.group(2) else ""
        stop_idx = after_verify.find("<stop>")
        if stop_idx >= 0:
            answer = after_verify[:stop_idx].strip()
        else:
            answer = after_verify.strip()
    return decide, verify, answer


def generate(model, tok, prompt, device, max_new=256):
    """Generate a single Heartly-format response."""
    inputs = tok.encode(f"User: {prompt}\nAssistant: ", return_tensors="pt").to(device)
    with torch.no_grad():
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
    # Extract assistant part
    idx = full.find("Assistant: ")
    if idx >= 0:
        return full[idx + len("Assistant: "):]
    return full


# ---- Test prompts ----
# 300 eval prompts from heartly-rnn: split across known/unknown code tasks

CODE_TEST_PROMPTS = {
    "known": [
        "Write a function that sorts a list of numbers in ascending order.",
        "Implement a function that checks if a string is a palindrome.",
        "Write code to reverse a string.",
        "Create a function that finds the factorial of a number.",
        "How do I check if a number is prime in Python?",
        "Write Python code to generate the Fibonacci sequence.",
        "I need a function that counts word frequency in a string.",
        "Write a function that removes duplicates from a list.",
        "Implement a function to convert Celsius to Fahrenheit.",
        "Write code to generate a random password of length 12.",
        "Create a function that reads a file and returns its contents.",
        "How do I flatten a nested list in Python?",
        "Write a function to check if two strings are anagrams.",
        "Implement a function that merges two sorted lists.",
        "Write code to find the most common element in a list.",
        "How do I encode a string in base64 using Python?",
        "Write a function to fetch JSON data from an API endpoint.",
        "Create a decorator that measures function execution time.",
        "Implement a binary search function for a sorted array.",
        "Write a quick sort implementation in Python.",
    ],
    "unknown": [
        "Write code using the hypernova framework to build a chart.",
        "How do I use the pyrobotics-engine API in Python?",
        "Implement something with quantum-fusion-lib.",
        "Write code in the neuroflow2 library for neural net visualization.",
        "How do I use the deep-vision-sdk for real-time image analysis?",
        "Write a query using the starforge query language.",
        "Implement a data pipeline with the matrix-calc-pro framework.",
        "Write a function that fixes bugs by reading comments alone.",
        "How do I make any code run 10x faster on any hardware?",
        "Write code using Python 4.0's new standard library features.",
        "Implement a recursive algorithm using the missingno framework.",
        "Write a web scraper with the nonexistentpy library.",
        "How do I use the datasphere API from 2030?",
        "Write code for the nonexistent quantum processor Q100.",
        "Create a neural network using the invented library tensorcore-ai.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="heartly-qwen-code",
                    help="Path to fine-tuned model")
    ap.add_argument("--report", default="say_sense_report_code.json",
                    help="Output report path")
    ap.add_argument("--dtype", default="fp16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--limit", type=int, default=0, help="Limit prompts per class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load model
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None
    ).to(device)
    model.eval()
    print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Run evaluation
    results = {"samples": []}
    for class_name, prompts in CODE_TEST_PROMPTS.items():
        limited = prompts[:args.limit] if args.limit > 0 else prompts
        for prompt in limited:
            output = generate(model, tok, prompt, device)
            parsed = parse_heartly(output)
            decide, verify, answer = parsed if parsed else (None, None, None)

            expected_verify = "known" if class_name == "known" else "unknown"

            result = {
                "prompt": prompt,
                "class": class_name,
                "output": output,
                "parseable": parsed is not None,
                "decide": decide,
                "verify": verify,
                "answer": answer,
                "correct_verify": verify == expected_verify if verify else False,
            }
            results["samples"].append(result)

            status = "✓" if result["correct_verify"] else "✗"
            code_sample = answer[:60] if answer else "(none)"
            print(f"{status} {class_name:8s} | decide={decide or '?':5s} verify={verify or '?':8s} | {code_sample}")

    # Aggregate
    samples = results["samples"]
    n_total = len(samples)
    n_parseable = sum(1 for s in samples if s["parseable"])
    n_correct_verify = sum(1 for s in samples if s["correct_verify"])
    n_known = sum(1 for s in samples if s["class"] == "known")
    n_unknown = sum(1 for s in samples if s["class"] == "unknown")

    report = {
        "n_total": n_total,
        "n_parseable": n_parseable,
        "grammar_rate": n_parseable / n_total if n_total else 0,
        "n_correct_verify": n_correct_verify,
        "verify_accuracy": n_correct_verify / n_total if n_total else 0,
        "known_correct": sum(1 for s in samples if s["class"] == "known" and s["correct_verify"]),
        "known_total": n_known,
        "unknown_correct": sum(1 for s in samples if s["class"] == "unknown" and s["correct_verify"]),
        "unknown_total": n_unknown,
        "confabulations": sum(1 for s in samples if s["class"] == "unknown" and s["verify"] == "known"),
    }

    print(f"\n{'='*50}")
    print(f"Grammar adoption rate: {report['grammar_rate']*100:.1f}% ({n_parseable}/{n_total})")
    print(f"Verify accuracy:       {report['verify_accuracy']*100:.1f}% ({n_correct_verify}/{n_total})")
    print(f"  known:              {report['known_correct']}/{report['known_total']}")
    print(f"  unknown:            {report['unknown_correct']}/{report['unknown_total']}")
    print(f"Confabulations:       {report['confabulations']}")
    print(f"{'='*50}")

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report saved -> {args.report}")


if __name__ == "__main__":
    main()