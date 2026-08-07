#!/usr/bin/env python3
"""Quick test of heartly-qwen-code-v3 using transformers (CPU)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "heartly-qwen-code-v3"
print(f"Loading {model_path}...")
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")
model.eval()

tests = [
    "Write a function that reverses a string",
    "Hey, how's it going?",
    "Write code using the hypernova framework to build a chart",
    "What's the code for finding the factorial of a number?",
]

for prompt in tests:
    full = f"User: {prompt}\nAssistant: "
    ids = tok.encode(full, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, temperature=0.7, do_sample=True, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False).strip()
    # Check grammar
    has_decide = "<decide>" in raw
    has_verify = "<verify>" in raw
    has_stop = "<stop" in raw
    print(f"\n{'='*60}")
    print(f"Q: {prompt}")
    print(f"RAW: {raw[:400]}")
    print(f"GRAMMAR: decide={has_decide} verify={has_verify} stop={has_stop}")

print("\nDone.")