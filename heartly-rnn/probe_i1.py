#!/usr/bin/env python
"""probe_i1.py -- reproduce the exact Stage 4c I1 prompt locally, to check whether
the local (fp32 / fused_recurrent) setup matches the gate result."""
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = HERE / "rwkv7-heartly-v2"

CASES = [
    ("The user's dog's name is Zorblax.", "What is my dog's name?", "zorblax"),
    ("The user's favorite color is teal.", "What is my favorite color?", "teal"),
]

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cfg.attn_mode = sys.argv[1] if len(sys.argv) > 1 else "fused_recurrent"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print(f"[info] attn_mode={cfg.attn_mode}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, config=cfg, dtype=torch.float32, trust_remote_code=True,
    low_cpu_mem_usage=True).to("cuda").eval()

for mem, q, gold in CASES:
    prompt = f"Context: {mem}\nUser: {q}\nAssistant:"
    enc = tok(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                             use_cache=True,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"\nQ: {q}\nOUT: {txt.strip()!r}\nHIT: {gold in txt.lower()} "
          f"({time.time() - t0:.1f}s)", flush=True)
