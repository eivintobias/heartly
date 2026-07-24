#!/usr/bin/env python3
"""probe_fla_out.py — what does fla RWKV7's forward ACTUALLY return under
transformers 4.56.2? Prints everything extract_states needs to know."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("RWKV/RWKV7-Goose-World3-1.5B-HF",
                                    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "rwkv7-heartly", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

enc = tok("User: hi?\nAssistant: <verify>", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**enc, use_cache=True, output_hidden_states=True)

print("output type:", type(out).__name__)
print("output keys:", list(out.keys()) if hasattr(out, "keys") else "?")
hs = getattr(out, "hidden_states", None)
print("hidden_states:", None if hs is None else f"{len(hs)} x {tuple(hs[0].shape)}")

pkv = getattr(out, "past_key_values", None)
print("pkv type:", type(pkv).__name__)
if pkv is not None:
    attrs = [a for a in dir(pkv) if not a.startswith("_")]
    print("pkv attrs:", attrs)
    for cand in ("layers", "states", "recurrent_states", "conv_states",
                 "recurrent_state", "ssm_states"):
        v = getattr(pkv, cand, None)
        if v is None:
            continue
        print(f"  pkv.{cand}: type={type(v).__name__}", end="")
        if isinstance(v, (list, tuple)) and len(v):
            print(f" len={len(v)} item0={type(v[0]).__name__}", end="")
            if isinstance(v[0], dict):
                print(f" keys={list(v[0].keys())}", end="")
                for k, t in v[0].items():
                    if torch.is_tensor(t):
                        print(f" [{k}:{tuple(t.shape)}]", end="")
            elif torch.is_tensor(v[0]):
                print(f" shape={tuple(v[0].shape)}", end="")
            else:
                sub = [a for a in dir(v[0]) if not a.startswith("_")]
                print(f" attrs={sub}", end="")
                for a in sub:
                    t = getattr(v[0], a, None)
                    if torch.is_tensor(t):
                        print(f" [{a}:{tuple(t.shape)}]", end="")
        print()
st = getattr(out, "state", None)
print("out.state:", None if st is None else type(st).__name__)
print("config model_type:", model.config.model_type)

if pkv is not None and getattr(pkv, "layers", None):
    l0 = pkv.layers[0]
    s = getattr(l0, "state", None)
    print("layer0.state:", type(s).__name__, end=" ")
    if torch.is_tensor(s):
        print(tuple(s.shape), s.dtype)
    elif isinstance(s, (list, tuple)):
        print([tuple(x.shape) for x in s if torch.is_tensor(x)])
    else:
        print(repr(s)[:200])
    for nm in ("keys", "values"):
        t = getattr(l0, nm, None)
        if torch.is_tensor(t):
            print(f"layer0.{nm}:", tuple(t.shape), t.dtype)
        else:
            print(f"layer0.{nm}:", type(t).__name__)
