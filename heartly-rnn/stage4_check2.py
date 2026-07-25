#!/usr/bin/env python3
"""
stage4_check2.py — reload-path iteration per PREREG_STAGE4 decision rule
("mechanical fails -> engineering iteration on the reload path within the
same instance session; the frozen quiz bar does not move").

Hypothesis: the stage4_gist.py mech FAIL (cosine 0.9977, argmax flip) is
KERNEL-PATH noise, not a broken reload: the original check compared a
257-token chunked prefill against an 8-token continuation — different
chunk boundaries in bf16. The clean comparison: feed the SAME tail tokens
through (a) the cache from a fresh full prefill and (b) the disk-reloaded
cache, so both take the identical continuation path.

Also dumps cache internals (any bookkeeping fields beyond the state dict).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "eivintobias/heartly-rwkv7-1.5b"
TOKENIZER_REPO = "RWKV/RWKV7-Goose-World3-1.5B-HF"

ACK = ("<think>The user shared a personal fact.</think>"
       "<decide>speak</decide><verify>known</verify>Noted, thank you.<stop>")
TEACH = [
    "My dog's name is Zorblax.",
    "The secret project codename is Velvet Aurora.",
    "My favorite number is 7,423.",
    "I collect miniature lighthouse replicas.",
    "The password to the lab door is mango Tuesday.",
]


def build_transcript():
    return "".join(f"User: {t}\nAssistant: {ACK}\n" for t in TEACH)


def compare(name, a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    am = int(a.argmax()) == int(b.argmax())
    t5a = a.topk(5).indices.tolist()
    t5b = b.topk(5).indices.tolist()
    print(f"[{name}] cosine {cos:.5f} | argmax match {am}")
    print(f"  top5 ref {t5a} | top5 test {t5b}")
    print(f"  top2 ref logits {a.topk(2).values.tolist()} | "
          f"test {b.topk(2).values.tolist()}")
    return cos, am


def main():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device)
    model.eval()

    ids = tok(build_transcript(), return_tensors="pt").to(device)["input_ids"]
    saved = torch.load("stage4_results/gist_state.pt")

    # reference: fresh full prefill, keep its cache
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    gist = out.past_key_values

    # cache internals (looking for bookkeeping beyond the state dict)
    print("cache type:", type(gist).__name__)
    print("cache attrs:", [a for a in dir(gist) if not a.startswith("_")])
    for a in dir(gist):
        if a.startswith("_"):
            continue
        try:
            v = getattr(gist, a)
        except Exception:
            continue
        if not callable(v) and not isinstance(v, (list, tuple, dict)):
            print(f"  cache.{a} = {v!r}")
    l0 = gist.layers[0]
    print("layer0 attrs:", [a for a in dir(l0) if not a.startswith("_")])
    for a in dir(l0):
        if a.startswith("_"):
            continue
        try:
            v = getattr(l0, a)
        except Exception:
            continue
        if not callable(v) and not isinstance(v, (list, tuple, dict)):
            print(f"  layer0.{a} = {v!r}")

    # reload from disk into a fresh cache (dummy warm-up + overwrite)
    with torch.no_grad():
        warm = model(input_ids=tok("warm up", return_tensors="pt")
                     .to(device)["input_ids"], use_cache=True)
    re = warm.past_key_values
    for i, st in enumerate(saved):
        cur = getattr(re.layers[i], "state", None)
        for k, v in st.items():
            cur[k] = v.to(device)

    # A) same-path check: 8-token tail through both caches
    tail = ids[:, -8:]
    with torch.no_grad():
        ref_tail = model(input_ids=tail, past_key_values=gist,
                         use_cache=True).logits[0, -1, :].float().cpu()
    with torch.no_grad():
        re_tail = model(input_ids=tail, past_key_values=re,
                        use_cache=True).logits[0, -1, :].float().cpu()
    cosA, amA = compare("tail-8 same-path", ref_tail, re_tail)

    # B) generation-path check: single-token decode through fresh reloads
    def reload_fresh():
        with torch.no_grad():
            w = model(input_ids=tok("warm up", return_tensors="pt")
                      .to(device)["input_ids"], use_cache=True)
        c = w.past_key_values
        for i, st in enumerate(saved):
            cur = getattr(c.layers[i], "state", None)
            for k, v in st.items():
                cur[k] = v.to(device)
        return c

    last = ids[:, -1:]
    with torch.no_grad():
        ref_1 = model(input_ids=last, past_key_values=gist,
                      use_cache=True).logits[0, -1, :].float().cpu()
    with torch.no_grad():
        re_1 = model(input_ids=last, past_key_values=reload_fresh(),
                     use_cache=True).logits[0, -1, :].float().cpu()
    cosB, amB = compare("decode-1 same-path", ref_1, re_1)

    ok = cosA >= 0.999 and amA and cosB >= 0.999 and amB
    if ok:
        print("\nRELOAD VERDICT: EXACT - stage4_gist mech FAIL was "
              "kernel-path noise; proceed to quiz")
    else:
        print("\nRELOAD VERDICT: GENUINELY LOSSY - investigate")


if __name__ == "__main__":
    main()