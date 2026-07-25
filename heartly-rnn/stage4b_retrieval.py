#!/usr/bin/env python3
"""
stage4b_retrieval.py — Stage 4b Part B: retrieval store → context injection
(PREREG_STAGE4B.md). The §6.2 episodic path, end to end.

For each of the 5 Stage-4 quiz questions: retrieve top-1 memory from the
20-row store (memory_store.py), inject it into the prompt in three frozen
formats (I1 context prefix, I2 prior-QA turn, I3 knowledge grant), and
generate on a FRESH model (no state reload — that's the design).

Runs on the vast.ai 3090 stack (transformers 4.56.2 + fla 0.5.1, bf16).
Output: stage4b_results/retrieval_report.json + stdout log.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_store import build_store, FACT_MEMORIES

MODEL_REPO = "eivintobias/heartly-rwkv7-1.5b"
TOKENIZER_REPO = "RWKV/RWKV7-Goose-World3-1.5B-HF"
STOP_IDS = [61, 27081, 63]
MAX_NEW = 100
OUT_DIR = "stage4b_results"


def i1_context(memory, q):
    return f"Context: {memory}\nUser: {q}\nAssistant:"


def i2_prior_qa(memory, q):
    # memory carried as a prior grammar-complete QA turn (W2's format)
    return (f"User: What do you remember about this?\nAssistant: "
            f"<think>I have this in my memory.</think>"
            f"<decide>speak</decide><verify>known</verify>{memory}<stop>\n"
            f"User: {q}\nAssistant:")


def i3_grant(memory, q):
    return f"You know the following: {memory}\nUser: {q}\nAssistant:"


INJECTIONS = [("I1_context_prefix", i1_context),
              ("I2_prior_qa_turn", i2_prior_qa),
              ("I3_knowledge_grant", i3_grant)]


@torch.no_grad()
def greedy(model, prompt_ids, max_new=MAX_NEW):
    cache = None
    cur = prompt_ids
    out_ids = []
    for _ in range(max_new):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = int(out.logits[0, -1, :].argmax().item())
        out_ids.append(nxt)
        cur = torch.tensor([[nxt]], device=prompt_ids.device,
                           dtype=prompt_ids.dtype)
        if len(out_ids) >= len(STOP_IDS) and out_ids[-len(STOP_IDS):] == STOP_IDS:
            break
    return out_ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"
    store = build_store()

    # retrieval first (also re-verifies the local gate on this machine)
    retrievals = {}
    for f in FACT_MEMORIES:
        score, mem = store.retrieve(f["question"], k=1)[0]
        retrievals[f["key"]] = {"memory": mem, "score": score,
                                "retrieval_hit": mem == f["memory"]}
    r_hits = sum(v["retrieval_hit"] for v in retrievals.values())
    print(f"retrieval backend={store.backend} | top-1 {r_hits}/5", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device)
    model.eval()

    report = {"model": MODEL_REPO, "retrieval_backend": store.backend,
              "retrieval_top1": r_hits, "retrievals": retrievals,
              "injections": {}}
    for name, builder in INJECTIONS:
        print(f"\n== {name} ==", flush=True)
        hits = 0
        rows = []
        for f in FACT_MEMORIES:
            mem = retrievals[f["key"]]["memory"]
            prompt = builder(mem, f["question"])
            ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]
            out_ids = greedy(model, ids)
            txt = tok.decode(out_ids, skip_special_tokens=False)
            hit = f["gold"] in txt.lower()
            hits += int(hit)
            rows.append({"key": f["key"], "question": f["question"],
                         "memory": mem, "output": txt.strip(), "hit": hit})
            print(f"-- {f['question']}\n   {txt.strip()[:130]!r} | hit {hit}",
                  flush=True)
        print(f"{name}: {hits}/5", flush=True)
        report["injections"][name] = {"hits": hits, "rows": rows}

    print("\n==== SUMMARY ====")
    print(f"retrieval top-1: {r_hits}/5")
    for name, r in report["injections"].items():
        print(f"{name}: {r['hits']}/5")
    with open(os.path.join(OUT_DIR, "retrieval_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_DIR}/retrieval_report.json")


if __name__ == "__main__":
    main()
