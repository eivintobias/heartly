#!/usr/bin/env python3
"""
stage4_gist.py — Stage 4: save/load the RWKV7 recurrent state across sessions.
Pre-registered in PREREG_STAGE4.md. Runs ON the vast.ai instance (Linux:
bf16 + fla 0.5.1 + transformers 4.56.2). Will NOT run on Windows (fla/triton).

Phases:
  1. TEACH  — forward a scripted 5-fact conversation, capture the fla Cache.
  2. SAVE   — dump per-layer state dicts to stage4_results/gist_state.pt.
  3. CHECK  — rebuild a fresh cache from disk, compare next-token logits
              vs the original cache (frozen bar: cosine >= 0.999 + argmax).
  4. QUIZ   — baseline (no state) vs gist-loaded answers to 5 fact questions
              (frozen bars: baseline 0/5, gist >= 4/5 + grammar parsed).

Output: stage4_results/gist_state.pt + stage4_results/stage4_report.json.
"""
import json
import os
import traceback

import torch

MODEL_REPO = "eivintobias/heartly-rwkv7-1.5b"
# established gotcha: the fine-tuned dir's saved tokenizer is broken — load
# the tokenizer from the base repo (same world vocab as rwkv-4).
TOKENIZER_REPO = "RWKV/RWKV7-Goose-World3-1.5B-HF"
STOP_IDS = [61, 27081, 63]  # '<stop>' in the RWKV world vocab
MAX_NEW = 120

ACK = ("<think>The user shared a personal fact.</think>"
       "<decide>speak</decide><verify>known</verify>Noted, thank you.<stop>")

# FABRICATED facts — unknowable to any baseline (see PREREG_STAGE4.md).
FACTS = [
    {"key": "dog",      "teach": "My dog's name is Zorblax.",
     "question": "What is my dog's name?", "gold": "zorblax"},
    {"key": "codename", "teach": "The secret project codename is Velvet Aurora.",
     "question": "What is the secret project codename?", "gold": "velvet aurora"},
    {"key": "number",   "teach": "My favorite number is 7,423.",
     "question": "What is my favorite number?", "gold": "7,423"},
    {"key": "collect",  "teach": "I collect miniature lighthouse replicas.",
     "question": "What do I collect?", "gold": "lighthouse"},
    {"key": "password", "teach": "The password to the lab door is mango Tuesday.",
     "question": "What is the lab door password?", "gold": "mango tuesday"},
]


def build_transcript():
    parts = []
    for f in FACTS:
        parts.append(f"User: {f['teach']}\nAssistant: {ACK}\n")
    return "".join(parts)


@torch.no_grad()
def forward_ids(model, ids, cache=None):
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return out, out.past_key_values


def extract_state(cache):
    """Per-layer state dicts -> CPU tensor clones."""
    layers = getattr(cache, "layers", None)
    assert layers is not None, "cache has no .layers"
    saved = []
    for i, layer in enumerate(layers):
        st = getattr(layer, "state", None)
        if not (isinstance(st, dict) and st):
            raise RuntimeError(f"layer {i}: expected state dict, got {type(st)}")
        saved.append({k: v.detach().cpu().clone() for k, v in st.items()
                      if torch.is_tensor(v)})
    return saved


def reload_cache(model, tok, saved, device):
    """Fresh fla Cache (dummy warm-up) with every layer's state overwritten."""
    warm_ids = tok("warm up", return_tensors="pt").to(device)["input_ids"]
    with torch.no_grad():
        out = model(input_ids=warm_ids, use_cache=True)
    cache = out.past_key_values
    layers = getattr(cache, "layers", None)
    n = 0 if layers is None else len(layers)
    assert layers is not None and n >= len(saved), \
        f"warm cache has {n} layers, need {len(saved)}"
    for i, st in enumerate(saved):
        cur = getattr(layers[i], "state", None)
        assert isinstance(cur, dict), f"layer {i} has no state dict"
        for k, v in st.items():
            cur[k] = v.to(device)
    return cache


@torch.no_grad()
def greedy(model, prompt_ids, cache, max_new=MAX_NEW):
    """Manual greedy loop — single-prompt only (batching corrupts RWKV state)."""
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
    return out_ids, cache


def main():
    os.makedirs("stage4_results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"model": MODEL_REPO, "phases": {}}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device)
    model.eval()
    print(f"loaded {MODEL_REPO} | {model.config.num_hidden_layers} layers | "
          f"device {device}", flush=True)

    # ---- 1. TEACH -------------------------------------------------------
    print("\n== 1. TEACH ==", flush=True)
    transcript = build_transcript()
    print(transcript, flush=True)
    ids = tok(transcript, return_tensors="pt").to(device)["input_ids"]
    out, gist = forward_ids(model, ids)
    print(f"transcript tokens: {ids.shape[1]} | cache layers: "
          f"{len(gist.layers)}", flush=True)
    ref_logits = out.logits[0, -1, :].float().cpu()

    # ---- 2. SAVE --------------------------------------------------------
    print("\n== 2. SAVE ==", flush=True)
    saved = extract_state(gist)
    path = os.path.join("stage4_results", "gist_state.pt")
    torch.save(saved, path)
    mb = os.path.getsize(path) / 1e6
    shapes = {k: list(v.shape) for k, v in saved[0].items()}
    print(f"saved {len(saved)} layer state dicts -> {path} ({mb:.1f} MB)")
    print(f"layer0 state keys/shapes: {shapes}", flush=True)
    report["phases"]["save"] = {"path": path, "mb": mb,
                                "n_layers": len(saved), "layer0": shapes}

    # ---- 3. MECHANICAL CHECK --------------------------------------------
    print("\n== 3. MECHANICAL CHECK ==", flush=True)
    saved_disk = torch.load(path)
    fresh = reload_cache(model, tok, saved_disk, device)
    # same-path comparison: feed the SAME 8-token tail through BOTH the
    # original gist cache and the disk-reloaded cache. (The first version
    # compared a 257-token chunked prefill against an 8-token continuation —
    # bf16 chunk-boundary noise, cosine 0.9977 argmax-flip, and was replaced
    # in-session per PREREG_STAGE4's decision rule; same-path isolates the
    # reload itself: see stage4_check2.py, cosine 1.00001.)
    tail = ids[:, -8:]
    out_ref, _ = forward_ids(model, tail, gist)
    out2, _ = forward_ids(model, tail, fresh)
    ref_logits = out_ref.logits[0, -1, :].float().cpu()
    rl = out2.logits[0, -1, :].float().cpu()
    cos = torch.nn.functional.cosine_similarity(ref_logits, rl, dim=0).item()
    amatch = int(ref_logits.argmax()) == int(rl.argmax())
    mech_pass = cos >= 0.999 and amatch
    print(f"next-token logits: cosine {cos:.5f} | argmax match: {amatch} | "
          f"{'PASS' if mech_pass else 'FAIL'}", flush=True)
    report["phases"]["mech"] = {"cosine": cos, "argmax_match": bool(amatch),
                                "pass": bool(mech_pass)}
    if not mech_pass:
        print("reload path broken — stopping before quiz "
              "(PREREG decision rule: frozen bar does not move)")
        with open("stage4_results/stage4_report.json", "w",
                  encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return

    # ---- 4. QUIZ --------------------------------------------------------
    print("\n== 4. QUIZ ==", flush=True)
    quiz = []
    for f in FACTS:
        qprompt = f"User: {f['question']}\nAssistant:"
        qids = tok(qprompt, return_tensors="pt").to(device)["input_ids"]

        # (a) baseline: fresh model, no state
        b_ids, _ = greedy(model, qids, None)
        b_text = tok.decode(b_ids, skip_special_tokens=False)

        # (b) gist: fresh reload from disk — "new session, same memory"
        gcache = reload_cache(model, tok, saved_disk, device)
        g_ids, _ = greedy(model, qids, gcache)
        g_text = tok.decode(g_ids, skip_special_tokens=False)

        b_hit = f["gold"] in b_text.lower()
        g_hit = f["gold"] in g_text.lower()
        g_parsed = "<verify>" in g_text
        quiz.append({"key": f["key"], "question": f["question"],
                     "gold": f["gold"],
                     "baseline_text": b_text, "baseline_hit": bool(b_hit),
                     "gist_text": g_text, "gist_hit": bool(g_hit),
                     "gist_parsed": bool(g_parsed)})
        print(f"\n-- {f['question']}")
        print(f"   baseline: {b_text.strip()[:140]!r}")
        print(f"   gist    : {g_text.strip()[:140]!r}")
        print(f"   hits: baseline {b_hit} | gist {g_hit} | parsed {g_parsed}",
              flush=True)

    b_n = sum(q["baseline_hit"] for q in quiz)
    g_n = sum(q["gist_hit"] for q in quiz)
    g_p = sum(q["gist_parsed"] for q in quiz)
    verdict = bool(mech_pass and b_n == 0 and g_n >= 4)
    report["phases"]["quiz"] = quiz
    report["summary"] = {"baseline_hits": b_n, "gist_hits": g_n,
                         "gist_parsed": g_p,
                         "VERDICT": "PASS" if verdict else "FAIL"}
    print(f"\n==== SUMMARY ====\nbaseline {b_n}/5 | gist {g_n}/5 "
          f"(parsed {g_p}/5) | mech {'PASS' if mech_pass else 'FAIL'} | "
          f"VERDICT: {'PASS — state persists' if verdict else 'FAIL'}",
          flush=True)
    with open("stage4_results/stage4_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("report -> stage4_results/stage4_report.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()