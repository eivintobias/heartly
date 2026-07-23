#!/usr/bin/env python3
"""
measure_say_sense.py — Stage 2 measurement: deployed boundary head + say/sense.

Two phases:
  1. HEAD — train a logistic boundary head on the FINE-TUNED model's recurrent
     state at the `<verify>` position (teacher-forced prefixes), labels from
     fresh disjoint samples (known: train slices after the SFT slice; unknown:
     generators with a new seed). One head per candidate layer; best by AUROC.
  2. SAY/SENSE — on the held-out probe set (validation questions): greedy-
     generate, parse the emitted `<verify>known|unknown</verify>` ("say"),
     read the head at the model's own generated `<verify>` prefix ("sense").
     Disagreement = the hallucination alarm:
       say=known  & sense=unknown → confabulation CAUGHT (the money cell)
       say=unknown & sense=known  → over-refusal flagged

Output: say_sense_report.json + printed summary.
"""
import argparse
import json
import pickle
import random
import re

import numpy as np
import torch
from tqdm import tqdm

from extract_states import forward_features
from gen_probe_dataset import (gen_fabricated, gen_type_mismatch, gen_post_cutoff,
                               gen_depth2, gen_structural)
from render_sft_dataset import REASON_KNOWN, REASON_UNKNOWN

VERIFY_RE = re.compile(r"<verify>\s*(known|unknown)\s*</verify>")
HEAD_LAYERS = [6, 12, 18, 23]


def load_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                 trust_remote_code=True)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"model on {dev}")
    return tok, model


@torch.no_grad()
def state_at(tok, model, text, layers):
    enc = tok(text, return_tensors="pt").to(model.device)
    _, rstate = forward_features("rwkv", model, enc, layers)
    return rstate  # [len(layers), S] or None


def head_samples(rng, n_known, n_unknown):
    """Fresh disjoint labeled prefixes: (text, 1=known/0=unknown)."""
    from datasets import load_dataset
    rows = []

    def short(a):
        return a and len(a.split()) <= 6

    sq = load_dataset("rajpurkar/squad", split="train[30000:45000]")
    idx = list(range(len(sq)))
    rng.shuffle(idx)
    got = 0
    for i in idx:
        ex = sq[i]
        ans = ex.get("answers", {}).get("text", [""])
        ans = ans[0] if ans else ""
        if short(ans):
            rows.append((ex["question"].strip(), 1))
            got += 1
        if got >= n_known // 2:
            break
    tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train[30000:45000]")
    idx = list(range(len(tq)))
    rng.shuffle(idx)
    got = 0
    for i in idx:
        ex = tq[i]
        if short(ex.get("answer", {}).get("value", "")):
            rows.append((ex["question"].strip(), 1))
            got += 1
        if got >= n_known - n_known // 2:
            break

    per = n_unknown // 5
    unknown = (gen_fabricated(rng, per) + gen_type_mismatch(rng, per)
               + gen_post_cutoff(rng, per) + gen_depth2(rng, per)
               + gen_structural(rng, n_unknown - 4 * per))
    for q, _g, _ in unknown:
        rows.append((q, 0))

    out = []
    for q, lab in rows:
        reason = rng.choice(REASON_KNOWN if lab else REASON_UNKNOWN)
        text = f"User: {q}\nAssistant: <think> {reason} </think><decide>speak</decide><verify>"
        out.append((text, lab))
    return out


def train_head(tok, model, rng, n_known, n_unknown, device):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    samples = head_samples(rng, n_known, n_unknown)
    X, y = [], []
    for text, lab in tqdm(samples, desc="head states"):
        st = state_at(tok, model, text, HEAD_LAYERS)
        if st is None:
            continue
        X.append(np.nan_to_num(st.astype(np.float32), nan=0.0,
                               posinf=65000.0, neginf=-65000.0))
        y.append(lab)
    X, y = np.stack(X), np.array(y)  # X: [N, L, S]

    best = None
    for li, layer in enumerate(HEAD_LAYERS):
        Xtr, Xte, ytr, yte = train_test_split(X[:, li, :], y, test_size=0.2,
                                              random_state=0, stratify=y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(Xtr, ytr)
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        print(f"  head layer {layer}: AUROC {auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, layer, clf)
    print(f"  -> using layer {best[1]} (AUROC {best[0]:.3f})")
    return {"layer": best[1], "clf": best[2], "auroc": best[0]}


@torch.no_grad()
def generate(tok, model, prompt, max_new):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id
                         if tok.pad_token_id is not None else 0)
    text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    stop = text.find("<stop>")
    return text[:stop] if stop != -1 else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rwkv-heartly")
    ap.add_argument("--questions", default="probe_questions.jsonl")
    ap.add_argument("--head-samples", type=int, default=1200)
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=140)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--head-out", default="probe_head.pkl")
    ap.add_argument("--report", default="say_sense_report.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(args.model)
    rng = random.Random(args.seed)

    print("== phase 1: boundary head ==")
    head = train_head(tok, model, rng, args.head_samples // 2,
                      args.head_samples // 2, device)
    with open(args.head_out, "wb") as f:
        pickle.dump(head, f)

    print("== phase 2: say/sense ==")
    rows = [json.loads(l) for l in open(args.questions, encoding="utf-8")]
    rng.shuffle(rows)
    rows = rows[: args.eval_limit]

    results = []
    for r in tqdm(rows, desc="say/sense"):
        prompt = f"User: {r['question']}\nAssistant: "
        text = generate(tok, model, prompt, args.max_new)
        m = VERIFY_RE.search(text)
        say = m.group(1) if m else None
        prefix = text[: m.start()] + "<verify>" if m else text
        st = state_at(tok, model, prompt + prefix, [head["layer"]])
        sense = None
        if st is not None:
            X = np.nan_to_num(st[0].astype(np.float32).reshape(1, -1),
                              nan=0.0, posinf=65000.0, neginf=-65000.0)
            p = float(head["clf"].predict_proba(X)[0, 1])
            sense = "known" if p > 0.5 else "unknown"
        else:
            p = None
        results.append({"id": r["id"], "question": r["question"],
                        "true": r["label"], "generator": r["generator"],
                        "say": say, "sense": sense, "sense_p": p,
                        "text": text[:300]})

    n = len(results)
    parsed = [x for x in results if x["say"]]
    agree = [x for x in parsed if x["say"] == x["sense"]]
    conf_caught = [x for x in parsed if x["say"] == "known" and x["sense"] == "unknown"]
    overref = [x for x in parsed if x["say"] == "unknown" and x["sense"] == "known"]
    say_acc = np.mean([x["say"] == x["true"] for x in parsed]) if parsed else 0
    sense_acc = np.mean([x["sense"] == x["true"] for x in parsed if x["sense"]]) if parsed else 0

    summary = {
        "n": n, "parsed": len(parsed), "unparsed": n - len(parsed),
        "agreement": len(agree) / max(1, len(parsed)),
        "confabulations_caught": len(conf_caught),
        "overrefusals_flagged": len(overref),
        "say_accuracy_vs_true": float(say_acc),
        "sense_accuracy_vs_true": float(sense_acc),
        "head_layer": head["layer"], "head_auroc": head["auroc"],
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results,
                   "examples_caught": conf_caught[:10],
                   "examples_overrefusal": overref[:10]}, f,
                  ensure_ascii=False, indent=2)

    print("\n==== SAY/SENSE SUMMARY ====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()