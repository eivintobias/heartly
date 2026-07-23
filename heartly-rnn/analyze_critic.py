#!/usr/bin/env python3
"""analyze_critic.py — operating-curve analysis for the Stage 2.5 critics.

Recomputes the train/test split from train_critic.py (seed 0, tracked forced
into test), loads cached features + trained critic pkls, and reports:
  * detection rate at fixed FPRs (1%, 5%, 10%, 25%) from the TEST ROC
  * score stats per class (median P(correct) for corrects / confabs / tracked)
  * per-generator detection at the 5% point
This is the honest read of "how usable is the critic at each false-flag
budget" — the single pre-registered threshold is noisy at n_correct=48 train.
"""
import json
import pickle

import numpy as np
from sklearn.metrics import roc_curve

from train_critic import load_rows, split_rows

DATA = "critic_data.jsonl"
OUT = "stage2p5_results"

rows = load_rows(DATA)
rows.sort(key=lambda r: r["id"])
ids = np.array([r["id"] for r in rows])
y = np.array([r["label"] for r in rows])
train_rows, test_rows = split_rows(rows, 0)
tr_ids = {r["id"] for r in train_rows}
te_mask = np.array([r["id"] not in tr_ids for r in rows])
yte = y[te_mask]

for name in ("A_qwen_transcript", "B_rwkv_late"):
    z = np.load(f"{OUT}/features_{name}.npz")
    assert z["ids"].tolist() == ids.tolist()
    with open(f"{OUT}/critic_{name}.pkl", "rb") as f:
        pk = pickle.load(f)
    li, clf = pk["layer_slot"], pk["clf"]
    s = clf.predict_proba(z["X"][te_mask][:, li, :])[:, 1]

    fpr, tpr, _ = roc_curve(yte, s)
    print(f"\n== {name} (layer {pk['layers'][li]}) ==")
    for target in (0.01, 0.05, 0.10, 0.25):
        ok = fpr <= target
        det = float(tpr[ok].max()) if ok.any() else 0.0
        print(f"  detection @ FPR<={target:.0%}: {det:.3f}")

    conf, corr = s[yte == 0], s[yte == 1]
    tracked_s = np.array([s[i] for i, r in enumerate(test_rows) if r["tracked"]])
    print(f"  median P(correct): corrects={np.median(corr):.3f} "
          f"confabs={np.median(conf):.3f} tracked5={np.median(tracked_s):.3f}")
    print(f"  tracked5 scores: {np.round(tracked_s, 3).tolist()}")

    # how often is a random confab ranked below a random correct (AUROC check)
    # + top-k precision: among the k lowest-scored, how many are confabs
    order = np.argsort(s)
    for frac in (0.05, 0.10, 0.25):
        k = int(len(s) * frac)
        prec = float((yte[order[:k]] == 0).mean())
        print(f"  bottom {frac:.0%} of scores: {prec:.3f} confab fraction "
              f"(base rate {float((yte == 0).mean()):.3f})")