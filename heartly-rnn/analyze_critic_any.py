#!/usr/bin/env python3
"""analyze_critic_any.py — operating-curve readout for ANY critic run folder.

Same analysis as analyze_critic.py (Stage 2.5) but parameterized:
  --data     critic jsonl (default critic_data.jsonl)
  --out      run folder with features_<name>.npz + critic_<name>.pkl
  --names    comma-separated critic names (default A_qwen_transcript)
  --seed     split seed (default 0, must match the train_critic.py run)

Reports detection at fixed test-ROC FPR budgets (1/5/10/25%), median
P(correct) per class, tracked-5 scores, and bottom-k purity.
"""
import argparse
import pickle

import numpy as np
from sklearn.metrics import roc_curve

from train_critic import load_rows, split_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="critic_data.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="A_qwen_transcript")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.data)
    rows.sort(key=lambda r: r["id"])
    ids = np.array([r["id"] for r in rows])
    y = np.array([r["label"] for r in rows])
    train_rows, test_rows = split_rows(rows, args.seed)
    tr_ids = {r["id"] for r in train_rows}
    te_mask = np.array([r["id"] not in tr_ids for r in rows])
    yte = y[te_mask]

    for name in args.names.split(","):
        z = np.load(f"{args.out}/features_{name}.npz")
        assert z["ids"].tolist() == ids.tolist(), f"id mismatch for {name}"
        with open(f"{args.out}/critic_{name}.pkl", "rb") as f:
            pk = pickle.load(f)
        li, clf = pk["layer_slot"], pk["clf"]
        s = clf.predict_proba(z["X"][te_mask][:, li, :])[:, 1]

        fpr, tpr, _ = roc_curve(yte, s)
        print(f"\n== {name} (layer {pk['layers'][li]}, repo {pk['repo']}) ==")
        for target in (0.01, 0.05, 0.10, 0.25):
            ok = fpr <= target
            det = float(tpr[ok].max()) if ok.any() else 0.0
            print(f"  detection @ FPR<={target:.0%}: {det:.3f}")

        conf, corr = s[yte == 0], s[yte == 1]
        tracked_s = np.array([s[i] for i, r in enumerate(test_rows) if r["tracked"]])
        print(f"  median P(correct): corrects={np.median(corr):.3f} "
              f"confabs={np.median(conf):.3f} tracked5={np.median(tracked_s):.3f}")
        print(f"  tracked5 scores: {np.round(tracked_s, 3).tolist()}")

        order = np.argsort(s)
        for frac in (0.05, 0.10, 0.25):
            k = int(len(s) * frac)
            prec = float((yte[order[:k]] == 0).mean())
            print(f"  bottom {frac:.0%} of scores: {prec:.3f} confab fraction "
                  f"(base rate {float((yte == 0).mean()):.3f})")


if __name__ == "__main__":
    main()