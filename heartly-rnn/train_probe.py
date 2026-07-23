#!/usr/bin/env python3
"""
train_probe.py — Train known/unknown probes on dumped states.

For every states/states_*.npz:
  - per-layer probes on hidden states (all models)
  - per-layer probes on recurrent state (mamba: ssm_state; rwkv: carried state)
Metrics: AUROC, accuracy, ECE (10-bin). Plus a per-generator AUROC breakdown
for each model's best feature. Appends a report section to RESULTS.md.

Usage:
  python train_probe.py                 # probe everything in states/
  python train_probe.py --verified-only # known side restricted to model-verified
"""
import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np


def ece(probs, labels, n_bins=10):
    """Expected calibration error, 10 equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            err += (m.sum() / len(probs)) * abs(probs[m].mean() - labels[m].mean())
    return float(err)


def probe(X, y, seed=0):
    """StandardScaler + LogisticRegression, stratified 80/20. Returns metrics + test info."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    # fp16 storage can overflow to inf (rwkv state magnitudes) — clamp for sklearn
    X = np.nan_to_num(X, nan=0.0, posinf=65000.0, neginf=-65000.0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.1))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {
        "auroc": float(roc_auc_score(yte, p)),
        "acc": float(accuracy_score(yte, p > 0.5)),
        "ece": ece(p, yte),
        "y_test": yte,
        "p_test": p,
    }


def load_labels(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["id"]: r for r in rows}


def run_file(npz_path, labels, verified_only, seed):
    from sklearn.metrics import roc_auc_score

    d = np.load(npz_path)
    tag = os.path.basename(npz_path).replace("states_", "").replace(".npz", "")
    ids = d["ids"]
    y = np.array([1 if labels[i]["label"] == "known" else 0 for i in ids])
    gens = np.array([labels[i]["generator"] for i in ids])

    mask = np.ones(len(ids), dtype=bool)
    note = ""
    if verified_only:
        ver = d["verified"] if "verified" in d else np.zeros(len(ids), dtype=bool)
        if ver.sum() == 0:
            note = "  (no verification data — using corpus labels)"
        else:
            # unknown labels are trusted by construction; filter known side
            mask = (y == 0) | ver
            note = f"  (known side: {int(((y == 1) & mask).sum())}/{int(y.sum())} model-verified)"

    report = [f"\n## {tag}  ({os.path.basename(npz_path)})", note,
              f"n = {int(mask.sum())} (known {int(((y == 1) & mask).sum())} / "
              f"unknown {int(((y == 0) & mask).sum())})", "",
              "| feature | layer | AUROC | acc | ECE |",
              "|---|---|---|---|---|"]

    best = {"auroc": -1.0}
    for kind in (["hidden", "rstate"] if d["rstate"].shape[-1] > 0 else ["hidden"]):
        F = d[kind]  # [N, L, D]
        for li in range(F.shape[1]):
            X = F[:, li, :].astype(np.float32)
            m = probe(X[mask], y[mask], seed)
            layer_n = int(d["layer_idx"][li])
            report.append(f"| {kind} | {layer_n} | {m['auroc']:.3f} | "
                          f"{m['acc']:.3f} | {m['ece']:.3f} |")
            if m["auroc"] > best["auroc"]:
                best = {"auroc": m["auroc"], "kind": kind, "layer": layer_n,
                        "y_test": m["y_test"], "p_test": m["p_test"]}

    # per-generator breakdown at the best feature
    if best["auroc"] > 0:
        # recompute test indices consistent with probe() split
        from sklearn.model_selection import train_test_split
        F = d[best["kind"]]
        li = list(d["layer_idx"]).index(best["layer"])
        X = F[:, li, :].astype(np.float32)
        idx = np.arange(len(ids))
        _, te_idx = train_test_split(idx[mask], test_size=0.2,
                                     random_state=seed, stratify=y[mask])
        gen_te = gens[te_idx]
        y_te, p_te = best["y_test"], best["p_test"]
        report += ["", f"mean P(known) per generator @ best ({best['kind']} layer {best['layer']}):",
                   "", "| generator | true label | mean P(known) | n |", "|---|---|---|---|"]
        for g in sorted(set(gens[mask])):
            m = gen_te == g
            if m.sum() >= 5:
                true_lab = "known" if y_te[m].mean() > 0.5 else "unknown"
                report.append(f"| {g} | {true_lab} | {p_te[m].mean():.3f} | {int(m.sum())} |")
    report.append("")
    return "\n".join(report), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", default="states")
    ap.add_argument("--input", default="probe_questions.jsonl")
    ap.add_argument("--verified-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default="RESULTS.md")
    args = ap.parse_args()

    labels = load_labels(args.input)
    files = sorted(glob.glob(os.path.join(args.states_dir, "states_*.npz")))
    if not files:
        raise SystemExit(f"no states_*.npz in {args.states_dir}/ — run extract_states.py first")

    header = [f"\n---\n# Probe run — {datetime.now():%Y-%m-%d %H:%M}",
              f"verified_only={args.verified_only} | split=80/20 stratified | "
              f"probe=LogisticRegression(C=0.1)\n"]
    summary = ["\n## Summary (best per model)", "",
               "| model | best feature | layer | AUROC |", "|---|---|---|---|"]

    sections = []
    for f in files:
        rep, best = run_file(f, labels, args.verified_only, args.seed)
        sections.append(rep)
        print(rep)
        if best["auroc"] > 0:
            summary.append(f"| {os.path.basename(f)[7:-4]} | {best['kind']} | "
                           f"{best['layer']} | {best['auroc']:.3f} |")

    out = "\n".join(header + sections + summary) + "\n"
    with open(args.results, "a", encoding="utf-8") as fh:
        fh.write(out)
    print(f"\nappended -> {args.results}")


if __name__ == "__main__":
    main()