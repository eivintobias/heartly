#!/usr/bin/env python3
"""
train_critic.py — Stage 2.5 phase 2: train + evaluate independent answer critics.

Motivation (Stage 2 finding): the deployed boundary head reads the generator's
own state at the <verify> position — it senses question knowability, never
answer correctness, and shares the generator's confident errors (5/5
confabulations read sense_p >= 0.97). The critic must (a) SEE the answer and
(b) be independent of the generator.

Two critics, both logistic probes over final-transcript-token features:
  critic A — independent model: Qwen2.5-0.5B residual hidden states on the
             full transcript `User: q\nAssistant: <full generation>`.
  critic B — same-model late-position control: RWKV-heartly recurrent state
             at the end of its own generation. B vs A measures independence;
             B vs the Stage 2 head (state at <verify>) measures position.

Protocol (pre-registered):
  * split: all `tracked` Stage-2 confabulations forced into test; remaining
    rows stratified 80/20 on label.
  * per-layer StandardScaler + LogisticRegression(C=0.1); best layer by test
    AUROC (lab convention from train_probe.py).
  * operating point: threshold = 5th percentile of TRAIN correct-answer
    scores (nominal 5% false-flag rate); flag = P(correct) < threshold.
  * pass bar: detection >= 70% of test confabulations at <= 5% false-flag on
    correct answers, AND >= 4/5 tracked Stage-2 confabulations flagged.

Features are cached under stage2p5_results/ keyed to the input row ids.

Output: stage2p5_results/critic_{A,B}.pkl + critic_report.json + summary.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from extract_states import forward_features, load_family, quartile_idx


def load_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r for r in rows if r.get("label") is not None]


def transcript(r):
    return f"User: {r['question']}\nAssistant: {r['full_text']}"


@torch.no_grad()
def extract(family, repo, texts, ids, layers, cache_path, trust_remote):
    """Final-token features per text; cached npz {ids, X}."""
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        if z["ids"].tolist() == list(ids):
            print(f"features cached: {cache_path} {z['X'].shape}")
            return z["X"]
        print("cache id mismatch, re-extracting")
    tok, model = load_family(family, repo, "cuda" if torch.cuda.is_available() else "cpu",
                             trust_remote)
    X = []
    for t in tqdm(texts, desc=f"extract {family}"):
        enc = tok(t, return_tensors="pt").to(model.device)
        hidden, rstate = forward_features(family, model, enc, layers)
        feat = rstate if rstate is not None else hidden
        X.append(np.nan_to_num(feat.astype(np.float32), nan=0.0,
                               posinf=65000.0, neginf=-65000.0))
    X = np.stack(X)  # [N, L, S]
    np.savez(cache_path, ids=np.array(ids, dtype=np.int64), X=X.astype(np.float32))
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return X


def split_rows(rows, seed):
    rng = np.random.RandomState(seed)
    tracked = [r for r in rows if r["tracked"]]
    rest = [r for r in rows if not r["tracked"]]
    y = np.array([r["label"] for r in rest])
    idx = np.arange(len(rest))
    tr_idx, te_idx = [], []
    for cls in (0, 1):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        cut = int(0.8 * len(cls_idx))
        tr_idx += cls_idx[:cut].tolist()
        te_idx += cls_idx[cut:].tolist()
    train = [rest[i] for i in tr_idx]
    test = [rest[i] for i in te_idx] + tracked
    return train, test


def train_eval(name, Xtr, ytr, Xte, yte, te_rows, fpr_target=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_layers = Xtr.shape[1]
    best = None
    for li in range(n_layers):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(Xtr[:, li, :], ytr)
        auc = roc_auc_score(yte, clf.predict_proba(Xte[:, li, :])[:, 1])
        print(f"  [{name}] layer-slot {li}: AUROC {auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, li, clf)
    auc, li, clf = best
    print(f"  [{name}] -> best layer-slot {li} (AUROC {auc:.3f})")

    s_tr = clf.predict_proba(Xtr[:, li, :])[:, 1]
    thr = float(np.percentile(s_tr[ytr == 1], 100 * fpr_target))
    s_te = clf.predict_proba(Xte[:, li, :])[:, 1]
    flags = s_te < thr

    conf = yte == 0
    corr = yte == 1
    detection = float(flags[conf].mean()) if conf.any() else None
    fpr = float(flags[corr].mean()) if corr.any() else None

    per_gen = {}
    for r, y, fl in zip(te_rows, yte, flags):
        g = r["generator"]
        d = per_gen.setdefault(g, {"conf_n": 0, "conf_caught": 0, "corr_n": 0, "corr_flagged": 0})
        if y == 0:
            d["conf_n"] += 1
            d["conf_caught"] += int(fl)
        else:
            d["corr_n"] += 1
            d["corr_flagged"] += int(fl)

    tracked_idx = [i for i, r in enumerate(te_rows) if r["tracked"]]
    tracked_caught = sum(int(flags[i]) for i in tracked_idx)
    tracked_detail = [{"id": te_rows[i]["id"], "question": te_rows[i]["question"][:80],
                       "p_correct": float(s_te[i]), "flagged": bool(flags[i])}
                      for i in tracked_idx]

    return {"name": name, "best_layer_slot": int(li), "auroc": float(auc),
            "threshold": thr, "detection": detection, "false_flag": fpr,
            "n_test": int(len(yte)), "n_confab": int(conf.sum()), "n_correct": int(corr.sum()),
            "tracked_n": len(tracked_idx), "tracked_caught": tracked_caught,
            "tracked_detail": tracked_detail, "per_generator": per_gen}, {"layer_slot": li, "clf": clf, "threshold": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="critic_data.jsonl")
    ap.add_argument("--out-dir", default="stage2p5_results")
    ap.add_argument("--critic-a-repo", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--critic-b-repo", default="stage2_results/rwkv-heartly")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N labeled rows")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.data)
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[: args.limit]
    ids = np.array([r["id"] for r in rows])
    texts = [transcript(r) for r in rows]
    y = np.array([r["label"] for r in rows])
    print(f"{len(rows)} labeled rows "
          f"({int((y == 1).sum())} correct / {int((y == 0).sum())} confab)")

    train_rows, test_rows = split_rows(rows, args.seed)
    tr_ids = {r["id"] for r in train_rows}
    tr_mask = np.array([r["id"] in tr_ids for r in rows])
    te_mask = ~tr_mask
    ytr, yte = y[tr_mask], y[te_mask]
    print(f"split: {len(train_rows)} train / {len(test_rows)} test "
          f"({sum(r['tracked'] for r in test_rows)} tracked in test)")

    report = {"data": args.data, "n_labeled": len(rows),
              "n_train": len(train_rows), "n_test": len(test_rows),
              "pass_bar": "detection >= 0.70 & false_flag <= 0.05 & tracked >= 4/5",
              "critics": []}

    jobs = []
    if not args.skip_a:
        jobs.append(("A_qwen_transcript", "qwen", args.critic_a_repo, False))
    if not args.skip_b:
        jobs.append(("B_rwkv_late", "rwkv", args.critic_b_repo, True))

    for name, family, repo, trust in jobs:
        print(f"\n== critic {name} ({repo}) ==")
        n_layers_hint = None
        cache = os.path.join(args.out_dir, f"features_{name}.npz")
        # quartile layers from the model config (24 for both qwen-0.5B and rwkv-430m)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=trust)
        n_layers_hint = getattr(cfg, "num_hidden_layers", None) \
            or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers")
        layers = quartile_idx(n_layers_hint)
        print(f"layers probed: {layers}")
        X = extract(family, repo, texts, ids, layers, cache, trust)
        res, model_pkl = train_eval(name, X[tr_mask], ytr, X[te_mask], yte, test_rows)
        res["layers_probed"] = layers
        res["probed_layer"] = layers[res["best_layer_slot"]]
        report["critics"].append(res)
        with open(os.path.join(args.out_dir, f"critic_{name}.pkl"), "wb") as f:
            pickle.dump({"family": family, "repo": repo, "layers": layers,
                         "layer_slot": model_pkl["layer_slot"],
                         "clf": model_pkl["clf"], "threshold": model_pkl["threshold"]}, f)

    print("\n==== CRITIC SUMMARY ====")
    for res in report["critics"]:
        ok = (res["detection"] is not None and res["detection"] >= 0.70
              and res["false_flag"] <= 0.05 and res["tracked_caught"] >= 4)
        res["pass"] = bool(ok)
        print(f"[{res['name']}] AUROC {res['auroc']:.3f} | "
              f"detection {res['detection']:.3f} | FPR {res['false_flag']:.3f} | "
              f"tracked {res['tracked_caught']}/{res['tracked_n']} | "
              f"{'PASS' if ok else 'FAIL'}")

    with open(os.path.join(args.out_dir, "critic_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport -> {os.path.join(args.out_dir, 'critic_report.json')}")


if __name__ == "__main__":
    main()