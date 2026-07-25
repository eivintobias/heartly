#!/usr/bin/env python3
"""
fit_critic.py — Stage 3.6: fitted critics on the generator's own distribution.

Pre-registered in PREREG_STAGE3P6.md (frozen 2026-07-25 before any run).
No new model runs: reads the CACHED Stage-3.5 critic-B features
(stage3_critic_results/features_B_rwkv_late.npz) and the labeled rows
(critic_data_rwkv7_final.jsonl). Split / threshold / bar identical to
train_critic.py (seed 0, tracked -> test, stratified 80/20; threshold = 5th
percentile of train correct scores; bar = det >= 0.70 @ FPR <= 0.05 AND
tracked >= 4/5).

Selection rule: layer + hyperparameters by TRAIN 5-fold CV AUROC only; test
is touched once per method. For the baseline we ALSO print the lab-convention
test-best-layer AUROC (comparability with Stage 3.5's 0.835 — NOT the number
the bar applies to).
"""
import json
import os

import numpy as np


# --- verbatim protocol copies from train_critic.py (kept dependency-free) ---

def load_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r for r in rows if r.get("label") is not None]


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


# --- method zoo (frozen list from PREREG_STAGE3P6.md) ---

def _lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.1))


def _mlp(layers):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=layers, max_iter=300,
                                       early_stopping=True, random_state=0))


def _gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=0, max_iter=150,
                                          max_leaf_nodes=15)


def _calibrated():
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(_lr(), method="sigmoid", cv=5)


def _cv_auroc(make_clf, X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(make_clf(), X, y, cv=cv, scoring="roc_auc",
                             n_jobs=-1)
    return float(scores.mean())


def fit_method(name, Xtr, ytr):
    """Returns (score_fn, info) where score_fn(X)->P(correct) for X [N,L,S]."""
    n_layers = Xtr.shape[1]

    def concat(X):
        return X.reshape(X.shape[0], -1)

    if name == "logreg-perlayer":
        aucs = [_cv_auroc(_lr, Xtr[:, li, :], ytr) for li in range(n_layers)]
        li = int(np.argmax(aucs))
        clf = _lr().fit(Xtr[:, li, :], ytr)
        info = {"cv_auroc_per_layer": [round(a, 4) for a in aucs],
                "cv_selected_layer_slot": li}
        return (lambda X: clf.predict_proba(X[:, li, :])[:, 1]), info

    if name == "logreg-concat":
        clf = _lr().fit(concat(Xtr), ytr)
        info = {"cv_auroc": None,
                "cv_note": "no hyperparameters to select — CV skipped (PREREG selection rule has nothing to decide)"}
        return (lambda X: clf.predict_proba(concat(X))[:, 1]), info

    if name == "mlp-perlayer":
        aucs = [_cv_auroc(lambda: _mlp((64,)), Xtr[:, li, :], ytr)
                for li in range(n_layers)]
        li = int(np.argmax(aucs))
        clf = _mlp((64,)).fit(Xtr[:, li, :], ytr)
        info = {"cv_auroc_per_layer": [round(a, 4) for a in aucs],
                "cv_selected_layer_slot": li}
        return (lambda X: clf.predict_proba(X[:, li, :])[:, 1]), info

    if name == "mlp-concat":
        clf = _mlp((128, 32)).fit(concat(Xtr), ytr)
        info = {"cv_auroc": None,
                "cv_note": "no hyperparameters to select — CV skipped (PREREG selection rule has nothing to decide)"}
        return (lambda X: clf.predict_proba(concat(X))[:, 1]), info

    if name == "gbm-concat":
        clf = _gbm().fit(concat(Xtr), ytr)
        info = {"cv_auroc": None,
                "cv_note": "no hyperparameters to select — CV skipped (PREREG selection rule has nothing to decide)"}
        return (lambda X: clf.predict_proba(concat(X))[:, 1]), info

    if name == "logreg-calibrated":
        aucs = [_cv_auroc(_lr, Xtr[:, li, :], ytr) for li in range(n_layers)]
        li = int(np.argmax(aucs))
        clf = _calibrated().fit(Xtr[:, li, :], ytr)
        info = {"cv_auroc_per_layer": [round(a, 4) for a in aucs],
                "cv_selected_layer_slot": li}
        return (lambda X: clf.predict_proba(X[:, li, :])[:, 1]), info

    raise ValueError(f"unknown method {name}")


# --- evaluation (mirrors train_critic.train_eval metric definitions) ---

def evaluate(name, score_fn, Xtr, Xte, ytr, yte, te_rows):
    from sklearn.metrics import roc_auc_score

    s_tr = score_fn(Xtr)
    s_te = score_fn(Xte)
    auc = float(roc_auc_score(yte, s_te))

    out = {"name": name, "auroc": auc}

    # deployed points at several nominal FPR targets (thresholds from TRAIN)
    for tgt in (0.01, 0.05, 0.10, 0.25):
        thr = float(np.percentile(s_tr[ytr == 1], 100 * tgt))
        flags = s_te < thr
        conf = yte == 0
        corr = yte == 1
        out[f"det@{int(tgt*100)}"] = float(flags[conf].mean()) if conf.any() else None
        out[f"fpr@{int(tgt*100)}"] = float(flags[corr].mean()) if corr.any() else None
        if tgt == 0.05:
            out["threshold"] = thr
            out["detection"] = out["det@5"]
            out["false_flag"] = out["fpr@5"]
            tracked_idx = [i for i, r in enumerate(te_rows) if r["tracked"]]
            out["tracked_n"] = len(tracked_idx)
            out["tracked_caught"] = sum(int(flags[i]) for i in tracked_idx)
            out["tracked_detail"] = [
                {"id": te_rows[i]["id"], "p_correct": float(s_te[i]),
                 "flagged": bool(flags[i])} for i in tracked_idx]

    # medians per class
    conf = yte == 0
    corr = yte == 1
    tracked_mask = np.array([r["tracked"] for r in te_rows])
    out["median_p_correct"] = {
        "corrects": float(np.median(s_te[corr])) if corr.any() else None,
        "confabs": float(np.median(s_te[conf])) if conf.any() else None,
        "tracked5": float(np.median(s_te[tracked_mask])) if tracked_mask.any() else None,
    }

    # bottom-k purity (fraction confab in lowest-scoring k% of test)
    order = np.argsort(s_te)
    base = float(conf.mean())
    for k in (5, 10, 25):
        n_k = max(1, int(round(len(s_te) * k / 100)))
        bottom = order[:n_k]
        out[f"bottom{k}_purity"] = float(conf[bottom].mean())
    out["base_rate_confab"] = base

    ok = (out["detection"] is not None and out["detection"] >= 0.70
          and out["false_flag"] <= 0.05 and out["tracked_caught"] >= 4)
    out["pass"] = bool(ok)
    return out


ALL_METHODS = ["logreg-perlayer", "logreg-concat", "mlp-perlayer",
               "mlp-concat", "gbm-concat", "logreg-calibrated"]
DATA = "critic_data_rwkv7_final.jsonl"
FEATS = os.path.join("stage3_critic_results", "features_B_rwkv_late.npz")
OUT_DIR = "stage3p6_fitted_results"


def _load():
    rows = load_rows(DATA)
    rows.sort(key=lambda r: r["id"])
    ids = np.array([r["id"] for r in rows])
    y = np.array([r["label"] for r in rows])
    z = np.load(FEATS)
    assert z["ids"].tolist() == ids.tolist(), "feature/row id mismatch"
    X = z["X"]
    train_rows, test_rows = split_rows(rows, 0)
    tr_ids = {r["id"] for r in train_rows}
    tr_mask = np.array([r["id"] in tr_ids for r in rows])
    te_mask = ~tr_mask
    print(f"{len(rows)} labeled rows, features {X.shape}; "
          f"split {len(train_rows)} train / {len(test_rows)} test")
    return rows, X, y, tr_mask, te_mask, train_rows, test_rows


def _print_res(res):
    print(f"  AUROC {res['auroc']:.3f} | "
          f"det@5 {res['det@5']:.3f} | fpr@5 {res['fpr@5']:.3f} | "
          f"tracked {res['tracked_caught']}/{res['tracked_n']} | "
          f"{'PASS' if res['pass'] else 'FAIL'}")
    print(f"  det@1/10/25: {res['det@1']:.3f} / {res['det@10']:.3f} / "
          f"{res['det@25']:.3f} | medians "
          f"corr {res['median_p_correct']['corrects']:.3f} "
          f"conf {res['median_p_correct']['confabs']:.3f} "
          f"trk {res['median_p_correct']['tracked5']:.3f} | "
          f"bottom10 purity {res['bottom10_purity']:.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="",
                    help="comma subset of the frozen method list (default: all)")
    ap.add_argument("--merge", action="store_true",
                    help="merge per-method JSONs into fit_report.json and print summary")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.merge:
        report = {"data": DATA, "features": FEATS,
                  "pass_bar": "detection >= 0.70 & false_flag <= 0.05 & tracked >= 4/5",
                  "methods": []}
        for name in ALL_METHODS:
            p = os.path.join(OUT_DIR, f"fit_report_{name}.json")
            if not os.path.exists(p):
                print(f"MISSING: {name} ({p})")
                continue
            res = json.load(open(p, encoding="utf-8"))
            report["methods"].append(res)
            print(f"\n== {name} ==")
            _print_res(res)
        with open(os.path.join(OUT_DIR, "fit_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nmerged {len(report['methods'])} methods -> "
              f"{os.path.join(OUT_DIR, 'fit_report.json')}")
        return

    methods = args.methods.split(",") if args.methods else ALL_METHODS
    for name in methods:
        assert name in ALL_METHODS, f"{name} not in frozen list {ALL_METHODS}"

    rows, X, y, tr_mask, te_mask, train_rows, test_rows = _load()
    Xtr, Xte = X[tr_mask], X[te_mask]
    ytr, yte = y[tr_mask], y[te_mask]

    if "logreg-perlayer" in methods:
        # lab-convention baseline for comparability with Stage 3.5 (best TEST
        # layer AUROC for per-layer logreg) — printed only, bar does not apply.
        from sklearn.metrics import roc_auc_score
        conv = []
        for li in range(X.shape[1]):
            clf = _lr().fit(Xtr[:, li, :], ytr)
            conv.append(round(float(roc_auc_score(
                yte, clf.predict_proba(Xte[:, li, :])[:, 1])), 4))
        print("lab-convention test-best-layer logreg AUROC per slot: "
              + ", ".join(f"{a:.3f}" for a in conv))

    for name in methods:
        out_path = os.path.join(OUT_DIR, f"fit_report_{name}.json")
        print(f"\n== {name} ==", flush=True)
        score_fn, info = fit_method(name, Xtr, ytr)
        res = evaluate(name, score_fn, Xtr, Xte, ytr, yte, test_rows)
        res.update(info)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        _print_res(res)
        print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
