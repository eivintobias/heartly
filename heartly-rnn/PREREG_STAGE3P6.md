# Pre-registration — Stage 3.6: fitted critic on the generator's own distribution

Written 2026-07-25, BEFORE any run. Signed off by the method: the method list,
selection rule, and bar below are frozen. Nothing outside this list will be
tried against the bar.

## Question

Stage 3.5 showed a generic per-layer logistic probe on the generator's own
state features RANKS confabulations well (AUROC 0.835) but cannot THRESHOLD
them (deployed 5%-FPR point false-flags 57–87% of correct answers; bar fails
at every scale and asymmetry tested). Is the threshold failure a FITTING
failure (the probe family / layer choice / calibration is wrong) or an
INFORMATION failure (the features simply do not separate the tails)?

- If any reasonable fitting method on the SAME features finds a working
  threshold → the problem was fitting; a fitted critic is the fix.
- If none do → the information is not in the features at the tails; the
  signature-critic threshold line closes and ranking becomes the product.

## Data and features (no new model runs)

- Labels: `critic_data_rwkv7_final.jsonl` — 1,480 labeled rows
  (229 correct / 1,251 confab_content; 0 confab_unknown at 1.5B;
  pre-registered tracked 5 = ids 0, 2, 3, 5, 7 from `pick_tracked.py`).
- Features: the CACHED Stage-3.5 critic-B features
  `stage3_critic_results/features_B_rwkv_late.npz` (RWKV7 recurrent state at
  end of own generation, 4 probed layer-slots), keyed to the same row ids.
  No re-extraction, no new features, no new generations.

## Protocol (identical to Stage 3.5 / train_critic.py)

- Split: seed 0; tracked rows forced into test; remaining stratified 80/20 on
  label; test n=301.
- Threshold: 5th percentile of TRAIN correct-answer scores (nominal 5% FPR);
  flag = P(correct) < threshold.
- Bar (unchanged): detection ≥ 70% of test confabs at ≤ 5% false-flag on
  correct answers, AND ≥ 4/5 tracked flagged.

## Methods (frozen list)

All fit on train only, on the same cached features:

1. `logreg-perlayer` — baseline replication (StandardScaler + LR C=0.1);
   expected to reproduce Stage 3.5's ≈0.835 AUROC. Sanity anchor.
2. `logreg-concat` — all 4 layer features concatenated, same LR.
3. `mlp-perlayer` — small MLP (hidden 64) per layer.
4. `mlp-concat` — MLP (128→32) on concatenated layers.
5. `gbm-concat` — histogram gradient boosting on concatenated layers.
6. `logreg-calibrated` — per-layer LR wrapped in sigmoid calibration
   (5-fold CV on train).

## Selection rule

Layer choice and every hyperparameter choice are made by TRAIN 5-fold
cross-validation AUROC only. The test set is touched once per method.
(For comparability with prior stages we will also PRINT the lab-convention
test-best-layer AUROC for the baseline — but the bar applies only to the
CV-selected configuration.)

## Reporting (per method)

AUROC; detection at the deployed 5%-FPR threshold; deployed false-flag rate;
tracked caught (of 5); detection at 1%/10%/25% FPR; median P(correct) per
class (corrects / confabs / tracked); bottom-5%/10%/25% score purity.
PASS/FAIL vs the bar. Machine record: `stage3p6_fitted_results/fit_report.json`.

## Decision rule (fixed)

- Any method PASSES → the threshold problem was a fitting failure.
  Follow-up: holdout-confirm on a fresh harvest before claiming; paper update.
- All FAIL at AUROC ≈ 0.83–0.86 → the information ceiling is in the features.
  The signature-critic THRESHOLD line closes; ranking-as-product (bottom-k
  review queue) becomes the critic deliverable; program moves to Stage 4.
- All FAIL at AUROC substantially ABOVE 0.86 (clearly better ranking, still
  no threshold) → mixed result; the ranking gain is recorded, the line still
  closes (the bar is the bar).