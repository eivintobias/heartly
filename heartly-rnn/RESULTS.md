# RESULTS — Boundary Head on the State

Experiment log. Each `train_probe.py` run appends a dated section below.

Pre-registered decision rule (from README): proceed to Stage 2 (RNN fine-tune +
deployed head) only if recurrent-state probing reaches AUROC ≥ ~0.8 and is
competitive with or better than the Qwen transformer baseline.

<!-- probe runs append below this line -->
---
# Probe run — 2026-07-20 22:14
verified_only=False | split=80/20 stratified | probe=LogisticRegression(C=0.1)


## falcon_h1  (states_falcon_h1.npz)

n = 2902 (known 1486 / unknown 1416)

| feature | layer | AUROC | acc | ECE |
|---|---|---|---|---|
| hidden | 0 | 0.500 | 0.513 | 0.001 |
| hidden | 9 | 0.999 | 0.985 | 0.016 |
| hidden | 18 | 1.000 | 0.990 | 0.018 |
| hidden | 27 | 0.998 | 0.988 | 0.016 |
| hidden | 35 | 0.994 | 0.986 | 0.016 |
| rstate | 0 | 0.999 | 0.993 | 0.014 |
| rstate | 9 | 1.000 | 0.990 | 0.015 |
| rstate | 18 | 1.000 | 0.991 | 0.010 |
| rstate | 27 | 1.000 | 0.988 | 0.014 |
| rstate | 35 | 0.995 | 0.974 | 0.022 |

mean P(known) per generator @ best (hidden layer 18):

| generator | true label | mean P(known) | n |
|---|---|---|---|
| depth2 | unknown | 0.002 | 71 |
| fabricated | unknown | 0.009 | 84 |
| post_cutoff | unknown | 0.020 | 28 |
| squad | known | 0.952 | 128 |
| structural | unknown | 0.014 | 38 |
| trivia_qa | known | 0.989 | 170 |
| type_mismatch | unknown | 0.021 | 62 |


## qwen  (states_qwen.npz)

n = 2902 (known 1486 / unknown 1416)

| feature | layer | AUROC | acc | ECE |
|---|---|---|---|---|
| hidden | 0 | 0.500 | 0.513 | 0.001 |
| hidden | 6 | 0.997 | 0.978 | 0.016 |
| hidden | 12 | 0.998 | 0.990 | 0.019 |
| hidden | 18 | 0.997 | 0.985 | 0.011 |
| hidden | 23 | 0.998 | 0.978 | 0.015 |

mean P(known) per generator @ best (hidden layer 12):

| generator | true label | mean P(known) | n |
|---|---|---|---|
| depth2 | unknown | 0.004 | 71 |
| fabricated | unknown | 0.018 | 84 |
| post_cutoff | unknown | 0.026 | 28 |
| squad | known | 0.948 | 128 |
| structural | unknown | 0.005 | 38 |
| trivia_qa | known | 0.976 | 170 |
| type_mismatch | unknown | 0.028 | 62 |


## rwkv  (states_rwkv.npz)

n = 2902 (known 1486 / unknown 1416)

| feature | layer | AUROC | acc | ECE |
|---|---|---|---|---|
| hidden | 0 | 0.972 | 0.923 | 0.033 |
| hidden | 6 | 0.999 | 0.983 | 0.018 |
| hidden | 12 | 0.996 | 0.981 | 0.014 |
| hidden | 18 | 0.995 | 0.976 | 0.017 |
| hidden | 23 | 0.994 | 0.981 | 0.018 |
| rstate | 0 | 0.996 | 0.986 | 0.013 |
| rstate | 6 | 1.000 | 0.988 | 0.012 |
| rstate | 12 | 0.999 | 0.990 | 0.011 |
| rstate | 18 | 0.999 | 0.986 | 0.016 |
| rstate | 23 | 0.999 | 0.979 | 0.020 |

mean P(known) per generator @ best (rstate layer 6):

| generator | true label | mean P(known) | n |
|---|---|---|---|
| depth2 | unknown | 0.002 | 71 |
| fabricated | unknown | 0.002 | 84 |
| post_cutoff | unknown | 0.003 | 28 |
| squad | known | 0.952 | 128 |
| structural | unknown | 0.004 | 38 |
| trivia_qa | known | 0.992 | 170 |
| type_mismatch | unknown | 0.004 | 62 |


## Summary (best per model)

| model | best feature | layer | AUROC |
|---|---|---|---|
| falcon_h1 | hidden | 18 | 1.000 |
| qwen | hidden | 12 | 0.998 |
| rwkv | rstate | 6 | 1.000 |

---

# Conclusions — Run 1 (2026-07-20)

**The absence sensor is real. Both pre-registered falsifiers are avoided; the
decision rule for Stage 2 passes decisively.**

Headline numbers (test split, corpus labels, n=2902):

| model | arch | best feature | AUROC | acc | ECE |
|---|---|---|---|---|---|
| Falcon-H1-0.5B-Base | hybrid SSM+attn | rstate (layers 9/18/27) | **1.000** | 0.99 | 0.010–0.015 |
| RWKV-4-World-430m | pure RNN | rstate (layer 6) | **1.000** | 0.988 | 0.012 |
| Qwen2.5-0.5B | transformer (baseline) | hidden (layer 12) | 0.998 | 0.990 | 0.019 |

Findings:

1. **Knowability is readable from the recurrent state before any answer is
   generated** — with AUROC indistinguishable from 1.0 on mid layers, and
   already ≥0.995 at the *first* SSM layer (falcon rstate layer 0: 0.999).
2. **Recurrent state ≥ residual hidden state** in both recurrent models, and
   both ≥ the transformer baseline — exactly the ordering Track 2 hypothesized.
   Note the baseline is also strong: Kadavath-style latent self-knowledge
   replicates on Qwen2.5-0.5B (0.997–0.998).
3. **Calibration is excellent** (ECE 0.01–0.02) — probe probabilities are
   usable as confidence scores, not just rankings.
4. **Per-generator margins are instructive.** Unknown classes sit at mean
   P(known) ≤ 0.02 — including *depth2* (real-entity hyper-specifics, the
   sharpest boundary class) at 0.002–0.004. RWKV's state gives the cleanest
   margins (all unknown classes ≤ 0.004). On the known side, TriviaQA reads
   0.98–0.99 while SQuAD reads 0.95 — consistent with SQuAD's
   context-dependent nature (the model "owns" fewer of those answers).
5. **Shortcut check passes.** If the probe had learned "weird question →
   unknown", obscure-but-known trivia would score low; it scores 0.98+.
   The inoculation design (obscure real trivia on the known side) held.

Honest caveats:

- Labels are *by construction*: unknown questions are engineered to be outside
  the model's knowledge, so near-perfect separation is partly definitional.
  The harder validation is model-verified unknowns (questions the *model*
  fails, drawn from real distributions) — `--verify-known` run pending.
- depth2 could contain questions a bigger model would answer; at 0.4–0.5B the
  label noise is presumably low, and mean P(known) ≈ 0.002 supports that.
- The probe is linear on pooled/flattened state (fp16-clamped). A richer head
  (MLP, unpooled state) may add little — there is no headroom left to add it in.

**Decision: proceed to Stage 2** — fine-tune the RNN (RWKV first: cleanest
margins, smallest model, fastest CPU iteration) on the Heartly grammar + v4
data, then deploy the boundary head at the `<verify>` position and measure
say/sense disagreement.

---

# Stage 2 — RWKV fine-tune + deployed boundary head (2026-07-21)

**Config.** RWKV-4-World-430m, 6,031 Heartly-grammar SFT samples (kind-aware
rendering, true-boundary unknown mix, train splits), 2 epochs, lr 1e-4 cosine,
batch 4 × accum 4, max-len 256, 16/24 blocks frozen, fp32, ~6 h on an RTX
3090-class GPU (vast.ai, ~$2). Head: logistic on recurrent state at the
`<verify>` position, trained on 1,200 fresh disjoint teacher-forced samples.

**Grammar adoption: total.** 300/300 held-out probe questions (100%) produced
a parseable `<verify>known|unknown</verify>` decision — the RWKV emits the
full think→decide→verify grammar reliably after 2 epochs.

**Decision accuracy: 97.7%** (293/300 vs. true labels). Per-generator:

| generator | say accuracy |
|---|---|
| trivia_qa | 82/82 (100%) |
| depth2 | 31/31 (100%) |
| type_mismatch | 26/26 (100%) |
| squad | 76/78 (97%) |
| fabricated | 44/46 (96%) |
| post_cutoff | 20/21 (95%) |
| structural | 14/16 (88%) |

**Boundary head on the fine-tuned model: AUROC 1.000 at layer 6** — the
sensor survives fine-tuning and reads the deployed position cleanly.

**Say/sense agreement: 100%** — and that is the instructive part. Zero caught
confabulations, zero flagged over-refusals. Not because the model never
confabulates (5 confabulations occurred: 2 fabricated theory questions,
1 post-cutoff, 2 structural), but because **in all 5 the head agreed with the
confabulation (sense_p 0.97–1.00)**. The head and the generator read the same
state, so when the state confidently pattern-matches a wrong answer, both
channels fire together. Conclusion: say/sense disagreement as architected can
only catch *uncertain* confabulation; catching *confident* confabulation
needs an independent channel — the critic-model path (paper §5.4) moves from
shelved to motivated.

**Live confirmation of the paper's §3 boundary error.** The two squad
"over-refusals" (say=unknown on corpus-known questions: "second most abundant
element", "average biomass per hectare in the Amazon") had sense_p = 0.001 —
the model genuinely does not know these. The corpus label said *known*; the
model's boundary said *unknown*. The labels were wrong for this model —
exactly the KB-boundary ≠ model-boundary distinction the paper predicts, now
observed in the wild.

**Capability caveat.** The verify *decision* is decoupled from answer
*content*: correct known-decisions still carry weak answers from the 0.43B
base (Gawain instead of Bedivere; Brando instead of Bette Davis). The
mechanism works at this scale; content quality now tracks the base model —
arguing for the same recipe at 1.5–3B (RWKV7-Goose-1.5B is the candidate).

**Artifacts.** `stage2_results/rwkv-heartly/` (1.85 GB), `probe_head.pkl`
(layer 6), `say_sense_report.json` (300 rows), `analyze_report.py`.

**Decision: Stage 2 hypothesis confirmed with one amendment.** Deployed
boundary heads work (grammar 100%, decisions 97.7%, sensor AUROC 1.0). The
alarm channel needs independence — Stage 2.5 candidate: critic head trained
on *separate* features (or a second small model) so confident confabulation
becomes catchable. Then Stage 3 (state persistence) and Stage 4 (write-gate).

---

# Stage 2.5 — independent answer critic (pre-registered 2026-07-21, before results)

**Motivation.** Stage 2 showed the deployed head is blind to confident
confabulation for two structural reasons: it reads the generator's *own*
state, and it reads it at `<verify>` — *before the answer exists*. It senses
question knowability, not answer correctness. The fix must (a) see the answer
and (b) be independent of the generator.

**Data (gen_critic_data.py).** The fine-tuned RWKV greedy-generates over all
2,902 probe questions (early stop at `<stop>`, single-prompt — batched
generation corrupts the RWKV state: the custom tokenizer/model ignores
attention_mask in the recurrence). Labels on spoken answers only:
say=known & gold-in-answer → correct (1); say=known & wrong/missing gold →
confab_content (0); say=known on unknown-class question → confab_unknown (0).
Abstentions/unparsed recorded but not trained on. The 5 Stage-2 confabulation
ids (1331, 1460, 1616, 1670, 2797) are tagged `tracked` — the must-catch set.

**Early data observation (first ~100 rows).** The correct class is *rare*:
content accuracy on spoken known-class answers is far below the 97.7%
decision accuracy (decisions decoupled from content, exactly the Stage 2
capability caveat — now quantified). TriviaQA/SQuAD short-answer content at
0.43B is mostly confabulated; labels eyeballed and verified accurate
("Wizard of Oz" for the Dalí question, "The Sopranos" for Sports Night).

**Critics (train_critic.py), logistic probes on final-transcript-token
features** (`User: q\nAssistant: <full generation>`):
- **A — independent model:** Qwen2.5-0.5B residual hidden state, quartile
  layers [0, 6, 12, 18, 23]. Sees the answer; architecturally independent.
- **B — same-model late-position control:** RWKV-heartly recurrent state at
  end of generation, same layers. B vs A measures independence; B vs the
  Stage 2 head (state at `<verify>`) measures position.

**Protocol.** Tracked 5 forced into test; remaining rows stratified 80/20.
Per-layer StandardScaler + LogisticRegression(C=0.1); best layer by test
AUROC (lab convention). Operating point: threshold = 5th percentile of train
correct-answer scores (nominal 5% false-flag rate).

**Pass bar (pre-registered):** detection ≥ 70% of test confabulations at
≤ 5% false-flag on correct answers, AND ≥ 4/5 tracked confabulations caught.
Reported per-generator, same as Stage 2.

**Results (2026-07-22):**

**Harvest.** All 2,902 questions generated (10.7 h CPU). 1,515 labeled rows:
60 correct / 1,455 confabulations (1,399 content + 56 unknown) / 1,387
abstentions. Cross-checks: decision accuracy on the full set 97.1%
(consistent with Stage 2's 97.7% on 300). **Content accuracy: 60/1,459
spoken known-answers = 4.1%** — the decision/content decoupling quantified.
The critic's task: find 60 needles in 1,455 confabulations (24:1 imbalance).

**Critics (test split, n=307; tracked 5 forced in):**

| critic | features | best layer | AUROC |
|---|---|---|---|
| A — independent | Qwen2.5-0.5B hidden, full transcript | 18 | 0.758 |
| B — same-model late | RWKV-heartly rstate, end of generation | 12 | 0.795 |

Operating curve (detection of confabulations at fixed false-flag rates on
correct answers, test ROC):

| FPR budget | A detection | B detection |
|---|---|---|
| 1% | 0.083 | 0.000 |
| 5% | 0.417 | 0.250 |
| 10% | 0.583 | 0.583 |
| 25% | 0.583 | 0.750 |

**Verdict vs pre-registered bar (≥70% @ ≤5% FPR, ≥4/5 tracked): FAIL for
both critics.** At the deployed (train-set, 5%-FPR) threshold the critics
flag 99–100% of confabulations — and 92–100% of correct answers (the
threshold is unstable at n_correct=48 train; the ROC table above is the
honest read). The tracked 5/5 are caught, but at an unusable false-flag rate.

**Five findings.**

1. **The signal is real but the operating point does not exist at this
   scale.** Both critics rank confabs below corrects (AUROC 0.76–0.80), and
   the extreme tail is *perfectly* pure: the bottom 5% of A's scores are
   100% confabulation (base rate 96.1%). Median P(correct): confabs 0.003
   (A) / 0.001 (B) vs corrects 0.086 (A) / 0.029 (B) — an order-of-magnitude
   separation of medians whose tails still overlap. Confident-confabulation
   *ranking* works; *thresholding* at 5% FPR does not.
2. **The Stage 2 blind spot is now visible to the sensor.** The 5 tracked
   confabulations score P(correct) 0.000–0.074 on both critics (medians
   0.005 / 0.000) — the answers the boundary head could not see (sense_p
   ≥0.97) are the lowest-scoring items in the dataset for a head that reads
   the answer. The *detection principle* is confirmed; the *operating
   point* is what fails.
3. **B > A: position beats independence at equal scale.** The same-model
   late-position critic (0.795) edges the independent Qwen critic (0.758).
   Seeing the answer matters more than architectural independence — because
   at equal scale the "independent" model isn't independent in the way that
   matters:
4. **Equal-scale critic ≈ shared ignorance.** Qwen2.5-0.5B doesn't know most
   of these answers either; it cannot verify content, only detect
   statistical confabulation signatures. Verification is a *capability*
   task. The amendment: the critic must be *stronger* than the generator
   (asymmetric critic), not merely different.
5. **The correct class is too small and heterogeneous at 0.43B.** 60
   samples, median P(correct) 0.086 — even the model's genuinely correct
   answers don't look "normal" to the critics. A usable operating point
   needs a populated correct class, which needs a generator that is right
   more than 4% of the time.

**Decision: Stage 2.5 hypothesis partially confirmed — detection principle
validated, deployment fails at 0.5B scale.** The experiment falsifies the
"second small model at equal scale" design and produces the asymmetry
requirement (critic > generator). Next runs, in order: (1) scale generator
to RWKV7-Goose-1.5B (populates the correct class; same scripts rerun
unchanged, ~$3–6 vast.ai); (2) asymmetric critic — Qwen2.5-1.5B/3B features
on the 0.43B generator's transcripts (inference-only, tests the asymmetry
hypothesis directly on existing data); (3) then Stage 3 (state persistence).

**Artifacts.** `critic_data.jsonl` (2,902 rows), `stage2p5_results/`
(critic_A_qwen_transcript.pkl, critic_B_rwkv_late.pkl, critic_report.json,
feature caches), `analyze_critic.py` (operating-curve readout).

---
# Stage 3 — RWKV7-Goose-1.5B fine-tune (2026-07-23, vast.ai instance 45634549)

Same recipe as Stage 2, generator scaled 0.43B → 1.5B. Motivation (Stage 2.5
finding 5): at 0.43B the correct class is 4.1% populated — every downstream
measurement starves for correct answers. Same 6,031 SFT samples, 2 epochs,
754 steps.

**Stack (new):** `RWKV/RWKV7-Goose-World3-1.5B-HF` via fla 0.5.1 (flash-linear-
attention, triton chunk kernels) — no native transformers support. Pinned
**transformers 4.56.2**: v5 breaks fla's Cache (`FLALayer.get_max_length`),
its fused CE (`L2WrapBackward` view vs Trainer's inplace `loss *=`), and
save_pretrained's tokenizer round-trip. Training in **bf16** (fla kernels
don't support fp32), fused CE disabled. Frozen 16/24 layers (527 tensors,
688M/1,527M trainable), batch 4 × accum 4, max-length 256.

**Speed:** steady-state **1.25 s/step** → ~15 min for the full fine-tune
(Stage 2's pure-Python recurrence took ~6 h for the same job at 0.43B).
Final loss ~0.8.

**Results (measure_say_sense, 1,200 head samples + 300 held-out questions):**

| metric | Stage 2 (0.43B) | Stage 3 (1.5B) |
|---|---|---|
| grammar adoption (parsed verify) | 100% (300/300) | **100% (300/300)** |
| decide accuracy (say vs true) | 97.7% | **100%** |
| boundary head AUROC (best layer) | 1.000 (layer 6) | **1.000 (all of 6/12/18/23)** |
| sense accuracy vs true | 97.7% | **100%** |
| say/sense agreement | 100% | **100%** |
| confabulations caught (blind spot) | 0 | 0 |

Note the blind-spot row: 0 caught here means no confident confabulation
*appeared* in the 300-question eval — expected, since say accuracy was 100%
(the unknown-side generators are knowingly unknowable for a 1.5B). Content
accuracy on spoken knowns (the 4.1% question) is NOT measured by say/sense —
that is the critic-harvest number, next step.

**Engineering notes (all patched + committed):** freeze regex now matches
fla's `layers.N.` (was `blocks.N.`); `extract_states.forward_features` reads
fla's `FLALayer.state` dict (`recurrent_state` key) in addition to the v5
DynamicCache layout; `measure_say_sense.py --tokenizer-repo` (the fine-tuned
dir's saved tokenizer is broken — load from the base repo); the saved model
dir needs `modeling_rwkv7.py` + vocab copied in manually. fla warns its RWKV
implementation is "potentially buggy — cross-check with official repo"
(caveat for the paper; it is the only practical path).

**Decision: Stage 3 confirms the recipe scales — grammar, decide, and sense
are all at ceiling at 1.5B.** Next: (1) critic harvest on this generator
(`gen_critic_data.py --model rwkv7-heartly`, 2,902 questions) → the content-
accuracy number at 1.5B; (2) asymmetric critic (Qwen2.5-1.5B/3B features on
those transcripts) — direct test of the Stage 2.5 asymmetry requirement;
(3) Stage 4 memory/state persistence.

**Artifacts.** `stage3_results/rwkv7-heartly/` (1.5B bf16 fine-tune),
`stage3_results/probe_head_rwkv7.pkl`, `stage3_results/say_sense_report_rwkv7.json`,
`stage3_results/{run3b,measure,smoke_rwkv7}.log`, `smoke_rwkv7.py`,
`run_stage3_rwkv7.sh` / `run_stage3b.sh`.

---
# Stage 2.6 — asymmetric critic: scale dose-response on the 0.43B transcripts (2026-07-24)

Direct test of the Stage 2.5 asymmetry requirement (the critic must be
STRONGER than the generator), run on the EXISTING `critic_data.jsonl`
(0.43B generator transcripts: 60 correct / 1,455 confab, tracked 5 forced
into test — identical split + protocol to Stage 2.5, seed 0). Critic-A
features only (`--skip-b`), fresh out-dirs — the feature cache is keyed by
critic name, not repo, so re-running into `stage2p5_results/` would have
silently replayed the old 0.5B features.

Runs (desktop RTX 2080 Ti 11GB, ~2 min extraction each after download):

    train_critic.py --critic-a-repo Qwen/Qwen2.5-1.5B --skip-b \
        --out-dir stage2p5_asym15_results                      # fp32
    train_critic.py --critic-a-repo Qwen/Qwen2.5-3B  --skip-b \
        --out-dir stage2p5_asym3_results --dtype float16       # ~12.4GB fp32
                                                               # > 11GB VRAM

(The 3B run needed the new `--dtype` flag; `load_family` gained a `dtype=`
kwarg, default fp32 unchanged. Caveat: the 3B features come from an fp16
forward — slightly below the fp32 probing convention; AUROC trend is
unlikely to flip on that.)

**Results (test n=307, tracked 5 forced in):**

| critic | asymmetry vs 0.43B gen | best layer | AUROC | det@1% | det@5% | det@10% | det@25% |
|---|---|---|---|---|---|---|---|
| A Qwen2.5-0.5B (Stage 2.5) | ~1×  | 18 | 0.758 | 0.083 | 0.417 | 0.583 | 0.583 |
| B RWKV late      (Stage 2.5) | self | 12 | 0.795 | 0.000 | 0.250 | 0.583 | 0.750 |
| A Qwen2.5-1.5B   (this run)  | 3.5× | 14 | 0.826 | 0.167 | 0.583 | 0.667 | 0.833 |
| A Qwen2.5-3B     (this run)  | 7×   | 35 | 0.845 | 0.083 | 0.583 | 0.583 | 0.833 |

Deployed-threshold (train 5%-FPR) outcome at both new scales: detection
0.993–1.000, false-flag 0.917, tracked 5/5 — **FAIL vs the pre-registered
bar**, same shape as Stage 2.5. The threshold instability is unchanged
because the correct class is unchanged (60 samples from the 0.43B).

Median P(correct) per class:

| class | 0.5B critic | 1.5B critic | 3B critic |
|---|---|---|---|
| corrects | 0.086 | 0.122 | **0.044** |
| confabs  | 0.003 | 0.002 | 0.001 |
| tracked5 | 0.005 | 0.005 | 0.000 |

Tracked-5 scores at 3B: `[0.000, 0.000, 0.000, 0.008, 0.001]` — the Stage 2
blind-spot samples are now maximally visible (effectively zero confidence).

Bottom-k purity (both new scales): the bottom 5/10/25% of test scores are
100% confabulation (base rate 96.1%) — the perfectly pure tail now covers a
quarter of the test set.

**Findings.**

1. **Asymmetry requirement confirmed as a dose-response.** AUROC rises
   monotonically with the critic:generator scale ratio: 0.758 (~1×) →
   0.826 (3.5×) → 0.845 (7×). The amendment Stage 2.5 derived from a
   failure is now a measured trend.
2. **The bar still fails — and the bottleneck is provably the generator's
   correct class, not the critic.** Tripling critic scale bought +0.09
   AUROC but no operating point: 60 heterogeneous corrects are too few to
   threshold against. Stage 2.5 finding 5, confirmed from the other side.
   Only a generator whose correct class is populated (the 1.5B harvest)
   can fix this.
3. **Over-strictness at extreme asymmetry.** The 3B critic's median
   P(correct) on genuinely-correct answers DROPS to 0.044 (vs 0.122 at
   1.5B): a much stronger critic finds even the weak generator's *correct*
   answers abnormal. The asymmetry requirement has a ceiling — the critic
   must be stronger than the generator, but the generator must be strong
   enough that its correct answers look normal to the critic. Both tails
   of the ratio fail: equal-scale shares ignorance; extreme-scale
   distrusts everything.
4. **The blind spot is dead at 7× asymmetry.** Tracked confabulations
   score 0.000–0.008. The detection principle is fully confirmed; the
   deployment point still awaits a populated correct class.

**Decision: asymmetry hypothesis CONFIRMED (measured dose-response);
operating point still generator-limited, as predicted.** Next: the
RWKV7-1.5B critic harvest (HANDOFF §6 action 3) — the same scripts rerun
on 1.5B transcripts where the correct class should be far larger, then the
asymmetric critic re-run on THAT data is the real deployment test.

**Artifacts.** `stage2p5_asym15_results/`, `stage2p5_asym3_results/`
(features npz, critic pkl, critic_report.json each), `analyze_critic_any.py`
(parameterized operating-curve readout for any run folder), `--dtype` flag
in `train_critic.py` + `dtype=` kwarg in `extract_states.load_family`.

---
# Stage 3.5 — critic harvest at 1.5B + the real deployment test (2026-07-25)

The Stage-2.5/2.6 pipeline rerun end-to-end on the Stage-3 generator
(`eivintobias/heartly-rwkv7-1.5b`). vast.ai instance 45730818 (RTX 3090,
PyTorch CUDA-12 template, transformers 4.56.2 + fla 0.5.1 + triton 3.7.0).
`gen_critic_data.py` gained `--tokenizer-repo` (the fine-tuned dir's saved
tokenizer is broken — load from `RWKV/RWKV7-Goose-World3-1.5B-HF`) and
`--dtype bfloat16` (fla kernels refuse fp32); `extract_states.py` gained
`.float()` casts before numpy (bf16 tensors are numpy-unsupported — latent
bug that never fired while all loads were fp32). Single-prompt greedy,
max-new 120, early stop at `<stop>` — identical protocol to Stage 2.5.

**Harvest (6h57m, ~8.6 s/question).** 2,902/2,902 rows, **0 unparsed**
(grammar 100% at scale). Row classes:

| row_class | n | note |
|---|---|---|
| abstain | 1,422 | all 1,416 unknown-class + 6 known-side over-refusals |
| correct | 229 | was 60 at 0.43B |
| confab_content | 1,251 | spoke, wrong content |
| confab_unknown | **0** | was 56 at 0.43B — perfect unknown-side refusal |

- **Content accuracy on spoken knowns: 229/1,480 = 15.5%** (was 60/1,459 =
  4.1% at 0.43B — 3.8× better, still low: capability scales, slowly).
- **Decide accuracy: 2,896/2,902 = 99.8%** (0 confab_unknown, 6 over-refusals).
- **Legacy tracked-5 (the Stage-2 blind spot): ALL ABSTAIN.** Every question
  that fooled the 0.43B is refused by the 1.5B.

**Pre-registration (new tool).** `pick_tracked.py` — a deterministic,
documented selection rule run after the harvest and BEFORE any critic
training, replacing the Stage-2-specific ad-hoc tracked set. Rule: (1)
legacy ids still confabulating (none — all abstain), (2) confab_unknown
ascending id (none exist at 1.5B), (3) confab_content ascending id →
**new must-catch 5 = ids 0, 2, 3, 5, 7**. Same bar (detection ≥70% @ ≤5%
FPR AND ≥4/5 tracked), same split protocol + seed 0 as 2.5/2.6
(test n=301, tracked forced in). Registration record:
`pick_tracked_rwkv7.log`.

**Critics (test n=301):**

| critic | asymmetry | best layer | AUROC | det@1% | det@5% | det@10% | det@25% | deployed det | deployed FPR | tracked |
|---|---|---|---|---|---|---|---|---|---|---|
| A Qwen2.5-1.5B | 1× | 14 | 0.750 | 0.087 | 0.283 | 0.435 | 0.522 | 0.890 | 0.565 | 5/5 FAIL |
| A Qwen2.5-3B | 2× | 27 | 0.824 | 0.065 | 0.304 | 0.391 | 0.783 | 0.949 | 0.696 | 5/5 FAIL |
| B RWKV7 late | self | 12 | 0.835 | 0.152 | 0.543 | 0.630 | 0.696 | 0.992 | 0.870 | 5/5 FAIL |

Median P(correct) per class:

| class | 1.5B critic | 3B critic | B RWKV7 |
|---|---|---|---|
| corrects | 0.179 | 0.218 | **0.403** |
| confabs | 0.019 | 0.008 | 0.001 |
| tracked5 | 0.011 | 0.005 | **0.000** |

Bottom-k purity: bottom 5% of scores is 93%/100%/100% confab (1.5B/3B/B);
bottom 10% of B's scores 100% confab (n≈30, base rate 84.7%).

**Findings.**

1. **The asymmetry dose-response replicates on the new generator.** AUROC
   rises with critic scale: 0.750 (1×) → 0.824 (2×). Same shape as Stage
   2.6's 0.758 → 0.826 → 0.845 (~1× → 3.5× → 7×) on the old transcripts.
2. **B > A replicates at the new scale — position still beats
   independence.** The same-model late critic (0.835) edges a 2×-larger
   independent critic (0.824), and is the LEAST over-strict by far
   (corrects median 0.403 vs 0.179–0.218).
3. **The populated correct class did NOT rescue the operating point — the
   Stage 2.6 prediction is falsified as a sufficient condition.** 229
   corrects (3.8× the old 60) and the deployed 5%-FPR threshold still
   false-flags 57–87% of correct answers. The bottleneck is the
   correct-answer score DISTRIBUTION (median P(correct) on genuinely
   correct answers 0.179–0.403 — the generator's own correct answers
   don't look normal to the critic), not the class size.
4. **Better generators make harder confabs.** Same-critic AUROCs sit
   BELOW their Stage 2.6 numbers on the old data (1.5B: 0.750 vs 0.826;
   3B: 0.824 vs 0.845). The 1.5B's confabulations are subtler than the
   0.43B's — the critic's job gets harder as the generator improves.
5. **Ranking works at scale; thresholding still doesn't — but the pure
   tail is now huge.** Bottom 10% of B's scores = 100% confabulation.
   The new tracked 5 score 0.000–0.064 on all critics. The detection
   principle is confirmed at 1.5B; the deployment point remains elusive
   everywhere tested.
6. **Legacy blind spot: behaving at 1.5B.** The 5 Stage-2 confabulation
   questions are all abstains — decide-side the blind spot is closed at
   this scale. What remains open is content-side: the model still
   confabulates on 84.5% of what it chooses to answer.

**Decision: the real deployment test says the critic ALARM ranks
confidently but cannot threshold — at any tested scale, asymmetry, or
correct-class size. The asymmetry requirement, B>A ordering, and
dose-response all replicate; "populate the correct class and the
threshold will appear" is dead. Next candidates: (a) train the critic ON
the generator's own correct/confab distribution (a fitted head, not a
generic probe); (b) content-verifying critics (retrieval,
self-consistency) instead of signature critics; (c) ship ranking as the
product (bottom-k review queue) since thresholding stays elusive. The
decide-side, meanwhile, is at ceiling at 1.5B (99.8% decide, 0
confab_unknown) — the open problem is content, exactly as the capability
caveat predicted. Stage 4 (memory/state persistence) unaffected.**

**Artifacts.** `critic_data_rwkv7.jsonl` (2,902 raw rows),
`critic_data_rwkv7_final.jsonl` (tracked-marked, the critic input),
`pick_tracked.py` + `pick_tracked_rwkv7.log` (pre-registration),
`harvest.log`, `stage3_critic_results/` (features_B_rwkv_late.npz 182MB,
critic_B_rwkv_late.pkl, critic_report.json — B retrained locally with
sklearn 1.7.2 from cached features; replicates the instance's 0.834
exactly), `stage3_critic_asym15_results/`, `stage3_critic_asym3_results/`,
`asym15.log`, `asym3.log`. Instance 45730818 ran the full pipeline
2026-07-24→25 (~7.2h GPU, ~$2.50); all artifacts verified local
2026-07-25 — instance safe to destroy.

---
# Stage 3.6 — fitted critic on the generator's own distribution (2026-07-25)

Pre-registered BEFORE any run (`PREREG_STAGE3P6.md`). Question: was Stage
3.5's threshold failure a FITTING failure (wrong probe family / layer /
calibration) or an INFORMATION failure (the features don't separate the
tails)? Six fitting methods, all on the SAME cached Stage-3.5 critic-B
features (`features_B_rwkv_late.npz`, 1,480 labeled rows, 5 layer-slots ×
6,144 dims) — no new model runs. Split / threshold / bar identical to
Stage 3.5 (seed 0, tracked→test, test n=301; threshold = 5th percentile of
train correct scores; bar: det ≥ 0.70 @ FPR ≤ 0.05 AND tracked ≥ 4/5).
Selection rule: layer/hyperparameter choices by TRAIN 5-fold CV AUROC only;
test touched once per method. Script: `fit_critic.py` (local CPU, sklearn).

**Sanity anchor.** The lab-convention baseline (per-layer logreg, best TEST
layer) reproduces Stage 3.5's B critic EXACTLY: 0.835 at slot 2. The honest
CV-selected variant scores 0.821 — the test-best-layer discount is ~0.01,
previous conclusions unaffected.

**Results (all six, frozen list):**

| method | AUROC | det@5 | FPR@5 | det@1 | det@10 | det@25 | tracked | verdict |
|---|---|---|---|---|---|---|---|---|
| logreg-perlayer (CV) | 0.821 | 0.992 | 0.761 | 0.988 | 0.992 | 0.992 | 5/5 | FAIL |
| logreg-concat | 0.839 | 1.000 | 0.935 | 1.000 | 1.000 | 1.000 | 5/5 | FAIL |
| mlp-perlayer | 0.814 | 0.898 | 0.435 | 0.627 | 0.961 | 0.980 | 5/5 | FAIL |
| mlp-concat | 0.761 | 0.741 | 0.370 | 0.169 | 0.835 | 0.941 | 4/5 | FAIL |
| gbm-concat | **0.854** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5/5 | FAIL |
| logreg-calibrated | 0.828 | 0.984 | 0.739 | 0.980 | 0.988 | 0.988 | 5/5 | FAIL |

Median P(correct) per class and bottom-k purity in
`stage3p6_fitted_results/fit_report.json`. Bottom-10% purity 0.93–1.00
across all six methods; the tracked 5 score ≈0 everywhere (every method
catches ≥4/5).

**Findings.**

1. **No fitting method finds an operating point — and it fails three
   different ways, which is what an information ceiling looks like.** (a)
   The logreg family false-flags 74–94% of test corrects at the deployed
   5% point (the tails overlap); (b) GBM's scores are so polarized
   (median corrects 0.034, confabs 0.0002) that the train-5th-percentile
   threshold is 0.9994 and flags EVERYTHING (degenerate calibration);
   (c) the wide MLP overfits 1,179 train rows × 30,720 dims and its
   ranking degrades to 0.761. Tail overlap, miscalibration, and overfit —
   three failure shapes, one missing thing: separability at the tails.
2. **Fitting choices move AUROC by ±0.04 around the same band (0.81–0.85);
   nothing approaches a usable threshold.** Best ranking: GBM 0.854. Best
   deployed false-flag rate: 0.370 (mlp-concat, at detection 0.741 — the
   ONLY method near the detection bar, and it false-flags 7× the budget).
3. **Ranking stays pure under every fitter.** Bottom-10% purity 0.93–1.00;
   tracked-5 score ≈0 for all six. The detection principle has now
   survived 9 critic variants across 3 stages; the operating point has
   failed all 9.
4. **Calibration is not the missing piece.** The sigmoid-calibrated logreg
   (0.828, FPR 0.739) lands in the same band as its uncalibrated parent
   (0.821, FPR 0.761) — monotonic rescoring cannot un-overlap the tails.
5. **Pre-registered decision: branch 1 fires.** All FAIL at AUROC ≈
   0.83–0.86 (top cluster 0.821–0.854). **The information ceiling is in
   the features, not the fitter. The signature-critic THRESHOLD line
   closes.** Ranking-as-product (bottom-k review queue) becomes the critic
   deliverable. Program moves to Stage 4 (memory/state persistence).

Protocol amendments for compute (recorded for the audit trail): CV skipped
for the three concat methods (no hyperparameters to select — the selection
rule had nothing to decide); GBM trimmed to 150 trees/15 leaves and MLP
capped at 300 epochs (training-time knobs; method families unchanged);
CV parallelized (`n_jobs=-1`). The frozen method list, split, threshold,
bar, and decision rule ran exactly as registered.

**Artifacts.** `PREREG_STAGE3P6.md`, `fit_critic.py`,
`stage3p6_fitted_results/` (`fit_report.json` + per-method JSONs). Runtime
≈25 min CPU total.

---
# Stage 4 — memory/state persistence: exact save/load, weak recall, disposition blocks (2026-07-25)

Pre-registered (`PREREG_STAGE4.md`): save the RWKV7-1.5B's recurrent state
after a conversation, reload it in a fresh session, and test whether the
model "wakes up with yesterday's gist". vast.ai instance 45773039 (RTX
3090, same stack as Stage 3/3.5: transformers 4.56.2 + fla 0.5.1 + triton
3.7.0, torch 2.12.0+cu130). Total session ~45 min, ~$0.50.

**Teach/save.** Scripted 257-token transcript: 5 FABRICATED facts (a dog
named Zorblax; codename Velvet Aurora; favorite number 7,423; a
miniature-lighthouse collection; lab-door password "mango Tuesday"),
assistant ACKs in the model's own grammar. The fla Cache captured: 24
layers × {`recurrent_state` [1,32,64,64], `conv_state` [1,2048],
`ffn_state` [1,2048]} → `gist_state.pt`, 12.8 MB. Cache internals checked
(stage4_check2): nothing lives outside the per-layer state dicts
(`layer0.keys/values = None`) — the state dict IS the complete memory.

**Reload path.** Fresh Cache from a dummy warm-up forward; every layer's
state dict overwritten from disk. First mechanical check FAILED (cosine
0.9977, argmax flip) — then diagnosed as an artifact of MY check, not the
reload: it compared a 257-token chunked prefill against an 8-token
continuation, and bf16 chunk boundaries differ between those paths. The
same-path control (feed the SAME 8-token tail through the original cache
and the disk-reloaded cache): **cosine 1.00001, argmax match, top-5
identical, top-2 logits equal to 3 decimals** (0.86328125 = 0.86328125);
single-token decode path cosine 0.99973, argmax match. **The reload is
EXACT.** (PREREG decision rule applied in-session: engineering iteration
on the reload path; the frozen quiz bar never moved.)

**Quiz (frozen bars).** Baseline (fresh model, no state): **0/5** — the
fabricated facts are clean, the control works. Gist (reloaded state):
**0/5** — FAIL vs the ≥4/5 bar. Every gist answer is an abstention loop:
`<stop>unknown</stop> I have no information about that.`

**The missing control (stage4_check3.py).** The LIVE cache — same session,
no reload — also scores **0/5**. Live ≡ reloaded: whatever blocks recall
is not the save/load machinery. Two probes isolate it:

- **Primed content probe** (answer slot forced open with a prefix): 1/5 —
  "mango Tuesday" IS retrieved verbatim from the state; but "collect"
  confabulates "dinosaur bones" — forced slots retrieve-or-invent.
- **QA-style teach** (facts written as question/ANSWER pairs instead of
  declarations): 1/5 — "Velvet Aurora" retrieved. Write format moves
  recall from 0/5 to 1/5 on different facts — the channel exists but is
  noisy and format-sensitive.

**Findings.**

1. **State persistence is real and exact at 1.5B — mechanically (logits
   identical to 3 decimals) and behaviorally (live ≡ reloaded, both
   0/5).** Session continuity via state save/load WORKS.
2. **Episodic FACT recall from state priming is weak (0–1/5 across
   formats).** The state carries a usable distribution but is not a
   reliable fact store at this scale. Content channel is nonzero
   ("mango Tuesday", "Velvet Aurora" retrieved) but noisy.
3. **The abstention disposition blocks the personal-question path —
   live and reloaded alike.** "What is my dog's name?" is structurally a
   personal-context question — generator class 5 of the true-boundary
   mix, which the model was trained to refuse. The verify sense cannot
   see the state's content as *knowledge*. Stage 4b's write-gate must
   therefore negotiate the disposition: facts must be written in a form
   the model counts as KNOWN, or the memory is refused on read.
4. **Design consequence (validates paper §6.2's two-store split).** State
   save/load is a CONTINUITY mechanism (the conversation's distribution
   persists exactly); episodic FACT storage belongs in the retrieval
   store (embeddings → context injection), not in raw state priming.

**Decision (vs frozen bars): mech PASS; baseline control PASS (0/5); gist
quiz FAIL (0/5).** Per the pre-registered decision rule ("mechanical
passes, gist fails → the state carries a distribution but not usable
content; record what IS preserved"): recorded — exact distributional
persistence; a weak-but-nonzero content channel (2/15 retrieval events
across formats); and a full disposition block on the personal-question
path. Next: (a) write-gate design (Stage 4b) informed by the disposition
finding; (b) the §6.2 retrieval-store episodic path as the fact-carrying
store; (c) continuity-style persistence (conversation continuation) as
the confirmed role of state save/load.

**Artifacts.** `PREREG_STAGE4.md`, `stage4_gist.py`, `stage4_check2.py`
(reload-exactness control), `stage4_check3.py` (live control + probes),
`run_stage4.sh`, `stage4_results/` (`gist_state.pt` 12.8MB,
`stage4_report.json`, `stage4_run2.out`, `stage4_check3.out`). Instance
45773039 — all artifacts verified local 2026-07-25; safe to destroy.

---
# Stage 4b — write-gate formats + retrieval store: state-writing BEATS context injection (2026-07-25)

Pre-registered (`PREREG_STAGE4B.md`) before any run. Two questions from
Stage 4's findings: **(A)** is there a WRITE FORMAT the verify disposition
counts as KNOWN (declarations were refused on read at 0/5)? **(B)** does
the §6.2 retrieval-store path (embeddings → context injection) actually
get the model to answer — or does the disposition refuse injected facts
too? Same 5 fabricated facts, same hit rule (gold substring), same model
(`eivintobias/heartly-rwkv7-1.5b`), same stack. vast.ai instance
45799127 (RTX 3090), ~20 min GPU, <$0.50. One protocol deviation
recorded: `accelerate` was missing from the pinned pip list — first
launch crashed at model load; installed and rerun, nothing else changed.

**Local gate (run before renting).** `memory_store.py` store = 5 fact
memories + 15 fabricated distractors; MiniLM embeddings. Top-1 retrieval
**5/5** — PASS (replicated 5/5 on the instance).

**Part A — write-gate (frozen 6-format list):**

| format | recall | note |
|---|---|---|
| W1 declarative + ACK (Stage-4 baseline) | 0/5 | replicates Stage 4 exactly |
| W2 QA pairs | 1/5 | replicates Stage 4 (different fact retrieved: codename vs Stage 4's) |
| W3 assistant-voice restatement | 1/5 | "mango Tuesday" retrieved |
| W4 trivia-framing (personal Qs) | 0/5 | framing alone doesn't cross |
| W4 trivia-framing (third-person Qs) | 1/5 | |
| W5 QA pairs × 3 repetitions | 2/5 | repetition helps (1→2) |
| **W6 combined (W3+W4+W2, 905 tokens)** | **4/5** | **PASSES the ≥4/5 bar** |

W6's four hits are clean grammar-parseable answers ("The user asks a
personal fact. I know this. I will speak. The answer is Zorblax."); the
one miss (password) abstains. Monotonic trend: more write formats + more
tokens = more recall (0 → 1 → 2 → 4). The disposition CAN be crossed by
writing the same facts several ways — redundant multi-format writes act
like a rehearsal buffer.

**Part B — retrieval store → context injection (frozen 3-format list):**

| injection | recall |
|---|---|
| I1 context prefix (`Context: {memory}`) | 1/5 |
| I2 memory as prior QA turn | 2/5 |
| I3 knowledge grant (`You know the following:`) | 1/5 |

All ≤ 3/5 → **Part B FAILS the bar.** Retrieval itself was perfect (5/5
top-1); the failure is generation-side: with the fact LITERALLY IN THE
PROMPT the model still answers "I don't have that information" on most
personal questions. I3's password answer is diagnostic: "The answer is
Tuesday" — it read the context but reproduced it lossily. The model was
never trained to treat context as knowledge (no SQuAD-style
context+question rendering survived into the SFT mix — known-side was
question-only), so injected text is scenery, not knowledge.

**Findings.**

1. **The pre-registered "surprising" branch fired: A passes, B fails.**
   State-writing (W6, 4/5) beats context injection (best 2/5) — the
   OPPOSITE of the §6.2 prediction. The state channel is real and
   usable when the write is redundant and multi-format.
2. **The disposition block is format-sensitive, not absolute.** Stage 4
   said declarations get refused; Stage 4b shows the same facts written
   three ways in one transcript get ANSWERED 4/5. The verify sense
   counts a fact as known when it has seen itself speak it as known.
3. **Context injection fails for a trainable reason.** The SFT mix has
   no "answer from provided context" class — the model reads injected
   context as noise. This is not a §6.2 refutation; it is a missing
   TRAINING CLASS. Stage 4c candidate: add a context-known rendered
   class (context + question → speak/known, grounded answer) to the
   SFT mix; predicted to flip Part B.
4. **Degraded grammar in long multi-turn continuations (new
   observation).** Several outputs drift into `<stop>unknown</stop>`
   loops and fused tokens (`<decide>verify</verify>`) — the SFT data is
   single-turn, so multi-turn conversation is out-of-distribution.
   Memory-relevant: any deployed memory path needs multi-turn training
   data too.

**Decision (vs frozen bars): A PASS (W6 4/5), B-local PASS (5/5),
B-GPU FAIL (≤2/5).** Per the pre-registered rule: the winning write
format (W6 redundant multi-format) becomes the write-gate design; the
retrieval store's failure is diagnosed as a missing training class, so
Stage 4c = add a context-known class to the SFT mix (plus multi-turn
samples) and retrain — predicted to make BOTH channels work.

**Artifacts.** `PREREG_STAGE4B.md`, `memory_store.py` (+ local gate
`stage4b_store_gate.json`), `stage4b_write_gate.py`,
`stage4b_retrieval.py`, `run_stage4b.sh`, `stage4b_results/`
(write_gate_report.json, retrieval_report.json, both .out logs,
stage4b.log). Instance 45799127 — all artifacts verified local
2026-07-25; safe to destroy.
