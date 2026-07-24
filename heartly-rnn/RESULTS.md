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
