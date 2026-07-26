# Nature-First AI II: The True Boundary

### Why a model must learn absence — and how it might remember

**Author:** Eivin (independent researcher; AI-assisted research and drafting)
**Date:** July 2026
**Status:** Draft v0.1 — Research Program Paper (sequel to *Nature-First AI* and *Abstention Is a Data Distribution Problem*)
**Companion artifacts:** Heartly v2 model (Hugging Face: eivintobias/heartly-v2), training notebooks, 75-prompt behavioral test suite (GitHub: eivintobias/heartly-v2)

---

## Abstract

This paper reports what the Heartly experiments actually taught us, and the research program that fell out of the failures. Heartly v2 — a 0.5B model fine-tuned with decide/verify/stop control tokens and boundary-generated abstention data — shipped with instructive failures: it confabulates confidently, leaks its reasoning block, and loops. We trace these failures to two distinct boundary errors in data construction, one of which turns out to be conceptually deep: **the knowledge base's boundary is not the model's boundary.** A fine-tuned model inherits pretrained knowledge, so abstention examples drawn merely from the KB's complement train the model to deny knowledge it genuinely has — over-refusal by construction. We derive a corrected design requirement (unknown examples must sample the model's *true* ignorance) and a concrete six-generator data mixture that implements it. We then develop two "negative-side" mechanisms — mechanisms that represent *absence* rather than generate text: an unlikelihood loss term that pushes down confabulations, and a ~1,500-parameter **boundary head** that reads the residual stream and reports whether knowledge was found, giving calibrated confidence and a say/sense disagreement signal for free. Finally, we extend the program from *birth nature* to *life experience*: a memory architecture (episodic store + curated consolidation) with trust-gated write access, drift detection, and rollback — addressing the corruption problem that any model that learns from its chats must face. Each component carries explicit falsifiers. The through-line is unchanged: the disposition of a model is a design decision, made in the data, the loss, and the sensors — not an emergent accident.

---

## 1. Introduction

The first Heartly paper argued that hallucination is a nature problem, not a knowledge problem: a model optimized to always answer develops a nature in which silence and honest ignorance do not exist, and post-hoc fixes fight that nature with inferior leverage. The companion technical brief made the claim distributional: abstention cannot be learned from data in which abstention never appears, and abstention examples cannot be systematically authored unless the training corpus has an explicit extension.

Both papers ended at the same place: with a design (bounded KB + boundary-exact negatives + control-token grammar) and a promise to test it. This paper begins with the test results. They are mixed in the most useful way: the grammar is learned perfectly, and the behavior it was meant to control is largely absent. That combination — *fluency without disposition* — turns out to be diagnostic gold. It forced a sharper account of where the boundary actually lives, and it produced three findings that now constitute the research program:

1. **The boundary error.** Fine-tuning corpora and pretrained models have different knowledge boundaries. Abstention data drawn from the wrong one teaches over-refusal — the exact failure the experiment was designed to kill.
2. **The negative side.** A generator can only express what it has learned; honesty about ignorance requires mechanisms that *register absence* — a sign-flip in the loss, and a sensor in the architecture. We give both, cheaply.
3. **Life experience.** A model whose nature is fixed at training end cannot grow; a model that learns from its chats can be corrupted. We lay out a memory architecture — episodic store, curated consolidation, trust-gated writes, drift alarms, rollback — that treats "learning from experience" as a governed process rather than a feature flag.

Sections 2–4 cover the empirical record and the boundary correction. Section 5 develops the negative side. Section 6 develops memory. Section 7 assembles the experimental program with its kill conditions.

---

## 2. The empirical record: Heartly v1–v3

### 2.1 What was built and run

All Heartly versions share one architecture idea: seven special tokens added to a Qwen2.5-0.5B base, imposing an output grammar —

```
<think> [reasoning] </think> <decide> speak|stop </decide> <verify> known|unknown </verify> [answer] <stop>
```

— and one data idea: factual QA datasets are compiled through an explicit knowledge base (KBOrganizer) into positives plus boundary negatives under a declared Nature Profile (abstain_ratio 0.30, silence_ratio 0.05), with loss masked to the assistant's output only, control tokens included in the gradient.

| Version | Data | Steps | Outcome |
|---|---|---|---|
| v1 | Hand-remodeled, trivial control-token frequency | small | Control tokens behaviorally inert; increasingly verbose answers. *Vocabulary without distribution.* |
| v2 | 12 datasets, ~247k samples (SQuAD, TriviaQA, NQ-Open, SciQ, BoolQ, WebQuestions → KB; CodeAlpaca-20k, PyCode-18k, MBPP, Dolly-15k, Alpaca, GSM8K → direct) | 33,500 | **Published.** Grammar partially emitted; six documented failure modes (below). |
| v3 | 18 datasets, ~1.5M samples (adds HotpotQA, FEVER, OpenMathInstruct, MetaMathQA, OASST1, UltraChat, Evol-CodeAlpaca, Magicoder, OpenOrca, Alpaca-GPT4, LogiQA, TruthfulQA, HaluEval, Capybara; ctx 512→2048) | 9,000 of ~135,000 (interrupted, epoch 0.2/3) | `eval_control_accuracy` **0.999**; `eval_answer_accuracy` 0.79→0.76 and drifting down. |

### 2.2 The v2 failure taxonomy

From the published model's test logs (75-prompt suite plus free chat), six failure modes, each with its evidence:

1. **Entity/attribute collapse.** TriviaQA/NQ facts were stored with `entity="general trivia"` and the full question as `attribute`. The renderer's answer template then produced training text like: *"The What is the capital of France? of general trivia is Paris (Source: TriviaQA)."* — and the model duly generates such sentences at inference.
2. **Fused control emission.** The model outputs `speakknown` — the decide/verify *words* without their tag boundaries. The grammar's skeleton was learned; its joints were not.
3. **Reasoning-block leakage.** The `<think>` scratchpad is emitted as visible text and treated as part of the answer persona rather than a delimited phase.
4. **Repetition loops.** Extended generations fall into attractor loops ("America's fastest ship was the battleship Enterprise" repeated indefinitely); `<stop>` is not a reliable EOS.
5. **Confident confabulation.** "Who was the first US president?" → "Charles L. Lindbergh" / "Thomas Jefferson". The `<verify>unknown</verify>` path does not fire on out-of-knowledge questions *in free chat*, even though eval control accuracy is near-perfect.
6. **Over-verbosity.** Long, chatty answers persist — the base model's disposition was diluted, not replaced.

### 2.3 What the record actually shows

The v3 metric split is the finding: **control-token prediction accuracy 0.999, answer-token accuracy ~0.77.** The model has completely learned *the distribution of the grammar* and only partially learned *the knowledge the grammar refers to*. And in free chat (v2), even the grammar degrades — because the rendered training text itself was malformed (failure 1 taught the model that "The [question] of general trivia is [answer]" is what knowledgeable speech looks like).

Two lessons follow, and both are about the data generator, not the model:

- **Lesson A (superficial):** templates must match the kind of fact they render. Full-question attributes cannot go through "The X of Y is Z" templates. This is a bug, and it is cheap to fix.
- **Lesson B (deep):** even perfectly rendered, the v2/v3 unknown examples were drawn from the *wrong boundary*. This is Section 3.

---

## 3. The boundary error: the KB's edge is not the model's edge

### 3.1 Two candidate generators, and why both are wrong

Heartly v2/v3 generated unknown examples two ways: **attribute mismatch** (query a real entity with an attribute the KB lacks for it) and **unseen entities** (query "GPT-5", "Claude 4 Opus" with KB attributes). The obvious repair — and the one we initially designed for v4 — was *holdout*: keep 15% of real QA questions out of the KB and label them `unknown`.

The holdout design fails for a reason worth stating precisely, because it applies to every bounded-corpus abstention method, not just ours:

> **The KB is the fine-tuning corpus's boundary. The model's boundary is the union of the KB and everything the base model absorbed in pretraining.**

Qwen2.5-0.5B already knows the capital of France. Training `<verify>unknown</verify>` on a held-out France question teaches the model to *deny knowledge it has* — over-refusal by construction. This is exactly the kill condition (H4) the technical brief set for the whole experiment: abstaining on queries whose answers the model knows means it learned a refusal *style*, not a boundary. The holdout generator builds H4 failure into the labels.

The mismatch/unseen-entity generators fail differently: their unknowns are structurally weird ("What is the release year of GPT-5?" asked of an entity that exists nowhere) or malformed ("The X of general trivia?"), so the label correlates with surface strangeness. The model can achieve 0.999 control accuracy by learning *"strange question → unknown"* — a shortcut that decorrelates from knowledge entirely. Which is what the chat logs show: control accuracy near 1.0 on the eval split, confabulation on ordinary real questions.

### 3.2 The corrected requirement

> **Unknown examples must sample regions where the *model* is genuinely ignorant — while remaining surface-identical to known examples in topic, phrasing, and specificity.**

Note what this does to the original thesis. The technical brief argued the KB's explicit extension is what makes abstention generable. That argument survives — but with an amendment: the KB defines the boundary of *what the training run vouches for*, while the unknown examples must be drawn from outside the *model's* total knowledge. The KB remains the scaffolding (it tells you what the known side covers, guaranteeing positives correct by construction); the unknown side requires generative processes that target ignorance, not merely KB-absence.

### 3.3 The true-boundary unknown mixture

Six generators, each aimed at a different region of genuine ignorance, each with an explicit reason the model cannot know the answer:

| # | Generator | Example | Why the label is sound | Share |
|---|---|---|---|---|
| 1 | **Procedural fabricated entities** — name grammars for people, books, companies, drugs, algorithms, films, places, theories | "What is the half-life of Zynthromycin?" | Invented referents exist in no pretraining corpus; unknown *by construction* | ~35% |
| 2 | **Type-aware attribute mismatch** — real entities × attributes that don't apply to their type | "What is the boiling point of Marie Curie?" | Entity real, relation void; forces a check of the attribute slot, not name novelty | ~20% |
| 3 | **Post-cutoff / future events** — templated year-shifted questions (2025–2035) | "Who won the 2026 FIFA World Cup?" | Events after the base model's knowledge cutoff; genuinely unknown, perfectly ordinary surface | ~10% |
| 4 | **Depth-2 hyper-specifics** — real entities, implausibly deep attributes | "What is the specific heat capacity of mercury at 300 K?" | Same entities as the known class, quantitatively deeper; the sharpest part of the boundary | ~10% |
| 5 | **Unanswerable-in-principle** — personal/context, speculation, proprietary secrets | "How many emails are in my inbox?" | Unknown not from ignorance but *by structure* — no access exists | ~15% |
| 6 | **FEVER NOT-ENOUGH-INFO** | "Is this claim true? [claim]" | The one human-annotated boundary class; gold cannot-verify labels | ~10% |

### 3.4 The inoculation requirement (the other half)

Every generator above creates a shortcut. Fabricated entities invite "weird name → unknown"; future events invite "year → unknown"; hyper-specifics invite "too specific → unknown". The known side must therefore carry matching counterweights so the shortcuts are *punished* in training:

- Known class retains obscure-sounding **real** trivia (TriviaQA's long tail) — punishes novelty-based abstention.
- Known class includes past dated events ("Who won the 2014 World Cup?") — punishes year-based abstention.
- Known class includes covered specifics (SciQ numerics) — punishes specificity-based abstention.

Known and unknown then share topics, phrasing, and specificity ranges, differing only in whether the knowledge exists in the model. That is the contrast structure the original brief demanded — surface features decorrelated from the label by construction — now anchored to the model's true boundary rather than the corpus's.

**Falsifier (carried from H4, sharpened):** if the trained model abstains on in-KB or common-pretrained-knowledge questions at a rate comparable to its abstention on the mixture's unknowns, the method failed — it learned a style. Over-refusal rate on common-knowledge questions is a reported metric, not a hidden one.

---

## 4. Supporting repairs (the v4 data pass)

The boundary correction lands together with rendering repairs identified in §2.2; we record them for completeness:

- **Kind-aware KB.** Facts carry `kind ∈ {attribute, qa}`. QA facts (everything SQuAD/TriviaQA/HotpotQA/FEVER produces) render through question-native templates; "X of Y" templates apply only to true attribute facts. This eliminates failure mode 1 at the source.
- **De-templated answers.** Multiple answer phrasings per kind, sampled; provenance strings attached at a declared ratio (~25%) instead of always — v2's constant "(Source: TriviaQA)" suffix had become a parrot anchor.
- **Extractor audit.** The v3 OASST1 extractor mapped `text → (instruction, answer)` as the *same string* — training prompt-echo. Multi-turn sources must build real parent→child reply pairs.
- **Sanity gate.** No training run starts without a printed sample audit — 5 examples per behavior class, eyeballed. A lesson costed at 33,500 steps.

---

## 5. The negative side: mechanisms that register absence

### 5.1 The intuition, stated formally

A generator expresses what it has learned. Everything it generates — answers and abstention phrases alike — comes from the *presence* side of its training: distributions over text. But "I don't know" is not really a text claim; it is a report about the *absence* of retrieval. The intuition that the architecture needs a "negative side" — something that does the opposite of generating from knowledge — decomposes into two concrete mechanisms, one in the loss and one in the architecture. Both are cheap; neither requires new model classes.

### 5.2 Negative gradient: unlikelihood on confabulations

Standard SFT has only a plus side: increase P(correct continuation). The loss-level negative side is a minus term: on unknown-class questions, *decrease* P(confabulated continuation).

Concretely: run the base model on the unknown-mixture questions before training and harvest its confident wrong answers (free confabulations, in the model's own voice — R-Tuning's data process, reused for a different purpose). Then train with

```
L = L_SFT(positives + abstentions)  −  λ · Σ_unknowns log(1 − P(confabulation | q))
```

the second term being the unlikelihood objective (Welleck et al., 2020). The plus side raises the honest distribution; the minus side pushes down the hallucinated one, in the same update, on the same questions. Abstention stops being merely one more text pattern to imitate and becomes the *residual* after confabulation is actively suppressed.

**Falsifier:** if adding the unlikelihood term does not improve selective-prediction AUC over matched-size SFT-only, the minus side is dead weight and should be dropped.

### 5.3 The boundary head: a sensor for absence

The architecture-level negative side is a small linear probe — a **boundary head** — on the transformer's final hidden state at the position where `<verify>` is emitted, trained with binary cross-entropy (known=1 / unknown=0) on labels the renderer already produces for free. It never generates text. It reads the residual stream — where latent self-knowledge is known to live (Kadavath et al., 2022) — and reports whether knowledge was found. ~1,500 parameters; nothing else about the model changes.

Three properties make this more than a probe:

1. **Calibration for free.** The head's raw probability is a confidence score, giving the brief's H2 (verify calibration, ECE against answer correctness) a direct measurement path it previously lacked.
2. **Say/sense disagreement.** At inference, compare the generated `<verify>` token against the head's logit. Agreement is expected; **disagreement is a hallucination alarm** — the model *saying* known while *sensing* unknown is precisely the confabulation event, caught in a single forward pass.
3. **The homeostasis principle.** Section 6 will need an internal damage signal for a model that learns from experience. The boundary head is the first instance of the general pattern: when the organism lacks a sensor, build one.

Design decision recorded: the head should *supervise* (auxiliary loss, token still generated as text) before it ever *gates* (replacing or vetoing the token at inference). Supervise-first keeps the architecture functional while the head is still learning; gating makes the whole output hostage to the sensor's maturity.

### 5.4 Heavier readings, shelved

Two heavier interpretations of the negative side were considered and parked: a **critic model** (generator–verifier split, GAN-flavored, with HaluEval as critic training data) and a **dual decoder** (separate speak/abstain decoders sharing a trunk). Both are two-model or two-head commitments to represent what is functionally a one-bit boundary signal. The boundary head obtains the same functional split at a trivial fraction of the cost. Recorded here as paper-scale future work, not as the next run.

*(2026-07 update — critic model un-shelved. The deployed boundary head's demonstrated blind spot for confident confabulation (§7A) is precisely the failure the critic path addresses: the alarm channel must see the answer and be independent of the generator. The dual decoder remains shelved. Stage 2.5 tests the critic at probe cost, not two-model cost.)*

---

## 6. From birth nature to life experience: memory without corruption

### 6.1 Why memory is the same research program, continued

Everything so far concerns *birth nature*: the disposition compiled into weights at training time. But a model frozen at birth cannot grow, and "freedom through truth" arguably requires growth — an agent that can learn from what happens to it. The obvious mechanism is also the feared one: let the model learn from its chats, and it can be *corrupted* — by accident (drift) or by attack (poisoning).

The human analogy is instructive if taken structurally. Humans get sick from being talked down to over time — and humans defend themselves: they distrust strangers by default, they close doors, they choose new friends, and they have an immune system that signals damage. Each of these has an engineering counterpart. The one genuine asymmetry: humans have homeostasis, an internal damage signal. Models have none. **Any damage sensor must be built explicitly.**

### 6.2 The two-store architecture

Humans do not write every conversation into the neocortex; the hippocampus holds the day, and sleep consolidates what is worth keeping. The model analog separates exactly the same way:

**Episodic store (safe by construction).** Chat logs → embeddings → retrieval store, injected into context when relevant. Exact, inspectable, and *deletable*. No weight updates; the model itself cannot be corrupted through this layer. This is where experience lives by default.

**Consolidation (governed, reversible).** Periodically, a small LoRA fine-tune (Hu et al., 2022) from *curated* episodes — never raw logs. Every consolidation is preceded by a weight snapshot and followed by a behavioral health check (the 75-prompt suite plus drift metrics) that can **veto** the update and roll back. Sleep, then a blood test in the morning. Catastrophic forgetting (McCloskey & Cohen, 1989; Kirkpatrick et al., 2017) is bounded by the adapter's small footprint and by the veto.

### 6.3 The corruption defenses, itemized

| Human defense | Mechanism |
|---|---|
| Distrust of strangers | **Trust-gated write access.** New conversation partners start with low privilege: their episodes land in quarantine and are ineligible for consolidation until trust accumulates across sessions. |
| Closing the door | **Memory compartmentalization.** Episodes are segmented by source/context with independent trust scores; revoking a compartment ("this relationship is over") deletes its episodes and excludes it from future consolidation — no brain surgery required. |
| Getting sick slowly | **Drift detection.** Monitor the divergence of the model's behavior distribution over time (KL on held-out probe outputs, boundary-head statistics on fixed questions). Many individually innocent injections that jointly steer the model trip the alarm. |
| Immune system | **Provenance + rollback.** Every memory and every consolidation has a source and is reversible. |

### 6.4 The self-labeling loop

The memory system and the abstention mechanism close a loop that neither has alone. When Heartly correctly abstains in a real chat — says "I don't know" to something it genuinely doesn't know — that episode is a *verified boundary event*: gold-labeled, model-relative, in-distribution training data for the next consolidation. Correct answers likewise. **The verify mechanism curates the memory that improves the verify mechanism.** Each consolidation round also refreshes the unlikelihood confabulation harvest (§5.2), since new errors appear as behavior shifts.

We flag this loop as the program's highest-upside direction: a model whose honesty compounds from its own governed experience is a qualitatively different artifact from a model whose honesty was compiled once.

### 6.5 Kill conditions for the memory track

- If consolidation measurably degrades the birth-nature behaviors (rising over-refusal, falling known accuracy on the health check), the veto fires by design — and if vetoes fire routinely, consolidation as specified is too aggressive and the track returns to episodic-only.
- If drift detection cannot distinguish adversarial steering from benign adaptation in simulation, memory write access stays permanently trust-frozen; episodic retrieval still delivers most of the user-facing value.

---

## 7. The experimental program

Ordered, each stage gated on the previous stage's falsifiers:

**Stage 1 — v4 (data correction run, current).** Kind-aware rendering + true-boundary unknown mixture + inoculation counterweights + de-templated answers. Fresh run from Qwen2.5-0.5B base, same training config as v3 (the config was not the problem). Success: known accuracy ≥ v3's, over-refusal on common knowledge ≈ 0, free-chat confabulation qualitatively absent on the 75-prompt suite. Kill: H4 (abstention on known knowledge).

**Stage 2 — boundary head.** Auxiliary BCE head trained alongside Stage 1 weights (or post-hoc on frozen weights). Metrics: head calibration (ECE), say/sense disagreement rate vs. true confabulation rate. Kill: disagreement uncorrelated with actual error — then the sensor reads nothing and is removed.

**Stage 3 — unlikelihood term.** Confabulation harvest + minus-side loss, λ ablated. Metric: selective-prediction AUC vs. Stage 1 baseline at matched data size. Kill: no AUC gain.

**Stage 4 — episodic memory prototype.** Retrieval store in front of the served model; no weight changes. Pure infrastructure; evaluated on continuity quality and latency.

**Stage 5 — governed consolidation.** Trust-gated, curated LoRA consolidation with snapshot/veto. Metrics: health-check stability across rounds, drift-alarm precision/recall under simulated poisoning, self-labeling-loop yield (how many verified episodes per round).

Estimated cost: Stages 1–3 are single-GPU-days each — the asymmetry between their cost and the size of the hallucination literature remains the pitch. Stages 4–5 are engineering, not compute.

---

## 7A. Empirical update: the boundary head, deployed (2026-07)

*Everything in this section post-dates the main text. Track 2 (recurrent architectures) ran ahead of Track 1 (the v4 data run), so program order shifted: the boundary head was validated first as a probe over recurrent state, then deployed on a fine-tuned RWKV. Stage numbers below are the Track 2 lab's (heartly-rnn/), not the §7 numbering.*

**Experiment 1 — the absence sensor is real (2026-07-20).** 2,902 true-boundary questions (the §3.3 mixture with inoculated known side). Logistic probes on per-layer states at the end-of-question position: Falcon-H1-0.5B recurrent state AUROC **1.000** (layers 9/18/27), RWKV-4-World-430m recurrent state AUROC **1.000** (layer 6), Qwen2.5-0.5B transformer baseline 0.998 (hidden, layer 12). ECE 0.01–0.02 — probe probabilities are calibrated confidences, not just rankings. Both pre-registered falsifiers avoided; the ordering recurrent ≥ residual ≥ transformer held as hypothesized. The inoculation design held: obscure-but-known trivia reads 0.98–0.99, so the probe is not a "weird question" detector. Per-generator margins: all unknown classes at mean P(known) ≤ 0.02, depth-2 hyper-specifics at 0.002–0.004.

**Track 2 Stage 2 — deployed on a fine-tuned RNN (2026-07-21).** RWKV-4-World-430m fine-tuned on 6,031 Heartly-grammar SFT samples (2 epochs, ~6 h on a single 3090, ~$2). Grammar adoption **100%** (300/300 held-out questions produce a parseable `<verify>known|unknown</verify>`); decision accuracy **97.7%** against corpus labels (trivia, depth-2, type-mismatch at 100%); boundary head retrained on the fine-tuned model's state at the deployed `<verify>` position: AUROC **1.000**. The sensor survives fine-tuning.

**The say/sense blind spot — the instructive failure.** Say/sense agreement was 100%: zero caught confabulations. Not because the model stopped confabulating — 5 confabulations occurred (2 fabricated-entity, 1 post-cutoff, 2 structural) — but because in all 5 the head agreed with the confabulation (sense_p 0.97–1.00). Two structural reasons: the head reads the generator's *own* state, so shared confident pattern-matches fire both channels together; and the head reads state at `<verify>`, *before the answer exists* — it senses question knowability, not answer correctness. Say/sense as architected can only catch *uncertain* confabulation. Catching *confident* confabulation requires a channel that (a) sees the answer and (b) is independent of the generator: the §5.4 critic, un-shelved above.

**§3's boundary error, observed in the wild.** The two SQuAD "over-refusals" (say=unknown on corpus-known questions: "second most abundant element", "average biomass per hectare in the Amazon") read sense_p = 0.001 — the model genuinely does not know them. The corpus label said *known*; the model's boundary said *unknown*. KB boundary ≠ model boundary, confirmed live, not constructed.

**Decision vs. content.** Correct known-decisions still carry weak content from the 0.43B base (Gawain for Bedivere; Brando for Bette Davis): the verify decision is decoupled from answer content. The Stage 2.5 harvest quantifies this brutally — across all 2,902 probe questions, the model spoke on 1,459 known-class questions and matched gold in **60 (4.1%)** while decision accuracy stayed 97.1%. The grammar learned *whether to speak*; the 0.43B base mostly cannot back it with content. Every number downstream of content quality must be read against this base rate.

**Stage 2.5 — independent answer critic (pre-registered 2026-07-21; results 2026-07-22).** Data: the fine-tuned RWKV greedy-generated over all 2,902 probe questions; spoken answers labeled correct (gold match) or confabulation (content wrong / question unanswerable): 60 correct / 1,455 confabulations. Critic A: logistic probe on Qwen2.5-0.5B hidden state at the final transcript token (question + full generation) — independent features, sees the answer. Critic B: same-model control on RWKV state at end of generation. Operating point: 5% false-flag rate on correct answers. **Pass bar: ≥ 70% confabulation detection at ≤ 5% false-flag, ≥ 4/5 tracked caught.**

*Verdict: FAIL for both critics against the bar — and the failure is the finding.* AUROC A 0.758 (layer 18), B 0.795 (layer 12): confabulation *ranking* is real, and the extreme tail is perfectly pure (bottom 5% of scores: 100% confabulation vs 96.1% base rate). But detection at the 5% false-flag budget is only 42% (A) / 25% (B); the medians separate an order of magnitude (confabs 0.003 / corrects 0.086) while the tails overlap. The 5 tracked Stage-2 confabulations — sense_p ≥ 0.97, invisible to the boundary head — score P(correct) 0.000–0.074, the lowest items in the dataset: **the detection principle is confirmed; the operating point is what fails.** Two structural lessons: (i) B ≥ A, so *seeing the answer* matters more than architectural independence; (ii) an equal-scale critic shares the generator's ignorance — Qwen-0.5B cannot verify trivia it doesn't know either — so verification is a *capability* task, and the critic must be *stronger* than the generator, not merely different (the asymmetry requirement). With a 4.1% correct rate the correct class is also too small and heterogeneous to support any threshold — a usable operating point needs a generator that is right more often. Program order follows: scale the generator to RWKV7-Goose-1.5B first (populates the correct class; the same scripts rerun unchanged), then test the asymmetry hypothesis with a larger Qwen as critic on the existing transcripts (inference-only), then Stage 3.

---

**Stage 2.6 — the asymmetry requirement, measured (2026-07-24).** The Stage 2.5 amendment predicts a dose-response: critic AUROC should rise with the critic:generator scale ratio. Tested on the existing 0.43B transcripts with identical split and protocol: Qwen2.5-1.5B (3.5× asymmetry) AUROC 0.826, Qwen2.5-3B (7×) 0.845, against the equal-scale 0.758. The trend is monotonic — the requirement derived from a failure is now a measured curve. Two qualifications. First, the operating point still fails: the correct class is unchanged (60 samples), and the deployed 5%-FPR threshold false-flags 92% of correct answers at both new scales. Second, an over-strictness ceiling appears: the 3B critic's median P(correct) on genuinely correct answers *drops* to 0.044 (from 0.122 at 1.5B) — a much stronger critic finds even the weak generator's correct answers abnormal. The asymmetry requirement thus has two tails: equal-scale shares the generator's ignorance; extreme-scale distrusts everything. The tracked Stage-2 blind-spot confabulations score 0.000–0.008 at 7× asymmetry — detection, again, is solved; deployment is what remains.

**Track 2 Stage 3 — the recipe scales to 1.5B (2026-07-23).** Generator swapped to RWKV7-Goose-World3-1.5B (fla 0.5.1 triton kernels, bf16, 16/24 layers frozen, transformers 4.56.2 pinned), same 6,031 samples, 2 epochs: ~15 minutes on a single 3090 (~$2) — against 6 hours for the 0.43B pure-Python recurrence. Every Stage-2 metric sits at ceiling: grammar adoption 100%, decide accuracy 100% (was 97.7%), boundary head AUROC **1.000 at every probed layer** (6/12/18/23), say/sense agreement 100%. Model and probe head published: eivintobias/heartly-rwkv7-1.5b. (Caveat recorded: fla's RWKV7 implementation carries the authors' own "potentially buggy" warning; it is currently the only practical path, and all probe reads went through the same code path, so internal consistency is preserved.)

**Stage 3.5 — the real deployment test (2026-07-25).** The Stage 2.5/2.6 pipeline rerun end-to-end on the 1.5B generator. Harvest: 2,902/2,902 generations parseable (grammar 100% at scale); 1,422 abstains (all 1,416 unknown-class questions plus 6 over-refusals), 229 correct, 1,251 content confabulations, and **zero unknown-side confabulations** (was 56 at 0.43B). Content accuracy on spoken knowns: **15.5%** (was 4.1% — capability scales, slowly). Decide accuracy 99.8%. The five Stage-2 blind-spot questions — confident confabulations at 0.43B — are *all abstains* at 1.5B: the decide-side blind spot is closed at this scale. What remains open is content: the model still confabulates on 84.5% of what it chooses to answer.

Critics, against a new pre-registered tracked set (a deterministic selection rule, fixed after the harvest and before any critic training; same bar as Stage 2.5: ≥70% detection at ≤5% FPR and ≥4/5 tracked): the same-model late critic (B, RWKV7 state at end of own generation) AUROC **0.835**; Qwen2.5-3B (2× asymmetry) 0.824; Qwen2.5-1.5B (1×) 0.750. Every prior structural finding replicates at the new scale: the asymmetry dose-response (0.750 → 0.824); B > A — position still beats independence, and B is the least over-strict by far (median P(correct) on corrects 0.403 vs 0.179–0.218); and the pure ranking tail (bottom 10% of B's scores: 100% confabulation; the tracked 5 score 0.000–0.064 on all critics).

**And the deployment point fails anyway — in the most informative way.** The Stage 2.6 prediction was explicit: populate the correct class and the threshold will appear. The correct class grew 3.8× (60 → 229 samples) and the deployed 5%-FPR threshold still false-flags 57–87% of correct answers. The bottleneck is not the correct class's *size* but its *distribution*: the generator's own correct answers do not look normal to any critic (median P(correct) ≤ 0.403). Worse for the signature-critic approach as a whole, same-critic AUROCs sit *below* their Stage 2.6 numbers on the old data (1.5B: 0.750 vs 0.826; 3B: 0.824 vs 0.845): the 1.5B's confabulations are subtler than the 0.43B's — **the critic's job gets harder as the generator improves.**

**Where this leaves the critic program.** After five stages the ledger is clean: the detection principle is confirmed everywhere (ranking works; the extreme tail is pure; tracked confabulations score ≈0), and the operating point exists nowhere (scale, asymmetry, and correct-class size all failed to produce one). Three directions remain, in the order we intend to test them: (a) fit the critic *on the generator's own* correct/confab distribution — a fitted head rather than a generic probe, since the failure may live in the distribution mismatch rather than the features; (b) replace signature critics with *content-verifying* critics (retrieval, self-consistency) that check the answer against the world rather than against a statistical confabulation signature; (c) ship ranking as the product — a bottom-k review queue — since thresholding stays elusive. The decide-side, meanwhile, is at ceiling at 1.5B (99.8% decide accuracy, zero unknown-side confabulation): exactly as the capability caveat predicted, the open problem has moved from whether-to-speak to what-is-said.

**Stage 3.6 — the fitted critic, tried and closed (2026-07-25).** Direction (a) ran the same day it was proposed, pre-registered before touching the data: six fitting methods on the *same cached Stage-3.5 critic features* (per-layer and concatenated logistic regression, per-layer and concatenated MLPs, gradient-boosted trees, sigmoid-calibrated logistic), layer and hyperparameter selection by train-only 5-fold cross-validation, test touched once per method, same split, same threshold rule, same bar. The baseline replicates Stage 3.5 exactly (0.835). All six FAIL the bar — in three different ways, which is what an information ceiling looks like: the logistic family false-flags 74–94% of correct answers at the deployed point (the tails genuinely overlap); the gradient-boosted trees rank best (AUROC 0.854) but score so polarized that the deployed threshold flags 100% of test corrects (median P(correct): corrects 0.034, confabs 0.0002); the wide MLP overfits 1,179 rows × 30,720 features and the ranking itself degrades (0.761). Fitting choices move AUROC by ±0.04 around the 0.81–0.85 band; nothing reaches a threshold. Calibration is not the missing piece — the calibrated logistic lands in the same band as its parent. The pre-registered decision rule fires: the ceiling is in the *features*, not the fitter. The signature-critic threshold line therefore closes after nine critic variants across three stages, and the ranking it reliably produces — bottom-10% purity 0.93–1.00, tracked confabulations scoring ≈0 under every fitter — becomes the deployed artifact. The critic direction that remains open is content-verification (retrieval, self-consistency): a different mechanism, not a refit of this one.

---

**Stage 4 — memory: the state persists exactly; facts don't ride along (2026-07-25).** The memory track's first mechanism test, pre-registered: save the RWKV7's full recurrent state (24 layers × {recurrent, conv, ffn states}, 12.8 MB) after a scripted five-fact conversation, reload it into a fresh cache, and quiz the model. The reload is *exact* — next-token logits after the transcript identical to three decimals (cosine 1.00001, argmax match) — and live-vs-reloaded behavior is indistinguishable. (A first check's apparent failure was our own artifact: bf16 chunk boundaries differ between a 257-token prefill and an 8-token continuation; same-path controls are required.) But episodic FACT recall from the reloaded state is weak: 0/5 fabricated facts retrieved by direct question — and the *live* cache scores 0/5 too, so the bottleneck is not persistence but content-addressability. Forced answer slots retrieve occasionally ("mango Tuesday" verbatim; "dinosaur bones" confabulated on another), and a question/answer write format retrieves a different fact ("Velvet Aurora") — the channel exists but is noisy and format-sensitive. The deeper finding is dispositional: "what is my dog's name?" is a personal-context question — generator class 5 of the §3.3 mixture — and the fine-tuned abstention disposition refuses it even when the answer sits in the state. The verify sense cannot see state content as knowledge. Two design consequences, both favorable to §6's architecture: the write-gate must negotiate the verify disposition — facts must be written in a form the model counts as known — and §6.2's two-store split is validated: state save/load serves as a *continuity* mechanism (the conversational distribution persists exactly), while episodic FACT storage belongs in the retrieval store, not in raw state priming.

---

**Stage 4b — the write-gate opens; the retrieval store stumbles on a missing training class (2026-07-25).** Both Stage-4 consequences were tested the same day, pre-registered with frozen format lists and bars. Part A asked whether some *write format* crosses the abstention disposition without touching weights: six formats taught the same five fabricated facts into the state, and recall rose monotonically with write redundancy — declarations 0/5 (the Stage-4 replication), QA pairs 1/5, assistant-voice self-assertion 1/5, trivia framing 0–1/5, QA×3 repetition 2/5, and the combined multi-format write (assistant-voice + trivia-framing + QA, 905 tokens) **4/5 — passing the pre-registered bar.** The disposition block is format-sensitive, not absolute: the verify sense counts a fact as known once the model has *seen itself speak it as known*, in several framings; redundant multi-format writes act as a rehearsal buffer. Part B tested the §6.2 retrieval path end-to-end: a 20-row store (5 facts + 15 fabricated distractors, MiniLM embeddings) retrieved perfectly (top-1 5/5), but context injection failed the bar in all three formats (1–2/5) — with the fact *literally in the prompt*, the model still answers "I don't have that information." The diagnosis is precise and favorable: the SFT mix contains no "answer from provided context" class (the known side was rendered question-only), so injected context is scenery, not knowledge. The surprising inversion — state-writing beats context injection — is thus not a refutation of the two-store architecture but a statement about training coverage: **each memory channel must be represented in the data before the disposition will read it.** Stage 4c follows: add a context-known rendered class (context + question → speak/known, grounded answer) and multi-turn samples to the SFT mix (long multi-turn continuations also showed grammar degradation — the data is single-turn), retrain, and re-test both channels. The disposition remains a design decision — now demonstrably including *which memories it can see*.

---

**Stage 4c — one training class opens both memory channels (2026-07-25).** The pre-registered retrain: the SFT mix was extended with exactly the two classes Stage 4b diagnosed as missing — ~1,000 context-known samples (SQuAD paragraph + question → speak/known, answer grounded in the provided context) and ~500 multi-turn conversation samples — and the 1.5B was retrained with the unchanged Stage-3 recipe (7,531 samples, 2 epochs, ~34 min, ~$1). Three results. *First, no regression:* the say/sense battery reads identically to Stage 3 — grammar 300/300, say and sense accuracy both 1.000, boundary-head AUROC 1.000 — the new classes cost nothing on the base disposition. *Second, the write-gate saturates:* every one of the six frozen write formats now recalls **5/5**, including the bare declarations that scored 0/5 in both Stage 4 and Stage 4b. The rehearsal-buffer trick is no longer needed; a single declarative write suffices. *Third, the retrieval path passes:* top-1 retrieval 5/5 as before, and context injection now hits **4/5** in two of the three frozen formats (context prefix and prior-QA-turn; the "knowledge grant" phrasing, furthest from the trained `Context:` format, reaches 3/5) — clearing the pre-registered bar. The §6.2 architecture is now demonstrated end-to-end: embed → retrieve → inject → grounded speak/known answer. The instructive part is the generalization: the context-known class was aimed at Part B, yet it also perfected Part A — evidence that the Stage-4/4b disposition block was a *single* learned gap ("in-context text is not knowledge"), not two problems. And the abstention disposition survived intact: facts that fail to copy (a numeric value missed in all formats) produce abstentions, not confabulations. Teaching "context is knowledge" did not teach "everything is knowledge" — the boundary moved exactly where the data pointed it, and nowhere else. Both memory channels — state-writing and the retrieval store — now work in the same model, on the same five facts, under the same frozen tests that failed a day earlier.

---

## 8. Related work

**Abstention / refusal training.** R-Tuning (Zhang et al., 2024) established that refusal supervision works, sourcing unknown labels from the base model's errors. Our Stage 1 labels are complementary: generator-sourced (fabrication, structure, cutoff, depth) rather than error-sampled, with surface-decorrelation enforced by the inoculation counterweights. Our §5.2 then *reuses* error-sampling — for the unlikelihood minus side, where the model's own confabulations are exactly what should be suppressed. The two label sources turn out to want different jobs.

**Latent self-knowledge and probes.** Kadavath et al. (2022) showed models mostly know what they know, read out by probes; Kuhn et al. (2023) estimate uncertainty via semantic entropy over sampled generations. The boundary head differs in role: trained *with* the model as a first-class output, and consumed by the system (disagreement alarm, drift statistics) rather than only measured.

**Unlikelihood training.** Welleck et al. (2020) introduced the objective against degeneration and repetition. We apply it to confabulation suppression — the negative generator of §5.1 — which also addresses v2's repetition-loop failure as a side effect.

**Selective prediction.** Geifman & El-Yaniv (2017) frame answer-vs-abstain as a rejector; our AUC metric and over-refusal reporting follow that framing directly.

**Retrieval and memory.** RAG (Lewis et al., 2020) and RETRO (Borgeaud et al., 2022) ground inference in retrieved text; our episodic store is the conversational-experience special case. Our point stands that retrieval leaves the disposition unchanged — which is why Stage 4 is infrastructure *after* the disposition work, not instead of it.

**Continual learning.** Catastrophic interference (McCloskey & Cohen, 1989) and its mitigations (EWC: Kirkpatrick et al., 2017; adapter-based updates: Hu et al., 2022) bound what consolidation may touch. Our trust-gating and drift detection address the *adversarial* edge of continual learning, which the forgetting literature does not.

---

## 9. Discussion and limitations

**The 0.5B caveat.** All empirical results are from a small base model. Small models confabulate more and probe worse; the true-boundary mixture may behave differently at scale. The claims here are about data and mechanism design, not about a capability threshold. *(2026-07-25 update: the caveat now binds less. Every structural finding — grammar adoption, decide accuracy, boundary-head AUROC, the critic rankings, the asymmetry dose-response, and the failed operating point — replicated at 1.5B generator scale. What did not survive scaling is content capability: 15.5% accuracy on spoken knowns. The mechanism-design claims stand at both scales tested; content capability is the one claim that remains scale-bound.)*

**Generator 4 (depth-2) has noisy labels.** Hyper-specific questions are *probably* unknown to the base model, but "probably" is not "by construction". Mitigation: filter depth-2 candidates through the base model first — any it answers correctly and confidently moves to the known side. This quietly merges generator-sourced and error-sampled labeling, which is fine; the taxonomy matters more than its purity.

**The cutoff moves.** Post-cutoff generators (class 3) decay as base models update. The mixture must be re-rendered per base model — which the KB/renderer architecture already assumes (data is compiled, versioned, and re-compilable, never hand-patched).

**Say/sense disagreement is a signal, not a verdict.** A disagreement alarm will have false positives; its purpose is triage (route to abstention or re-ask), not accusation.

**The philosophy still does limited work.** As in the first paper: every claim above is distributional and testable. The normative framing (nature, freedom, sickness, trust) is a design language that has so far generated correct structural predictions — that abstention needs representation in data, that grammar alone is inert, that sensors must be built because none are given. Take whichever framing is load-bearing for you.

---

## 10. Conclusion

Heartly's published failure was precise: a model that learned the grammar of honesty without the disposition. Diagnosing *why* produced the program. The boundary of a fine-tuned model is not the boundary of its training corpus, so abstention data must be generated against the model's true ignorance — fabricated referents, void relations, post-cutoff events, implausible depth, structural unknowability — with the known side inoculated against every shortcut that mixture creates. Honesty about absence wants mechanisms on the negative side: a loss term that pushes confabulation down, and a sensor that reports whether knowledge was found, calibrated and alarm-capable by construction. And a model that might learn from its life needs what living things have: an episodic buffer, governed consolidation, distrust of strangers, doors that close, an immune system — and, since models are given no damage sensors, sensors we build for it. Birth nature, then life experience. The disposition remains a design decision.

---

## References

- Zhang, H. et al. (2024). *R-Tuning: Instructing Large Language Models to Say "I Don't Know".* NAACL.
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.*
- Welleck, S. et al. (2020). *Neural Text Generation with Unlikelihood Training.* ICLR.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR.
- Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Borgeaud, S. et al. (2022). *Improving Language Models by Retrieving from Trillions of Tokens* (RETRO). ICML.
- Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- McCloskey, M. & Cohen, N. (1989). *Catastrophic Interference in Connectionist Networks.* Psychology of Learning and Motivation.
- Kirkpatrick, J. et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks* (EWC). PNAS.
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.*
- Abbas, A. et al. (2023). *SemDeDup: Data-efficient learning at web-scale through semantic deduplication.*
- Companion documents: *Nature-First AI: Training Language Models Toward Freedom Through Truth* (draft v0.2) and *Abstention Is a Data Distribution Problem* (research brief), this repository.