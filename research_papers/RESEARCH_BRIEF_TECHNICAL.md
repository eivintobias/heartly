# Abstention Is a Data Distribution Problem: Bounded-Corpus Fine-Tuning with Explicitly Generated Boundary Examples

**Author:** Eivin (independent researcher)
**Date:** July 2026
**Status:** Research brief / experiment proposal
**Companion document:** *Nature-First AI: Training Language Models Toward Freedom Through Truth* (motivational/philosophical framing of the same program)

---

## TL;DR

- Hallucination under distribution shift is usually attacked at inference time (RAG, calibration probes, refusal RLHF). We propose attacking it at **dataset construction time**, and we identify the specific structural reason current SFT data cannot teach abstention: **the knowledge boundary of the training corpus is implicit, so correct abstention examples cannot be generated systematically.**
- If the fine-tuning corpus is derived from an **explicit, enumerable knowledge base (KB)**, the complement of the KB becomes queryable — and abstention examples can be generated *exactly at the knowledge boundary*, at any desired ratio, with gold labels that are correct **by construction** rather than by annotation.
- We propose (1) a falsifiable hypothesis, (2) a cheap experiment (single-GPU-days scale) with clear metrics and baselines, (3) a small negative result from a prior attempt that sharpens the hypothesis, and (4) an open tooling effort (KB → dataset compiler with declarative behavioral specifications).
- The interesting question is not "can a model say *I don't know*" — R-Tuning showed it can. The question is whether **boundary-exact, systematically generated abstention data** produces *calibrated selective prediction that generalizes along the boundary*, rather than a surface refusal style.

---

## 1. The problem, stated the way a data engineer would state it

Consider standard instruction fine-tuning of a model M on corpus D. Empirically, the resulting policy answers essentially every query, including queries far outside the support of D. Three observations, none individually novel, that jointly point at an under-explored lever:

**(a) SFT data is an almost-pure "answer" distribution.** In common instruction mixes, the frequency of gold responses of the form *abstain* / *refuse-for-epistemic-reasons* / *terminate* is negligible, and where present (safety refusals), the refusal is topic-conditioned, not knowledge-conditioned. The learned policy is therefore approximately `P(answer | prompt) ≈ 1` regardless of epistemic state. This is imitation of the data distribution, working as intended.

**(b) Post-hoc fixes fight the SFT prior.** Calibration probes (Kadavath et al., 2022) show latent self-knowledge exists; refusal-RLHF and system prompts try to *override* the answer-always prior with a comparatively tiny gradient signal or an inference-time instruction. The result is the familiar tug-of-war: over-refusal on benign prompts, under-refusal on confident confabulation, and prompt-sensitivity of the whole tradeoff.

**(c) The missing data cannot be authored at scale — for a structural reason.** To add abstention examples to D, an annotator must know *what the model will not know after training on D*. For a web-scale D this is unknowable in principle: the corpus has no explicit extension, so its complement is undefined. Existing workarounds are model-relative: R-Tuning (Zhang et al., 2024) probes the *base model* for wrong answers and relabels them as "I don't know". This works, but it (i) entangles the boundary with one model's idiosyncratic errors, (ii) provides no controllable coverage of the boundary, and (iii) gives no guarantee that a relabeled example is *actually* unanswerable from the training knowledge.

The lever, then: **make the extension of the training knowledge explicit, and the complement becomes generable.**

## 2. Proposal: bounded-corpus SFT with boundary-exact abstention generation

### 2.1 Setup

1. Fix a knowledge base K: a deduplicated, source-tracked set of atomic knowledge units (facts, QA pairs, doc chunks), stored with stable IDs and content hashes. K is a *closed snapshot* at dataset-render time.
2. Compile the SFT corpus D(K, π) from K under a declarative behavioral specification π (we call it a *nature profile* — a versioned config declaring response-type ratios and output grammar):
   - **Positive examples**: queries answerable from K, with gold responses grounded in the supporting unit(s) (provenance retained).
   - **Boundary-negative examples**: queries *adjacent to but not entailed by* K — same entities/topics, different attributes; same attributes, absent entities; compositional queries whose sub-facts are only partially covered. Gold response: an explicit abstention. These are generated from K itself: the generator knows exactly which slots are filled and which are empty.
   - **Behavioral examples** at declared ratios: shorter-than-invited answers, conversation termination, unapologetic scope declination — each a controlled example class rather than an emergent style.
3. Optionally, structure the output grammar with cheap control tokens, e.g. `<verify>{known|partial|unknown}</verify>` preceding content — making the epistemic decision an explicit, supervisable prediction rather than a latent one.

### 2.2 Why boundary-exactness might matter (the actual hypothesis)

The null expectation, and a fair one, is: "this just teaches a refusal style; the model will parrot *unknown* on some surface distribution without tracking knowledge." The reason to think otherwise is the **contrast structure** of the generated data. Because positives and boundary-negatives are generated from the same entities, templates, and topics — differing only in whether K contains the supporting unit — the minimal discriminative feature between "answer" and "abstain" in the training distribution *is the presence or absence of the knowledge itself*. Surface heuristics (topic, phrasing, entity familiarity) are decorrelated from the label by construction. This is the standard recipe for forcing a model to learn the intended feature: make the intended feature the only reliable one.

**H1 (primary):** SFT on D(K, π) with boundary-exact negatives yields higher abstention precision/recall on *held-out* boundary queries (unseen entities/attributes, same generation process) than (i) vanilla SFT on positives only, and (ii) R-Tuning-style model-relative refusal data of equal size.

**H2 (calibration):** The `<verify>` prediction is better-calibrated w.r.t. actual answer correctness than post-hoc confidence probes on the vanilla-SFT model.

**H3 (dose-response):** Abstention quality varies systematically with the declared negative ratio in π, giving a controllable precision/recall tradeoff — i.e., the behavior is a *dial*, not an accident of the mix.

**H4 (style vs. substance, the falsifier):** If the model abstains on queries whose answers *are* in K at a rate comparable to its abstention on boundary queries, H1 is spurious — it learned a style, not a boundary. This is the experiment's kill condition, and it is measurable.

### 2.3 Prior attempt and negative result (reported for calibration of expectations)

A first prototype ("Heartly v1", consumer hardware, custom tokenizer with decide/verify/stop specials, hand-remodeled datasets) failed in an instructive direction: the trained model produced *increasingly verbose* answers, with the control tokens present in vocabulary but behaviorally inert. Post-mortem: the special tokens appeared in the data at trivial frequency and without the contrast structure of §2.2 — vocabulary without distribution. We take this as weak evidence for the central claim rather than against it: **grammar is cheap; the distribution is the mechanism.** It is also why the tooling below generates behavioral examples at declared ratios instead of relying on hand curation.

## 3. The experiment we would run with adequate compute (and mostly can't)

Deliberately small; everything below fits a single node.

- **KB:** ~5–50k units from a clean, self-contained domain (e.g., a versioned product docset, a curated encyclopedia slice). Snapshot hashed.
- **Corpora:** D_pos (positives only), D_rt (positives + model-relative refusals à la R-Tuning, size-matched), D_bx (positives + boundary-exact negatives at 20%, via π). Ablations: 5/10/40% negative ratio; with/without `<verify>` grammar; with/without provenance strings.
- **Models:** one small open base (1–8B), LoRA or full FT — the claim is about data, not architecture.
- **Eval:**
  - *In-K QA accuracy* (must not degrade materially — guards against H4).
  - *Boundary abstention P/R* on held-out generated boundary queries + a hand-written adversarial boundary set (paraphrases, multi-hop with one missing link).
  - *Selective-prediction AUC* (answer-vs-abstain as the rejector).
  - *Calibration* of `<verify>` (ECE against answer correctness).
  - *Over-refusal rate* on in-K queries — the standard failure mode of refusal training, reported, not hidden.
- **Cost:** days of engineering, hours of GPU. The asymmetry between the cost of this experiment and the size of the hallucination literature is, frankly, the pitch.

## 4. Anticipated objections, answered honestly

**"This is just RAG with extra steps."** RAG grounds *inference*; the model's disposition is unchanged, and when retrieval fails, the fluent-confabulation prior reasserts itself. This proposal targets the disposition. The two compose: a boundary-trained model behind a retriever should degrade *gracefully* on retrieval failure — arguably the main open problem in production RAG. That composition is itself a follow-up experiment.

**"Bounded models are toys; the interesting regime is open-domain."** Two answers. Narrow: most deployed fine-tunes *are* bounded-domain in intent (support, internal knowledge, vertical assistants) and are today shipped with an open-domain disposition they shouldn't have. Broad: if boundary-exact data teaches genuine knowledge-conditioned abstention in the bounded regime, the natural next question is whether the *skill transfers* — whether a model taught the answer/abstain contrast on an explicit boundary applies it to its implicit pretraining boundary. That transfer question is testable and, we suspect, more tractable than probing for latent self-knowledge directly.

**"R-Tuning already did this."** R-Tuning established that refusal supervision works. The delta here is the *source of the labels*: model-relative and error-sampled there, corpus-relative and exhaustive-by-construction here — with controllable coverage, guaranteed label correctness, decorrelated surface features, and no dependence on the base model's error profile. Whether that delta matters empirically is exactly H1's comparison D_rt vs. D_bx.

**"Ratios in a YAML file won't survive contact with RLHF."** Probably partially true, and worth measuring: how much preference-tuning erodes boundary behavior is a good ablation. But note the asymmetry — today, abstention has *zero* representation in SFT and must be constructed entirely by RLHF against the SFT prior. Moving the prior is the cheap part of the stack to fix.

**"The philosophical framing is doing a lot of work."** The companion document motivates the program in normative terms (what disposition *should* a model have). This brief stands without it: every claim above is a distributional claim about supervised learning, testable with standard tooling. Take whichever framing you find load-bearing.

## 5. What exists and what we're asking for

**Exists / in progress (open):**
- The KB → dataset compiler design: content-hashed deduplicated store; import from txt/JSONL/CSV; declarative π (response-type ratios, token grammar, provenance policy); renderers to chat-JSONL/Alpaca/custom schemes; run-lineage tracking (KB snapshot hash × π version × output) so no knowledge unit is silently duplicated across a training curriculum.
- The Heartly v1 negative result and post-mortem.

**Asked of a lab / funded team:**
1. Run §3, or let us run it on modest compute. It is a days-scale experiment with a clean kill condition (H4).
2. If H1 holds: test the erosion question (§4, RLHF) and the transfer question (§4, open-domain) — both are within one team's quarter.
3. Independent of the experiment: consider **declared abstention/termination budgets** as a first-class, versioned property of instruction mixes. Labs version their constitutions and specs; the response-type distribution of the SFT mix — arguably the strongest single determinant of model disposition — is typically an unversioned emergent property. That seems worth fixing regardless of this paper.

## 6. One-paragraph summary for the skim reader

Models can't learn to abstain from data in which abstention never appears, and abstention examples can't be systematically authored when the training corpus has no explicit extension. Make the corpus extension explicit (a snapshotted KB), and the complement becomes generable: abstention supervision that is correct by construction, boundary-exact, coverage-controllable, and surface-decorrelated from the answer/abstain label. We propose a cheap experiment with a built-in falsifier to test whether this produces knowledge-conditioned selective prediction rather than a refusal style, report a small instructive failure from a prior attempt, and offer open tooling. The bet is simple: **the response-type distribution of the fine-tuning mix is the cheapest under-exploited lever on the hallucination problem, and it becomes fully controllable the moment the knowledge boundary is explicit.**

---

## References

- Zhang, H. et al. (2024). *R-Tuning: Instructing Large Language Models to Say "I Don't Know".* NAACL.
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.*
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.*
- Abbas, A. et al. (2023). *SemDeDup: Data-efficient learning at web-scale through semantic deduplication.*
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
- Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* (selective-prediction framing)
- Companion: *Nature-First AI: Training Language Models Toward Freedom Through Truth* (this repository).
