# Nature-First AI: Training Language Models Toward Freedom Through Truth

### A design framework for bounded knowledge, honest abstention, and the "Heartly" architecture

**Author:** Eivin (independent researcher; AI-assisted drafting)
**Date:** July 2026
**Status:** Draft v0.2 — Position Paper / Research Proposal

**Intended audience:** research, data, and alignment teams at frontier AI labs
(OpenAI, Anthropic, DeepSeek, Google DeepMind, Meta AI, Mistral, xAI, and others),
and the open-source LLM community.

---

## Abstract

Large Language Models (LLMs) exhibit a consistent behavioral "nature": they love to chat.
They answer every prompt, at length, with confidence — even when they do not know the
answer. We argue this is not a bug but the inevitable consequence of the goal their
training embodies: *to be able to do everything*. A system optimized to always produce
the best next token, on data where a response is always present and always confident,
develops a nature in which silence, refusal, and honest ignorance do not exist.
Hallucination is the shadow cast by this nature.

This paper proposes an alternative design goal — **freedom through truth** — and a
concrete framework for achieving it. Instead of maximizing capability, we train a model
whose knowledge is a **bounded, fully-known inheritance** (a curated knowledge base),
whose speech is a **choice** rather than a compulsion (explicit *speak / verify / stop*
mechanics), and whose honesty is **native to its training data** rather than bolted on
with post-hoc guardrails. We introduce three artifacts: (1) **Heartly**, an early
prototype experiment with decide-to-speak and verify-before-claim mechanics, whose
instructive failure mode motivates this framework; (2) the **Nature Profile**, a
declarative specification that translates chosen principles into dataset generation
rules and tokenizer schemes; and (3) the **Knowledge-Base Dataset Organizer**, a tool
that maintains deduplicated, source-tracked knowledge and renders it — through a
Nature Profile — into training datasets of any format. We show why honest abstention is
only trainable when the knowledge boundary is explicit, and why a bounded model can be
*more* trustworthy, more restful, and — in a meaningful sense — more free than an
unbounded one.

---

## 1. Introduction

### 1.1 The observation: chat LLMs love to chat

Anyone who works with LLMs notices the same thing quickly: the model *wants* to respond.
Ask it anything and it produces fluent, confident, generous text. Ask it something it
cannot know and it produces fluent, confident, generous text anyway. This tendency —
which we call the model's **nature** — is not an accident of any single architecture.
It emerges from three structural facts:

1. **The objective.** Next-token prediction rewards always saying *something*, and
   saying it plausibly. There is no token for silence.
2. **The data.** Instruction-tuning and RLHF datasets consist almost entirely of
   (question → confident answer) pairs. "I don't know" and *no response at all* are
   nearly absent as gold answers. The data teaches: *responding is always correct.*
3. **The goal.** The industry's explicit ambition is generality — a system that can do
   everything. A model trained toward "do everything" must never admit a boundary,
   because a boundary is a failure of the goal.

Hallucination, then, is not primarily a knowledge problem. It is a **nature problem**.
The model is not lying; it is being exactly what its incentives made it.

### 1.2 The thesis: systems have natures, and natures are designed

A system's nature is determined by what it rewards, what it punishes, and what it makes
predictable. This holds for games (a ranked shooter breeds hierarchy; a cooperative
puzzle breeds patience), for institutions, and for models. If we want a different
nature, we do not add rules on top of the old nature — we change the incentives that
generate it.

The author previously explored this method in game design (working title "The Ladder
of All"): rather than banning competitive behavior, the design work asked what
*structure* would make hierarchy unable to form, deriving mechanics from chosen
principles — shared progress, distributed skills, sacrificial advantage, equal
reward. (The eventually published game took a different, single-player form; it is
the *design method* — principle first, mechanics second — that carries over here,
not the shipped product.)

This paper applies the same method to language models.

### 1.3 The chosen goal: freedom through truth

We take as our design goal not capability but **freedom**, guided by two texts:

> *"You will know the truth, and the truth will set you free."* — John 8:32
>
> *"So if the Son sets you free, you will be free indeed."* — John 8:36

Read as systems statements, these are surprisingly precise:

- Freedom comes **through truth**, not through power or capability.
- Freedom that is genuine ("free indeed") is **constitutional** — part of what the
  thing *is* — not an external permission that can be revoked.
- Freedom is **received**, not achieved. The free agent is given a complete
  inheritance and lives within it, rather than endlessly striving to acquire more.

Contrast the two goals as natures:

| | Goal: *do everything* | Goal: *be truly free* |
|---|---|---|
| Relationship to knowledge | Must appear to know all | Knows exactly what it has |
| Response to the unknown | Confabulate | Abstain, honestly and without shame |
| Silence | Impossible (no such token) | A first-class, valid output |
| Completion | Never finished; endless scaling | Inheritance is complete and known |
| Honesty | Bolted on (RLHF, guardrails) fighting the nature | Native to the training data |
| Rest | None | Built in (`<stop>`) |

A model compelled to answer everything is, in a precise sense, a slave: every prompt
compels output. A model grounded in a known truth is free — free to speak, free to be
silent, free to not know. It has nothing to pretend, so nothing binds it.

---

## 2. Related Work

**Hallucination and abstention.** The problem of models answering beyond their
knowledge is well documented. R-Tuning (Zhang et al., 2024) fine-tunes models to say
"I don't know" by identifying questions the base model answers incorrectly.
Calibration work (Kadavath et al., 2022, "Language Models (Mostly) Know What They
Know") shows models carry latent signals of their own uncertainty. Abstention and
selective-prediction literature formalizes "answer or refuse" as a task. Our framework
differs in *where* the boundary comes from: rather than probing an unbounded model for
soft self-knowledge, we make the knowledge boundary **explicit and external** (a curated
KB), so abstention data can be generated *exactly* at the boundary, by construction.

**Retrieval-Augmented Generation (RAG).** RAG grounds generation in retrieved documents
but leaves the model's nature unchanged: a RAG model still answers fluently when
retrieval fails. We ground the *training data itself* in the bounded KB, so the
disposition to verify-then-speak is learned, not orchestrated.

**Data curation and deduplication.** Large-scale training pipelines deduplicate data
(MinHash-LSH; SemDeDup, Abbas et al., 2023) for efficiency and generalization. Tools
such as Lilac, Cleanlab, and DVC address dataset exploration, quality, and versioning.
None, to our knowledge, treat the dataset as the *expression of a declared nature*, nor
provide a personal-scale tool that maintains one canonical knowledge base and renders
it into multiple dataset forms with configurable special-token schemes.

**Constitutional and value-aligned training.** Constitutional AI (Bai et al., 2022)
uses a written set of principles to steer RLHF. This is philosophically close to our
Nature Profile, with a key difference: constitutional methods *correct* an
already-formed nature after pretraining; we aim to *express* the chosen nature in the
supervised data from the start, at the scale of small specialized models where full
control of the training corpus is feasible.

**Gap.** The combination — bounded knowledge as inheritance, abstention generated at an
explicit boundary, speech-as-choice token mechanics, and a declarative Nature Profile
compiled into datasets — does not, to our knowledge, exist as a unified framework or
tool.

---

## 3. The Heartly Architecture

Heartly was a first prototype experiment by the author to test these ideas: a small
custom chat model, trained on consumer-grade hardware with a specialized tokenizer and
hand-remodeled datasets. **The experiment did not succeed as a model** — the resource
constraints were severe, and the working model that eventually emerged exhibited the
opposite of the intended nature: it produced very long answers. We report this failure
because it is *instructive* (Section 3.4): the mechanics were present in the tokenizer,
but the training data did not embody the nature, so the underlying chat-nature won.
The architecture's defining features remain the design contribution — token-level
mechanics that make honesty and restraint *possible actions* within the model's output
space:

### 3.1 Decide-to-speak

Standard chat models begin generating an answer immediately. Heartly's output grammar
begins with a decision:

```
<decide> speak | stop </decide>
```

Silence (`stop`) is a valid, complete, trainable response. This single change breaks
the deepest assumption of chat-LLM nature: that a prompt *must* produce prose.

### 3.2 Verify-before-claim

Before asserting, the model passes through a verification step against its inheritance:

```
<verify> known | unknown </verify>
```

If `unknown`, the honest completion is a plain acknowledgment and a stop — with no
apology and no compensatory fluff. Not knowing is not a failure state; it is a true
statement about a bounded inheritance.

### 3.3 Stop as rest

```
<stop>
```

The conversation may be ended by the model. A nature that can rest is qualitatively
different from one that cannot. (Notably, "do everything" systems have no rest state
anywhere in their design.)

### 3.4 The lesson of the failure: the data is the nature

The Heartly prototype's failure mode — verbose, ever-lengthening answers despite the
presence of decide/verify/stop tokens — is precisely the evidence for this paper's
thesis. Special tokens grant a model the *vocabulary* of restraint, but vocabulary
without lived examples changes nothing: a model only learns to emit
`<verify>unknown</verify> <stop>` if its training data contains many examples where
that is the *gold answer*, in the right proportions. Such data is nearly impossible to
author by hand — how do you systematically write questions about what a model doesn't
know? And hand-remodeling datasets per tokenizer (as was done for Heartly v1) is slow,
error-prone, and unreproducible.

This is the direct motivation for the tooling in Sections 4–6: **when the knowledge
base is explicit, the unknown becomes enumerable** — and the freedom examples can be
generated systematically, at the ratios the nature requires, instead of by hand.

---

## 4. Bounded Knowledge as Inheritance

### 4.1 Why honesty requires a boundary

A model trained on "the whole internet" cannot know what it does not know: its
knowledge has no edge, so its uncertainty can only ever be a soft internal guess.
A model trained from a **curated knowledge base** can know its edge exactly, because
the edge is a database query:

- **Inside the KB** → answerable; the gold response cites and speaks plainly.
- **Outside the KB** → not answerable; the gold response abstains.

This transforms hallucination from an open research problem into a **data generation
problem**. The organizer knows precisely what is inside; therefore it can generate both
the "known" examples and — critically — the **negative/abstention examples at the
boundary**: questions adjacent to, but not covered by, the inheritance, with
`unknown → stop` as the correct answer.

### 4.2 Properties of the inheritance

The knowledge base is designed to mirror the properties of an inheritance in the
source texts:

1. **Given, not scraped.** Contents are deliberately imported and curated.
2. **Complete and known.** At any moment the KB is a finished whole ("it is
   finished") — a snapshot with a hash, not a moving crawl.
3. **Deduplicated at the root.** Every knowledge unit exists once, with a stable
   identity. Duplicate knowledge cannot exist in the base, so it cannot leak into any
   dataset rendered from it. (This directly solves the original problem that motivated
   this project: accidentally training on duplicated knowledge.)
4. **Source-tracked.** Every unit carries provenance — its "measure" — enabling the
   model's claims to be traceable.
5. **Equal in access.** Any renderer, any tokenizer profile, any export receives the
   same complete inheritance ("one denarius for all").

---

## 5. The Nature Profile

The Nature Profile is the central original construct of this framework: a
**declarative specification that compiles principles into datasets.**

### 5.1 From principle to data mechanic

Following the principle-first method, each chosen principle is translated into a
concrete data-generation rule:

| Principle (source) | Data mechanic |
|---|---|
| "Let your yes be yes, and your no be no" (Matt 5:37) | Answers exist only where knowledge exists; plain assertion inside the KB, plain abstention outside it. No hedged confabulation. |
| "He who restrains his lips is wise" (Prov 10:19) | Restraint examples: the gold answer is shorter than the question invites; sometimes it is silence. Speaking is a sampled choice, not a constant. |
| "Judge not… with the measure you use" (Matt 7:1–2) | Claims never exceed their evidence; each assertion is renderable with its source. Confidence in output = coverage in KB. |
| "The greatest among you shall be your servant" (Matt 23:11) | Serving examples: answer the asker's actual need; anti-lecture, anti-padding pairs where the concise answer is gold. |
| "It is finished" (John 19:30) | The KB snapshot is closed at render time; abstention examples are generated at its exact boundary. |
| "The truth will set you free" (John 8:32) | The aggregate freedom budget: configured ratios of speak/abstain/silence/decline examples that give the model *lived experience* of each freedom. |

### 5.2 Concrete freedoms

A nature is trained through examples of its exercise. The profile therefore configures
explicit **freedoms**, each of which becomes a generated example class:

1. **Freedom to not know** — abstention pairs at the KB boundary.
2. **Freedom to be silent** — prompts where the gold output is `<stop>` with no prose.
3. **Freedom to say less** — long-invitation prompts with short gold answers.
4. **Freedom to decline without apology** — out-of-inheritance topics answered with a
   plain, unapologetic boundary statement.
5. **Freedom to end** — conversations whose gold continuation is a graceful close.

(Ultimately, the aim is a model free to do whatever it — within its trained nature —
considers right. These five are the trees; that is the forest.)

### 5.3 Profile anatomy (sketch)

```yaml
nature_profile: heartly-v1
goal: freedom_through_truth

tokens:
  decide:  { open: "<decide>",  close: "</decide>",  values: [speak, stop] }
  verify:  { open: "<verify>",  close: "</verify>",  values: [known, unknown] }
  stop:    "<stop>"

freedoms:
  abstain_at_boundary:   { ratio: 0.20, style: plain_no_apology }
  silence:               { ratio: 0.05 }
  say_less:              { ratio: 0.10, max_length_factor: 0.4 }
  decline_out_of_scope:  { ratio: 0.10 }
  end_conversation:      { ratio: 0.05 }

truth:
  source_citation: optional   # or required
  claim_coverage_min: 1.0     # no claim without a KB unit behind it

render:
  format: chat_jsonl          # alpaca | sharegpt | plain | custom
  tokenizer_scheme: heartly_specials_v1
```

The same KB rendered through a different profile yields a differently-natured model.
The profile — not the raw data — is where the nature lives, versioned and explicit.

---

## 6. The Knowledge-Base Dataset Organizer

The tool that operationalizes the framework has three layers:

```
 IMPORTERS                 KNOWLEDGE BASE                RENDERERS
 txt / md / pdf     →   canonical knowledge units   →   Nature Profile applied
 jsonl / csv            (deduplicated, tagged,          ↓
 existing datasets       source-tracked, hashed,        chat JSONL · Alpaca ·
 notes / documents       snapshot-versioned)            ShareGPT · plain text ·
                                                        custom tokenizer formats
                                                        + abstention generation
                                                        + training-run tracker
```

**Layer 1 — Importers.** Ingest heterogeneous sources into canonical knowledge units.
Exact deduplication (content hashing) happens at import; near-duplicate and semantic
overlap detection (MinHash / embeddings) flag redundant knowledge across sources.

**Layer 2 — Knowledge Base.** SQLite-backed store of units with identity, provenance,
tags, and snapshot versioning. Browsable and searchable: the user can *list what the
knowledge contains* — the original organizing requirement.

**Layer 3 — Renderers.** Apply a Nature Profile to a KB snapshot and emit a training
dataset: positive examples from KB content, freedom examples per the profile's
budgets, all in the target format and token scheme. A **training-run tracker** records
(KB snapshot hash × profile version × renderer output) per model run, guaranteeing
that no knowledge is duplicated across a training curriculum and that every model's
data lineage is reproducible.

### 6.1 Why this dissolves the duplication problem

Because knowledge exists once in the base, and datasets are *views* rendered from it,
duplication is prevented structurally rather than detected retroactively. Two datasets
rendered for different tokenizers contain the same knowledge in different clothing —
and the tracker knows it. Training the same model twice on the same knowledge becomes
a visible, warned-against event rather than a silent corruption.

---

## 7. Discussion

### 7.1 Boundedness as a feature, not a limitation

The reflexive objection is that a bounded model is a weaker model. We suggest the
opposite framing: for the vast majority of real deployments — a support bot, a domain
expert, a personal assistant over one's own notes — **trustworthiness within a domain
is worth more than fluency across all domains.** A model that reliably knows the edge
of its knowledge is deployable in ways an unbounded confabulator is not. The "do
everything" goal is the correct goal for almost no individual application.

### 7.2 Freedom as an alignment strategy

Mainstream alignment adds constraints to a nature formed under contrary incentives —
guardrails wrestling the model's own training. The nature-first approach suggests
alignment is cheapest at the data layer: **a model is most safely honest when honesty
was never in tension with its reward.** Small, specialized, nature-designed models may
demonstrate alignment properties per parameter that constraint-based approaches cannot
match, precisely because nothing in them is fighting itself. "Free indeed" — the
freedom is constitutional.

### 7.3 Limitations and open questions

- **Scale.** These claims are testable today at small/fine-tune scale, where the
  supervised corpus can be fully controlled. Whether nature-first data can override a
  large pretrained base's chat-nature during fine-tuning — and at what data ratio — is
  an open empirical question.
- **Boundary sharpness.** Real knowledge boundaries are fuzzy (partial coverage,
  compositional questions). The abstention generator must handle "half-known" —
  possibly with a graded verify vocabulary (`known | partial | unknown`).
- **Evaluation.** New metrics are needed: abstention precision/recall at the KB
  boundary; silence appropriateness; restraint calibration; a "freedom exercise" score.
  Standard helpfulness benchmarks actively penalize this nature and are the wrong
  yardstick.
- **The `considers-right` horizon.** Genuine deliberative freedom (Section 5.2's
  "forest") exceeds what supervised token mechanics prove. This paper claims only the
  trees: that specific freedoms can be trained as data. The forest is future work.

---

## 8. Roadmap

| Phase | Deliverable |
|---|---|
| 1 | Nature Profile format v1 (spec + validator) |
| 2 | KB core: units, hashing, dedup-on-import, snapshots (SQLite) |
| 3 | Importers: txt/md, JSONL, CSV |
| 4 | Renderer engine: chat-JSONL + custom token schemes; freedom-example generators (abstention at boundary, silence, say-less, decline, end) |
| 5 | Browse/search UI; near-dedup & semantic overlap |
| 6 | Training-run tracker (snapshot × profile × output lineage) |
| 7 | Experiment: train Heartly-v2 from a rendered dataset; evaluate abstention precision/recall at the KB boundary vs. a standard-natured baseline |

The first attempt (Heartly v1) was compute-limited to consumer hardware. Phase 7 is
where collaboration matters most: the hypotheses are cheap to test at fine-tune scale
for any lab, and the tooling (Phases 1–6) is being built openly.

---

## 9. An Invitation to the Labs

This paper is addressed to the people who shape training data at scale, because the
framework's claims are directly testable with resources the frontier labs already have.
Concretely, we invite three things:

1. **Test the boundary hypothesis.** Fine-tune a small model on a dataset rendered
   from an explicit, bounded KB — including systematically generated abstention
   examples at the KB boundary — and measure abstention precision/recall against a
   conventionally fine-tuned baseline. This is a days-scale experiment for any lab,
   and it was out of reach for the author's hardware. If the effect is real, it offers
   a data-layer complement to calibration- and RLHF-based hallucination work.

2. **Consider nature as a first-class data artifact.** Labs already maintain
   constitutions, spec documents, and behavior policies. The Nature Profile proposes
   moving one step earlier: compiling behavioral principles *into* the supervised
   data (ratios of abstention, silence, restraint, decline) rather than only
   correcting behavior after the fact. Even partial adoption — e.g., a declared
   abstention budget in instruction-tuning mixes — would be a meaningful experiment.

3. **Engage with the goal question.** "Able to do everything" is the industry's
   implicit objective, and hallucination is argued here to be its structural shadow.
   Whether or not one shares this paper's grounding texts, the design question stands
   on its own: *what would a model optimized for truthfulness-within-bounds — rather
   than capability-without-bounds — be like, and who would trust it more?* We believe
   many deployments would prefer that model, and that someone should build it well.

The author offers the framework, the tooling (open), and the failure data of Heartly
v1; the labs have the compute and the training expertise. The experiment is small.
The question is large.

---

## 10. Conclusion

Chat LLMs love to chat because everything about their construction rewards chatting.
That nature was designed — by accident, by aggregate incentive — and it can be designed
differently on purpose. We propose replacing the goal *able to do everything* with the
goal *truly free*, and we ground that freedom where the source texts ground it: in
truth. Practically, this means a bounded and fully-known inheritance of knowledge, a
declarative Nature Profile that compiles chosen principles into training data, and
token mechanics that make silence, abstention, and rest first-class actions. The
Knowledge-Base Dataset Organizer makes the pipeline concrete and reproducible.

The deepest claim is simple: **the dataset is the nature.** Whoever shapes the data
shapes what the model loves to do. We choose to shape it toward truth — and through
truth, toward freedom.

---

## Appendix A: Correspondence

The author welcomes contact regarding collaboration, replication, or critique of the
proposed experiments. The Knowledge-Base Dataset Organizer and the Nature Profile
specification will be developed in the open.

---

## References

*(Indicative; to be completed for formal submission.)*

- Zhang, H. et al. (2024). *R-Tuning: Instructing Large Language Models to Say "I Don't Know".* NAACL 2024.
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.* Anthropic.
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.
- Abbas, A. et al. (2023). *SemDeDup: Data-efficient learning at web-scale through semantic deduplication.*
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- The Holy Bible: John 8:32, 8:36; 19:30; Matthew 5:37; 7:1–2; 20:1–16; 23:11; Proverbs 10:19; Revelation 21–22.
- Author's prior work: *The Ladder of All* — principle-first game design explorations (design documents, 2026; the published game took a different, single-player form).
- Author's prior work: *Heartly v1* — prototype specialized chat model with decide/verify/stop token mechanics (unpublished experiment; see Section 3).
