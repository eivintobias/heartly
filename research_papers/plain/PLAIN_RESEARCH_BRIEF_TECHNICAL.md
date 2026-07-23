# The technical brief — the plain version

*Easy-to-read copy of `../RESEARCH_BRIEF_TECHNICAL.md` ("Abstention Is a
Data Distribution Problem"). This is the engineering argument: why "I don't
know" can't be learned from normal training data, and the exact fix.*

---

## The claim in three short sentences

1. Models can't learn to say "I don't know" from data where that never
   happens — and in normal training data it basically never happens.
2. You can't fix this by writing examples by hand, because you don't know
   what the model doesn't know.
3. But if the training data is built from an explicit knowledge base, then
   "what the model won't know" is just *what's not in the base* — and you
   can generate unlimited, perfectly-labeled "I don't know" examples
   automatically.

## Why the usual fixes don't really work

- **Asking nicely (prompts, system messages):** fights the model's whole
  training. It learned "always answer" from millions of examples; one
  instruction sentence doesn't stand a chance. Result: refuses too much on
  some days, confabulates on others.
- **Fine-tuning on refusals (like R-Tuning did):** this actually works a
  bit — they showed a model *can* learn to say "I don't know". But their
  labels come from the model's own mistakes, which is a messy sample of the
  boundary, with no control over coverage and no guarantee the question is
  truly unanswerable.
- **Probes that read the model's confidence:** good for measurement, but
  they don't change behavior by themselves.

Our claim: the cheapest place to fix this is **the training data itself** —
but only if the data's knowledge boundary is explicit. That's the whole
trick.

## The design, plainly

1. Build a clean knowledge base (KB): deduplicated facts with sources,
   snapshotted like a code release.
2. Generate two kinds of training examples from it:
   - **Answerable questions** (the answer is in the KB) → gold answer:
     answer it.
   - **Boundary questions** (same style, same topics, but the needed fact
     is NOT in the KB) → gold answer: a plain "I don't know."
3. Because both kinds look identical on the surface, the only way for the
   model to tell them apart is to check *whether it actually has the
   knowledge*. No shortcuts available. That's what forces real learning.

## The hypotheses (what we're betting on), in plain words

- **H1:** a model trained this way abstains correctly on new boundary
  questions — better than a model trained on answers only, and better than
  R-Tuning's mistake-based labels.
- **H2:** the model's `<verify>` marker becomes a trustworthy confidence
  signal.
- **H3:** you can *dial* the behavior: more abstention examples in the mix
  → more careful model. Behavior as a knob, not an accident.
- **H4 — the kill switch (important):** if the model also refuses questions
  it SHOULD know, it just learned a "refusal style", not a boundary. Then
  the whole thing failed and we say so. Every experiment in this project
  carries a falsifier like this.

## What we learned from the first failed attempt (v1)

v1 had the special tokens but almost no abstention examples. The tokens did
nothing; the model kept chatting. Lesson carved in stone: **grammar is
cheap, distribution is the mechanism.** It's not enough that the model
*can* say "I don't know" — it has to see many examples where that's the
right answer.

## The experiment this brief proposes

Small on purpose: one knowledge base (5–50k facts), one small open model,
three datasets (answers-only / R-Tuning-style / boundary-exact), and
measurements: does it still answer what it should? does it abstain where it
should? does it over-refuse (the H4 check)? Cost: days of engineering,
hours of GPU. The size of this experiment vs. the size of the hallucination
literature is, frankly, the pitch.

## Objections, answered shortly

- **"This is just RAG with extra steps."** No — RAG patches the input at
  runtime and the model's nature is unchanged; when retrieval fails it
  confabulates again. This trains the nature. (The two can work together.)
- **"Bounded models are toys."** Most deployed models ARE bounded-domain in
  practice (support bots, company assistants) but ship with an
  answer-everything nature they shouldn't have.
- **"R-Tuning already did this."** They proved refusals can be trained.
  We change *where the labels come from*: constructed-at-the-boundary and
  guaranteed-correct, vs. scraped from the model's own errors. Whether that
  difference matters is exactly what H1 tests.

## One-paragraph summary

Models can't learn to abstain from data where abstention never appears, and
you can't hand-author those examples at scale because you don't know what
the model doesn't know. Make the training knowledge explicit (a
snapshotted KB), and "unknown" becomes generable: correct by construction,
exactly at the boundary, in any ratio, with no surface shortcuts to cheat
with. Then test, with a built-in kill condition, whether the model learns a
real knowledge-boundary instead of a refusal style. **The response-type mix
of the training data is the cheapest unused lever on hallucination — and it
becomes fully controllable the moment the knowledge boundary is explicit.**