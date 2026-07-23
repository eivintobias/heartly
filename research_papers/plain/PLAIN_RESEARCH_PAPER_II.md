# The True Boundary (Paper II) — the plain version

*Easy-to-read copy of `../RESEARCH_PAPER_II_TRUE_BOUNDARY.md`, including the
July 2026 experiment results. This is the main working document of the
project: what broke, what we learned, what we're building, and what the
numbers say so far.*

---

## What this paper is about

Heartly v2 shipped and failed in instructive ways. This paper is the
diagnosis and the research program that came out of it. Three big ideas:

1. **The boundary error** — a mistake in how we built "unknown" examples.
2. **The negative side** — two mechanisms that let a model *register
   absence* instead of always generating.
3. **Memory** — how a model could learn from experience without being
   corrupted by it.

Plus §7A at the end: the actual experiment results (updated 2026-07-22).

## Part 1 — The boundary error (the deep lesson)

**The corpus's edge is not the model's edge.** Our training data defines
what *the dataset* covers. But the model starts from a pretrained base that
already knows a lot (Qwen already knows the capital of France). So if we
label a France question as "unknown" just because it's held out of our
dataset, we're teaching the model to *deny knowledge it has*. That's
over-refusal built into the labels — the exact failure we were trying to
kill.

The fix: unknown examples must come from places where the **model** is
genuinely ignorant, not just where our dataset is empty. And they must look
on the surface exactly like known questions (same style, same topics), so
the model can't cheat with shortcuts.

**The six kinds of genuinely-unknown questions we generate:**

1. **Made-up things** — invented people, drugs, books, companies ("What is
   the half-life of Zynthromycin?"). Unknown by construction: they exist in
   no training corpus.
2. **Wrong-type questions** — real things asked about attributes that don't
   apply ("What is the boiling point of Marie Curie?").
3. **After the cutoff** — events dated after the base model's knowledge
   cutoff ("Who won the 2026 World Cup?").
4. **Absurdly specific** — real things, impossibly deep detail ("What is
   the specific heat capacity of glycerol at 256 Kelvin?").
5. **Unanswerable in principle** — personal data, secrets, future
   speculation ("What is my voicemail PIN?").
6. **FEVER not-enough-info** — the one human-labeled boundary class.

**And the inoculation (the other half):** the *known* side must contain
similar-looking questions — obscure-but-real trivia, past dated events,
covered specifics — so shortcuts like "weird name → unknown" or "year →
unknown" get punished during training. Both sides look alike; only the
knowledge differs.

## Part 2 — The negative side (mechanisms that register absence)

A generator can only say things — everything it produces comes from the
"presence" side of its training. But "I don't know" is a report about
*absence*. So we build two mechanisms for absence:

**A. Unlikelihood (in the loss, not yet run).** Normal training only pushes
UP the probability of correct text. We add a push DOWN: harvest the model's
own confident wrong answers, and train it to make those less likely. Plus
side raises honesty, minus side suppresses confabulation, same update.

**B. The boundary head (in the architecture — built and working).** A tiny
probe (~1,500 parameters) that reads the model's internal state at the
`<verify>` position and reports P(known). It never generates text. It gives
us: a calibrated confidence score, and a **say/sense alarm** — if the model
*says* "known" while the head *senses* "unknown", that's a confabulation
caught in one forward pass.

## Part 3 — Memory without corruption

Birth nature (what's trained in) is only half the story. A model that can't
learn from experience can't grow — but a model that learns from its chats
can be poisoned or can drift. The design, in plain words:

- **Episodic store (safe):** chats go into a searchable memory bank that
  gets retrieved into context when relevant. No weight changes — the model
  itself can't be corrupted through this layer. Deletable, inspectable.
- **Consolidation (governed):** every so often, a small LoRA fine-tune from
  *curated* episodes — never raw logs. Snapshot before, health check after,
  and a **veto** that rolls back if anything looks off. Sleep, then a blood
  test in the morning.
- **Trust gates:** new conversation partners start untrusted; their
  episodes quarantine until trust accumulates. Memories are compartmental-
  ized by source — you can delete one relationship's whole compartment.
- **Drift alarm:** watch the model's behavior statistics over time; many
  small nudges that jointly steer it trip the alarm.
- **The loop that makes it compound:** when the model correctly says
  "I don't know" in a real chat, that's a verified boundary event — perfect
  new training data for the next consolidation. The verify mechanism
  curates the memory that improves the verify mechanism.

**Kill conditions:** if consolidation degrades the birth behaviors, the
veto fires; if vetoes fire routinely, consolidation was too aggressive and
we drop back to episodic-only.

## Part 4 — What the experiments actually showed (§7A, to 2026-07-22)

**Experiment 1 — the absence sensor is real.** On 2,902 true-boundary
questions, a tiny probe reading the recurrent state of RNN-type models
(Falcon-H1, RWKV) separated known from unknown **perfectly** (AUROC 1.000),
before any answer was generated. The transformer baseline (Qwen) was nearly
as good (0.998). The model "knows that it doesn't know", and we can read
it. This unlocked Stage 2.

**Stage 2 — the grammar deploys for real.** RWKV-430m fine-tuned on 6,031
Heartly samples (2 epochs, ~$2 on a rented GPU): grammar adoption 100%,
decision accuracy 97.7%, deployed boundary head still perfect (1.000).

**The blind spot (the instructive failure).** Say/sense agreement was 100%
— zero alarms. But 5 confident confabulations happened, and in all 5 the
head agreed with the lie (sense_p ≥ 0.97). Two structural reasons: the head
reads the *same brain* that made the mistake, and it reads at `<verify>` —
*before the answer exists*. It senses "is this question answerable", never
"is this answer right". **Confident confabulation needs a channel that sees
the answer and isn't the generator.**

**Bonus confirmation:** two questions the dataset called "known" got
sense_p = 0.001 — the model genuinely doesn't know them. The corpus label
was wrong *for this model*. The boundary error, caught in the wild.

**The brutal content number.** The model spoke on 1,459 known questions and
was *right* only 60 times — **4.1%** — while its speak/abstain decisions
were 97% right. It learned *whether* to speak; the 0.43B base mostly can't
back it with content.

**Stage 2.5 — the independent critic: principle proven, tool not ready.**
We built two critics that DO read the answer: one on a separate model
(Qwen), one on the same model at the end of its own output. Verdict against
the pre-registered pass bar: **FAIL** — they rank wrong answers below right
ones genuinely (the 5 blind-spot lies score as the lowest items in the
whole dataset), but there's no threshold that catches most lies without
also flagging most truths. Two plain lessons:

- **A checker must be smarter than the thing it checks.** Qwen-0.5B can't
  fact-check trivia it doesn't know either. "Different" isn't enough —
  *stronger* is required.
- **We need an answerer that's right more often.** With only 4% correct
  answers, "right answer" is too rare a thing to learn to recognize.

**So the order is:** bigger answerer first (RWKV7-1.5B — more right answers
to learn from), then a bigger checker on the data we already have (free),
then memory (Stage 3).

## The one-line summary of the whole program

Honesty isn't a rule you bolt onto a model — it's built from three things:
**data** that shows both speaking and abstaining at the true boundary,
**sensors** that let the model register what it lacks, and **memory** that
lets it grow without being corrupted. Sensors inform — they never force.
The model that knows its own ignorance is, for the first time, free to
choose.