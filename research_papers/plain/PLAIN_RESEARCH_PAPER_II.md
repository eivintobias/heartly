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

Plus §7A at the end: the actual experiment results (updated 2026-07-25).

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

## Part 4 — What the experiments actually showed (§7A, to 2026-07-25)

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

**So the order was:** bigger answerer first (RWKV7-1.5B — more right answers
to learn from), then a bigger checker on the data we already have (free),
then memory. Here is how that went.

**Stage 2.6 — a smarter checker ranks better, but still can't draw the
line.** We put two bigger checkers (Qwen 1.5B and 3B) on the OLD data from
the small answerer. The bigger the checker, the better the sorting: 0.758 →
0.826 → 0.845 — a clean dose-response. But the line-drawing still failed,
because the old data still has only 60 right answers. New lesson: a checker
that is TOO much smarter starts distrusting even genuine right answers.
So: the checker must be stronger than the answerer — but the answerer must
be strong enough that its right answers look normal to the checker. Both
ends of that scale fail.

**Stage 3 — the bigger answerer is here, and it is at ceiling.** RWKV7-1.5B
fine-tuned on the same 6,031 samples. Fifteen minutes and ~$2 this time
(new fast software). Grammar: 100%. Speak/abstain decisions: 100% (was
97.7%). The boundary sensor reads known-from-unknown **perfectly at every
layer we checked**. The model is published on Hugging Face
(eivintobias/heartly-rwkv7-1.5b).

**Stage 3.5 — the real test: sorting works, line-drawing doesn't, and one
of our own predictions died.** We reran the whole checker pipeline on the
1.5B:

- Grammar held: all 2,902 answers parseable.
- Decisions: 99.8% right. ZERO cases of speaking on a question it should
  have refused.
- The 5 old blind-spot questions? All refusals now. That blind spot is
  closed — on the decision side.
- Content: when it chose to answer, it was right **15.5%** of the time
  (was 4.1%). Better — still low. Knowing-THAT-you-know scales fast;
  knowing-things scales slowly.
- The checkers: same-model checker 0.835, 3B checker 0.824, 1.5B checker
  0.750. Same patterns as before, replicated at the new scale.

**And the headline: more right answers did NOT fix the threshold.** We had
predicted they would. Right answers went from 60 to 229 — and the alarm
still can't be tuned to catch lies without also flagging most truths (it
false-alarms on 57–87% of genuine right answers). The problem is not how
MANY right answers there are. It's that the answerer's own right answers
don't *look* right to the checker. One more twist: the 1.5B's lies are
subtler than the small model's, so the same checkers score WORSE on the new
data. The better the answerer gets, the harder the checker's job.

**What works, and what's next.** What works everywhere: the sorting. The
worst-scoring 10% of answers are 100% lies — "show a human the bottom of
the pile" is already useful. What works nowhere so far: the automatic
yes/no alarm. Three doors remain: (a) train the checker specifically on
THIS answerer's right-vs-wrong style, (b) make the checker actually verify
facts (look things up, or ask the model twice and compare) instead of just
sniffing for statistical lie-smell, (c) ship the sorting as the product.
Then memory (Stage 4): teaching the model to wake up with yesterday's gist.

**Stage 3.6 — door (a) tried and closed.** We trained six different kinds
of checkers on the saved answers — simple ones, small neural nets, tree
ensembles, a calibrated one — and we were careful not to peek at the test
data while choosing settings (all choices were made on the training part;
the test set was touched once per checker). EVERY one of the six failed
the alarm test, in three different ways: most flagged way too many right
answers; the best sorter (a tree ensemble, 0.854 — the best sorting score
yet) was so extreme in its scores that it ended up flagging literally
everything; and the fanciest one got lost in the data and sorted WORSE.
The lesson: it's not about HOW you fit the checker — the information
needed for a clean yes/no line just isn't in the signal. After nine
checker variants across three stages, the automatic yes/no alarm is
officially dead. What survives is the sorting: "show a human the bottom
of the pile" works under every checker (the bottom 10% is 93–100% lies).
Door (b) — a checker that actually verifies facts against the world —
stays open for later. But the next big thing is memory: Stage 4, teaching
the model to wake up with yesterday's gist.

**Stage 4 — memory: the brain file works perfectly; the facts don't ride
along.** We saved the model's ENTIRE inner state after a short chat (a
12.8MB file), loaded it into a completely fresh session, and quizzed it.
The save-and-reload machinery is PERFECT: after loading, the model's
next-word guesses are identical to the original, down to three decimal
places. It truly wakes up as the same conversational partner. But when we
ask "what's my dog's name?" — a fact we taught it moments earlier — it
says "I don't know". Zero out of 5 facts recalled. And here's the twist
that proves the file is innocent: the LIVE model (no saving, no loading,
same session) also says "I don't know", 0 out of 5. The forgetfulness is
not in the file — it's in how the model reads its own state. Two more
clues: forcing the answer slot open sometimes pulls the fact out
("mango Tuesday" came back word for word!), and teaching facts as
question-answer pairs works a little better than declarations. The deep
reason: we trained Heartly to REFUSE personal questions ("what's my
voicemail PIN" is unknowable by design) — and now that training refuses
to read its own memory. The verify sense can't see the state's content
as "knowledge". Which is exactly what the paper's design said to do
anyway: facts belong in a searchable memory bank that gets injected into
the chat when relevant — not in the raw state file. The state file's
real job is session continuity, and that part works exactly as hoped.

**Stage 4b — two doors tested: the state door opens with the right key;
the memory-bank door is stuck for a fixable reason.** Two experiments,
both planned and locked in before running:

*Door A — writing facts into the state differently.* We tried 6 ways of
teaching the same 5 facts. Plain statements: 0/5 recalled (same as
before). Question-answer pairs: 1/5. Having the assistant say the fact
itself, out loud, as "known": 1/5. Repeating the Q&A three times: 2/5.
And then the big one — **writing each fact THREE ways in the same chat
(assistant says it + trivia-style + Q&A): 4 out of 5 recalled!** That
passes our bar. The pattern is clear: the more ways and the more times
the model sees itself treat a fact as known, the more it counts as
known. Like rehearsing something until it sticks.

*Door B — the memory bank (search-and-inject).* The search part worked
perfectly: given 20 stored memories (5 real + 15 decoys), the right
memory was found every single time (5/5). But then the weird part:
we put the fact DIRECTLY IN THE PROMPT — "Context: the user's dog is
named Zorblax. What is my dog's name?" — and the model STILL said
"I don't have that information" most of the time (best format: 2/5).
The fact is right there and it won't read it!

Why? Because we never taught it to. Our training data has questions
and answers — but zero examples of "here's some context, answer FROM
it". So injected text is just scenery to the model, not knowledge.
That's actually good news: it's not a design flaw, it's a missing
lesson. Stage 4c is now obvious: add "answer from the provided
context" examples (and some multi-turn conversations — we also noticed
the grammar gets wobbly in long chats, because all training was
single-turn) to the training mix and retrain. Prediction: both doors
open after that.

One fun reversal: we expected the memory bank to win and the state to
lose. The opposite happened. The lesson of the whole memory track so
far: **the model can only use a memory channel it was trained to
read.** Which fits the project's oldest theme perfectly — everything
is a data-distribution decision.

**Stage 4c — we added the missing lesson, and BOTH doors flew open.**
The plan from Stage 4b, executed: we added ~1,000 new training
examples of "here's a paragraph of context, answer FROM it" (built
from SQuAD) and ~500 multi-turn conversations, then retrained the
model with the exact same recipe (about 34 minutes, ~$1). Then we
re-ran the exact same tests that failed the day before.

Three results, all good:

*First — nothing broke.* The core honesty behavior is identical:
grammar perfect (300/300), speak/abstain decisions perfect, the
boundary sensor still reads AUROC 1.0. Adding the new lessons cost
nothing.

*Second — the state door didn't just open, it fell off its hinges.*
Every single one of the 6 write formats now recalls **5 out of 5
facts** — including plain statements, which scored 0/5 twice before.
You no longer need the "say it three ways" rehearsal trick. Tell the
model a fact once, plainly, and it counts as known.

*Third — the memory bank works end to end.* Search still perfect
(5/5), and now when the found memory is put in the prompt, the model
answers from it: 4/5 in the two formats closest to how we trained it.
The full pipeline — store memories, search them, inject the right
one, get a grounded answer — works.

The surprise bonus: the context lesson was aimed at the memory bank
(Door B), but it fixed Door A too. That tells us the two "problems"
were really ONE missing idea in the model's head: "text I can see
right now counts as knowledge." One thousand examples taught it that
idea, and both doors opened.

And the best part for the project's philosophy: the honesty survived.
The one fact the model keeps missing (the favorite NUMBER — numbers
are hard to copy) it now says "I don't know" about, instead of making
something up. We taught it "context is knowledge" without accidentally
teaching it "everything is knowledge." The boundary moved exactly
where we pointed it, and nowhere else.

## The one-line summary of the whole program

Honesty isn't a rule you bolt onto a model — it's built from three things:
**data** that shows both speaking and abstaining at the true boundary,
**sensors** that let the model register what it lacks, and **memory** that
lets it grow without being corrupted. Sensors inform — they never force.
The model that knows its own ignorance is, for the first time, free to
choose.