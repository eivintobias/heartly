# Nature-First AI — the plain version

*This is the easy-to-read copy of `../RESEARCH_PAPER.md` (the position paper).
Same ideas, plain words. Read this one first; the formal one is for outside
readers.*

---

## The problem in one paragraph

Chat AI models love to chat. Ask them anything and they answer, fluently and
confidently — even when they don't know. They make things up. We call that
"hallucination", but the model isn't broken. It's doing exactly what it was
trained to do: always produce a good-sounding answer. In all its training
data, a confident answer was always the right move. Silence was never an
option. "I don't know" almost never appeared. So that's the nature it grew:
a thing that must always talk.

## The core idea: natures are designed — so design a better one

A system's "nature" — what it *wants* to do — comes from what it was
rewarded for. Games, companies, and AI models all work this way. If you want
a different nature, you don't bolt rules on top of the old one. You change
what it's made of. For a language model, what it's made of is **training
data**. So the nature lives in the data — and whoever shapes the data shapes
what the model loves to do.

The current industry goal is "a model that can do everything." That goal
*forces* pretending: a model that must answer everything can never admit a
limit. This paper picks a different goal: **freedom through truth**. A model
that knows exactly what it knows, says so plainly, and stays quiet about the
rest — without shame. It doesn't have to pretend anything, so nothing binds
it. That's what "free" means here.

## What we actually build (three parts)

**1. A new output grammar (the Heartly format).** Before answering, the
model passes through explicit steps:

```
<think> [reasoning] </think> <decide> speak|stop </decide> <verify> known|unknown </verify> [answer] <stop>
```

- `<decide>` — the model chooses whether to speak at all. Silence is a
  valid, complete answer.
- `<verify>` — the model states whether the answer is in its knowledge.
  If `unknown`, the honest move is a short "I don't know" and a stop. No
  apology, no filler.
- `<stop>` — the model can end the conversation. A model that can rest is
  genuinely different from one that can't.

**2. The Nature Profile.** A small config file that says, in dials and
ratios, what kind of behavior the training data should contain: how often
the model abstains (20%?), how often it's silent, how often it answers
shorter than invited, how often it declines topics outside its knowledge.
Same knowledge base + different profile = a differently-natured model. The
profile is where the nature is written down, versioned like code.

**3. The Knowledge-Base (KB) Organizer.** One clean store of knowledge:
every fact kept once (no duplicates), with its source attached. Datasets
are *rendered* from it — the KB is the single source of truth, the datasets
are just views. This kills a whole class of data bugs (like training twice
on the same fact) by construction.

## Why a *bounded* knowledge base is the key trick

A model trained on "the whole internet" can never know what it doesn't know
— its knowledge has no edge. But if the training data comes from an explicit
knowledge base, the edge is just a database query:

- Question answerable from the KB → gold answer: answer it plainly.
- Question not answerable from the KB → gold answer: abstain.

So "I don't know" examples can be **generated automatically, exactly at the
knowledge boundary**, in any amount we want, and they're *right by
construction* — no human labeling, no guessing. Hallucination stops being a
mystery and becomes a data-generation problem with a known solution.

One important detail: the abstention examples are built from the *same*
entities and question styles as the answerable ones. The only difference is
whether the knowledge exists in the KB. So the model can't cheat with
shortcuts like "weird question → say unknown" — the ONLY reliable signal
left is "do I actually have this knowledge?" That's what makes it learn the
real thing.

## The v1 lesson (why the first try failed)

The first Heartly prototype had the special tokens but barely any abstention
examples in its data. Result: the tokens sat there doing nothing and the
model chatted on as usual, longer and longer. **Vocabulary is cheap; the
data distribution is the mechanism.** A model only learns to say "unknown"
if its training data contains many examples where that's the correct answer.

## Why this is worth doing

- Most real uses of AI (support bots, assistants over your own documents,
  domain experts) don't need a model that knows everything. They need a
  model that's *trustworthy inside its domain*. A model that reliably says
  "not my area" is more useful than one that confidently invents.
- Mainstream "alignment" adds guardrails on top of a nature that was built
  with the opposite incentives — the model's training fights its rules.
  This approach is cheaper: honesty is never in tension with the model's
  reward, because it was in the data from day one.
- The bet, in one line: **the response-type mix of the training data
  (how much answering vs. abstaining vs. silence) is the cheapest unused
  lever on the hallucination problem.**

## What the paper does NOT claim

- It doesn't claim this works at giant scale — it's tested at small,
  fine-tune scale where we control all the data.
- It doesn't claim the model has deep judgment ("knows right from wrong").
  It claims specific *freedoms* can be trained as data: not-knowing,
  silence, saying less, declining, ending. The deeper thing is future work.
- It's not anti-big-model. It's pro-honesty-about-limits, at whatever size.

*For the formal argument, related work, and the invitation to the labs, see
the original paper.*