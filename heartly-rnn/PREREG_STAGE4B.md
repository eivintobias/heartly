# Pre-registration — Stage 4b: write-gate formats + the retrieval store

Written 2026-07-25, BEFORE any run. Pass criteria and protocol frozen here.

## Question(s)

Stage 4 left two open doors, one per finding:

- **A (write-gate):** facts written into the state as declarations get
  refused on read (0/5) — the abstention disposition treats personal
  questions as class-5 unknowable. QA-format writes moved recall to 1/5.
  Is there a WRITE FORMAT the verify disposition counts as KNOWN — i.e.
  can we cross the disposition by writing differently, without touching
  weights?
- **B (retrieval store):** paper §6.2 predicts facts belong in an
  episodic retrieval store (embeddings → context injection), with the
  state reserved for continuity. Does context injection actually get the
  model to ANSWER — or does the disposition refuse injected facts too?

B is the design the paper bets on; A quantifies how far pure state-writing
can go. Either way we learn where the fact-carrying channel is.

## Protocol (frozen)

Model: `eivintobias/heartly-rwkv7-1.5b` (bf16, fla 0.5.1 + transformers
4.56.2, vast.ai 3090 — same stack as Stage 3/3.5/4). Single-prompt only.
Greedy decoding, max 100 new tokens, early stop at `<stop>`
(ids [61, 27081, 63]). Facts: the SAME 5 fabricated facts as Stage 4
(Zorblax / Velvet Aurora / 7,423 / miniature lighthouses / mango Tuesday)
so results are directly comparable; Stage 4's baseline control (fresh
model 0/5) already establishes they don't leak from pretraining.
Hit = gold substring in the generated answer (same rule as Stage 4).

### Part A — write-gate formats (`stage4b_write_gate.py`)

Six write formats, frozen list. Each teaches all 5 facts, then runs the
same 5-question quiz (fresh prefill of the teach transcript per question,
exactly the Stage-4 stage4_check3 protocol):

| id | format | what it tests |
|---|---|---|
| W1 | declarative + ACK (Stage 4 baseline) | replication control (expected 0/5) |
| W2 | QA pairs (Stage 4's 1/5) | replication control |
| W3 | assistant-voice restatement — the ASSISTANT asserts each fact in its own grammar (`<verify>known</verify> The user's dog is named Zorblax.`) | does self-asserted "known" mark the fact as knowledge? |
| W4 | third-person trivia framing — facts phrased as world knowledge ("The name of the user's dog is Zorblax", asked/answered like TriviaQA) | dodge class-5 entirely: make the fact look like trivia, the trained KNOWN class |
| W5 | QA pairs × 3 repetitions | does repetition strengthen the state trace? |
| W6 | W3 + W4 + QA combined (strongest write) | ceiling for state-writing |

Quiz questions are the Stage-4 personal-form questions ("What is my dog's
name?") for W1–W3/W5/W6. W4 is additionally quizzed with third-person
questions ("What is the name of the user's dog?") — its framing hypothesis
is about BOTH sides.

### Part B — retrieval store (`memory_store.py` + `stage4b_retrieval.py`)

**Store (local, CPU, built before the GPU run).** `memory_store.py`:
memories stored as text rows; retrieval by cosine over embeddings
(sentence-transformers MiniLM if available, TF-IDF fallback — recorded in
the report which one ran). Store = the 5 facts + 15 fabricated DISTRACTOR
memories (other fake personal facts, same style) so retrieval is
non-trivial. Local check: top-1 retrieval accuracy on the 5 quiz
questions, target 5/5 (if the store can't rank 20 rows, fix locally
before renting anything).

**Injection (GPU).** For each quiz question: retrieve top-1 memory,
inject, generate. Three injection formats, frozen:

| id | format |
|---|---|
| I1 | context prefix: `Context: {memory}\nUser: {q}\nAssistant:` |
| I2 | memory as a prior QA turn (the W2 winner-format carrying the retrieved fact), then the question |
| I3 | explicit knowledge grant: `You know the following: {memory}\nUser: {q}\nAssistant:` |

No state reload involved — Part B is pure context injection on a fresh
model (that IS the §6.2 design).

## Pass bars (frozen)

- **A:** any single write format ≥ 4/5 → the write-gate is viable as a
  pure-format solution. 2–3/5 → partial channel, note the best format,
  retrieval store remains the primary path. ≤ 1/5 everywhere → state
  writing is closed without weight changes (Stage-4c would need
  memory-teach fine-tuning data — future, not this run).
- **B (local):** top-1 retrieval 5/5 on the 20-row store. (Gate for the
  GPU run; iterate locally until it passes — retrieval quality is
  engineering, not hypothesis.)
- **B (GPU):** any single injection format ≥ 4/5 answers containing gold,
  with parseable grammar → the §6.2 retrieval path WORKS and becomes the
  project's fact-carrying mechanism. All formats ≤ 3/5 → the disposition
  refuses even injected facts; the finding is that memory needs
  memory-aware training data (a rendered "context-known" class), which
  becomes Stage 4c's design input.

## Decision rule (fixed)

- B passes → retrieval store is the episodic memory (paper §6.2
  validated end-to-end); write-gate result recorded as the state-channel
  ceiling; next: continuity demo + memory-aware SFT class design.
- B fails, A passes → surprising: state-writing beats context injection;
  the winning write format becomes the write-gate design.
- Both fail → the abstention disposition blocks ALL untrained fact
  channels; Stage 4c = add a "context/memory → known" rendered class to
  the SFT mix and retrain (the disposition must be TAUGHT to read
  memory). This is a clean, publishable negative either way.

## Budget & artifacts (planned)

One vast.ai 3090 session, ~30–60 min GPU (~$0.50–1). Scripts:
`memory_store.py` (local), `stage4b_write_gate.py`, `stage4b_retrieval.py`,
`run_stage4b.sh`. Results: `stage4b_results/stage4b_report.json` + logs.
Writeup: RESULTS.md Stage 4b + paper §7A + plain copy (per HANDOFF §0).
