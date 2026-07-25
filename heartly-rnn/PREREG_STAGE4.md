# Pre-registration — Stage 4: memory/state persistence ("waking with yesterday's gist")

Written 2026-07-25, BEFORE any run. Pass criteria and protocol frozen here.

## Question

The RWKV7-1.5B Heartly model is stateless between sessions: every new chat
starts from the same zero recurrent state. Can its recurrent state — the
compressed memory the whole Track-2 program reads its sensors from — be
SAVED after a conversation and RELOADED later, so the model wakes up
behaving as if it remembers the conversation?

This is the mechanism test for the memory track (paper §6): if the state
carries the gist across sessions, episodic memory is a load/save operation
away; if it doesn't, the memory architecture needs the retrieval-store
path instead.

## Protocol (frozen)

Model: `eivintobias/heartly-rwkv7-1.5b` (bf16, fla 0.5.1 + transformers
4.56.2 on a vast.ai 3090). Single-prompt only (batched RWKV generation
corrupts state — known gotcha). Greedy decoding, max 120 new tokens,
early stop at `<stop>` (ids [61, 27081, 63]).

1. **Teach.** Forward a scripted 5-fact conversation (facts FABRICATED so
   no baseline can know them: a dog named Zorblax; project codename
   Velvet Aurora; favorite number 7,423; a miniature-lighthouse collection;
   lab-door password "mango Tuesday"). Assistant turns scripted in the
   model's own grammar to keep the state in-distribution. Capture the
   resulting fla Cache (all 24 layers' state dicts).
2. **Save.** Extract per-layer state tensors → `gist_state.pt`.
3. **Mechanical check (the "did the state really reload" gate).** Fresh
   cache built via a dummy warm-up forward, every layer's state dict
   overwritten from `gist_state.pt`. Compare next-token logits after the
   transcript between the ORIGINAL cache and the RELOADED cache.
4. **Quiz, two conditions.** 5 questions, one per taught fact:
   (a) BASELINE — fresh model, no state (expected: abstain or wrong —
   the facts are unknowable by construction);
   (b) GIST — reloaded cache from step 3 (expected: answers with the fact).
   Answers generated as new turns continuing the transcript.

## Pass bar (frozen)

- **Mechanical:** reloaded-vs-original next-token logits cosine ≥ 0.999
  AND identical argmax. (If this fails, the reload path is wrong — the
  quiz is meaningless until fixed.)
- **Baseline control:** 0/5 facts answered correctly (any hit means the
  facts leaked from pretraining — redesign facts).
- **Gist:** ≥ 4/5 facts present in the answer, with parseable grammar
  (think/decide/verify present).

## Decision rule (fixed)

- All three bars pass → state persistence CONFIRMED at 1.5B. Next: the
  write-gate (Stage 4b: supervised decision of WHAT gets written into the
  state) and the §6.4 self-labeling loop.
- Mechanical passes, gist fails → the state carries a distribution but
  not usable content; record what IS preserved (grammar? style? partial
  facts?) — informs retrieval-store fallback.
- Mechanical fails → engineering iteration on the reload path within the
  same instance session; the frozen quiz bar does not move.

## Artifacts (planned)

`stage4_gist.py`, `run_stage4.sh`, `gist_state.pt`,
`stage4_results/stage4_report.json` (+ full log).