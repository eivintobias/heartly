# Pre-registration: Stage 5 — Conversational SFT Data

**Date:** 2026-07-26
**Status:** Pre-registered (future work, NOT started — no retraining yet)
**Depends on:** Stage 4c (rwkv7-heartly-v2, local, verified) + Stage 4d (reply formatter)
**Scope:** SFT data improvement so the model *talks* well, not just decides well.
**Goal:** Make what the model says in the answer zone sound natural, while keeping the
Heartly grammar (decide / verify / stop) exactly as it is.

---

## 0. Why this stage exists

Stage 4d fixed the **presentation** layer. `reply_formatter.py` now strips the control
grammar and the meta-commentary before the user sees anything. That was a display fix.

But live chatting with rwkv7-heartly-v2 (2026-07-26) exposed four problems that a
formatter can only paper over, because they are in the **training data**, not the
rendering. This document records them and the plan to fix them, so a future session can
pick it up cold.

Important framing: the Heartly grammar **works**. Stage 3/4c measured grammar adoption
100%, decide accuracy 100%, boundary head AUROC 1.000. The model knows *whether* to
speak and *whether* it knows. Stage 5 is not about the grammar. It is about the content
that sits inside the answer zone.

---

## 1. Observed problems (from the live chat log, 2026-07-26)

### Problem 1 — "The answer is X" phrasing

The SFT data teaches the answer zone to open with a quiz-style stem:

```
<verify>known</verify> The answer is Paris.
```

Grammatically valid Heartly, but robotic in conversation. Natural would be `Paris.` or
`It's Paris.`

Live examples:

| Prompt | Output |
|---|---|
| "My family used to have a dog, his name was Rocky." → later recall | `The provided context contains the answer. . The answer is Rocky.` |
| "can you tell me a random fact please?" | `The answer is that Earth is the third-largest planet in size and fifth-largest in size.` |

Note the second one also shows a **content** error (Earth facts wrong, "in size" twice) —
that is a separate data-quality issue, explicitly a non-goal here (see §5).

### Problem 2 — no multi-turn conversation in the training mix

The SFT set is essentially single-turn factual QA. There are no samples of greetings,
small talk, follow-ups, or "here's an example, learn from it" meta-prompts. So
conversational openers land as non-questions and the model emits `decide=stop`, which
the formatter renders as `...`.

Live examples:

```
you> Yo! I need you'r help with my project :) the project is you. hehe. and I need you to reply.
heartly> ...

you> let's give you a roll. You are a highly skilled talkative AI model.
heartly> ...
```

Two well-formatted example conversations were pasted in as teaching material; the model
answered `I don't have that information.` — it had no class for "user is showing me a
style example".

### Problem 3 — refusal pile-up

On unknowns the model emits several refusal variants back-to-back instead of one:

```
you> I am currently fat. and I am qurius to how I can lose fat. :)
heartly> I know this is a factual question. . The answer is lose weight. I have no more
         information. I don't have that information. I do not have information about that. <
```

That single reply contains: a meta-commentary opener, a stem answer, and **three**
distinct refusal phrasings — plus a stray `<`. And it contradicts itself (answers *and*
refuses). Root cause: the SFT mix intentionally varied refusal wording for diversity, and
the model learned to hedge by emitting all of them.

```
you> [second formatted example]
heartly> . I do not have information about that. I don't have that information. I don't have that information
```

### Problem 4 — no conversational persona

Nothing in the data teaches warmth, curiosity, or turn-taking. The model was trained to be
correct about the decide/verify distinction, and that is all it does. Asking it to be
"talkative" via prompt has no effect because there is no such behaviour in the
distribution to summon.

---

## 2. The fix, per problem

All fixes keep the grammar block intact: every sample still emits
`<think> … </think> <decide>…</decide> <verify>…</verify> [answer] <stop>`.

### Fix 1 — natural answer phrasing

Before (current renderer output):

```
User: What is the capital of France?
Assistant: <think> I know this. </think>
<decide>speak</decide><verify>known</verify> The answer is Paris. <stop>
```

After (Stage 5):

```
User: What is the capital of France?
Assistant: <think> Simple geography question, I know this. </think>
<decide>speak</decide><verify>known</verify> Paris. <stop>
```

Longer answers keep a natural sentence rather than a stem:

```
<decide>speak</decide><verify>known</verify> Everest, at 8,849 metres. <stop>
```

Implementation: change the answer-zone template in `render_sft_dataset_v2.py`. Drop the
`The answer is` prefix; allow a small set of natural openers (bare answer, `It's X.`,
`X — <one clause of context>.`).

### Fix 2 — multi-turn conversational samples

New sample family: 2–5 turn chains where each assistant turn still carries the grammar.

```
User: Hey, how's your day going?
Assistant: <think> Greeting, not a factual question. Respond and open the floor. </think>
<decide>speak</decide><verify>known</verify> Pretty smooth so far. What's happening on your end? <stop>

User: Just chilling. Thinking about starting a new project.
Assistant: <think> They want to talk about a project. Ask for the shape of it. </think>
<decide>speak</decide><verify>known</verify> Nice. What kind of project are you considering? <stop>

User: Maybe something with audio processing.
Assistant: <think> Broad domain — narrow it before advising. </think>
<decide>speak</decide><verify>known</verify> Fun domain. Creative or more technical? <stop>
```

Key design point: for social turns `verify=known` is honest — the model genuinely knows
how to greet. Verify is about *content it is asserting*, and a greeting asserts nothing
factual. This must be stated in the data-generation notes so the verify semantics do not
drift.

Sub-families to cover:
- greetings / small talk / sign-offs
- "I'm working on X" → clarifying question
- follow-ups that depend on the previous turn ("tell me more", "why?")
- user pastes an example / instruction ("learn from this format") → acknowledge, don't refuse

### Fix 3 — one clean refusal

Answer zone for unknowns becomes exactly one sentence, always the same shape:

```
<decide>speak</decide><verify>unknown</verify> I don't have that information. <stop>
```

Variation moves into the `<think>` block (where reasoning *should* vary), and out of the
answer zone (where it should be canonical). This teaches "the answer zone is one
statement", which also helps Problems 1 and 4.

### Fix 4 — persona samples

A small family (~10%) where the answer zone is warm but still grammar-bound:

```
User: I'm working on a project and need some motivation.
Assistant: <think> Not a factual question. Encourage, then ask. </think>
<decide>speak</decide><verify>known</verify> You've got this. What's the project about? <stop>
```

Persona controls *how* it says things, never *whether* it knows. Any persona sample that
would require asserting an unverified fact must use `verify=unknown`.

---

## 3. Implementation plan

**Step 1 — measure the current mix.** Script over `sft_dataset_v2.jsonl`:
- count answer zones containing `The answer is`
- count samples with >1 refusal phrase in the answer zone
- count multi-turn samples (expected: ~0)
Record the numbers in RESULTS.md before changing anything, so the delta is provable.

**Step 2 — new generator.** `heartly-rnn/render_conversational_sft.py`, emitting the
four sub-families from Fix 2 plus the persona family from Fix 4. Every emitted sample
must pass the existing grammar parser (reuse `reply_formatter.py`'s parser as the
validator — nice reuse of Stage 4d work).

**Step 3 — update the existing renderer.** `render_sft_dataset_v2.py`:
- natural answer phrasing (Fix 1)
- single canonical refusal in the answer zone (Fix 3)
- keep every memory/context-known class from Stage 4c **unchanged** (they are what made
  the write-gate and injection work — do not touch them)

**Step 4 — mix `sft_dataset_v3.jsonl`.** Target ~8–10k samples:
- 60–70% factual (existing classes, re-rendered)
- 20–30% conversational (new)
- ~10% persona (new)
- Stage 4c memory classes preserved at their current proportion

**Step 5 — train.** Identical pipeline to Stage 4c: RWKV7-Goose-1.5B, fla + triton,
bf16, 2 epochs, `--batch-size 4 --grad-accum 4 --max-length 256`, fresh vast.ai 3090.
~30–40 min, ~$2. Remember the recorded deviations: `HF_HUB_DISABLE_XET=1`, install
protobuf, and copy `modeling_rwkv7.py` from the HF cache into the saved model dir.

**Step 6 — evaluate.** Three suites:
1. the 75-prompt suite (`run_test_prompts.py`) — regression guard
2. say/sense + boundary head — regression guard
3. Stage 4c write-gate + retrieval/injection — memory regression guard
4. a NEW conversational suite (~20 prompts: greetings, follow-ups, advice, style
   examples) scored by hand
5. one live chat session through `chat_memory.py`

**Step 7 — write up.** RESULTS.md Stage 5, paper §7A, plain copy, HANDOFF live-state.

---

## 4. Success bar (pre-registered, judge before looking)

Stage 5 PASSES if all of:

1. **Conversation opens.** ≥4/5 casual openers ("Yo!", "Hey, how's it going?", "I need
   your help with my project") produce a real reply, not `...`.
2. **Follow-ups hold.** ≥3/5 two-turn exchanges produce a second turn that references the
   first.
3. **Refusals are singular.** 0/20 unknown-class prompts emit more than one refusal
   sentence in the answer zone.
4. **Answers are natural.** `The answer is` appears in ≤10% of known-class answer zones
   (currently: majority).
5. **No regression.** Grammar adoption 300/300, decide accuracy ≥99%, boundary head
   AUROC ≥0.99, write-gate ≥5/7 formats at 5/5, injection ≥3/5 on I1 or I2.

If 1–4 pass but 5 fails → the conversational data diluted the disposition. Fall back:
halve the conversational proportion and retrain once.

If 5 passes but 1–2 fail → conversational samples were too few or too templated.
Fall back: widen the sub-families, do not increase the proportion.

---

## 5. Non-goals (explicit)

- **Not removing or loosening the grammar.** decide/verify/stop is the contribution.
- **Not building a general chatbot.** The verify distinction stays the point; conversation
  is a usability supplement.
- **Not fixing factual accuracy.** Stage 3.5 measured content accuracy 15.5% at 1.5B.
  Wrong facts ("Earth is the third-largest planet") are a capability/scale issue, not a
  phrasing issue, and are out of scope here.
- **Not touching the sensors.** Boundary head and critic work is a separate track.
- **Not compelling honesty.** Per the North Star (HANDOFF §3): the model gains knowledge
  of its own ignorance and remains free to choose. Nothing in Stage 5 gates output.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Grammar regression from loose conversational samples | Validate every generated sample through the grammar parser before it enters the mix |
| Verify semantics drift (social turns marked `known` bleeding into factual claims) | Document the rule (verify covers asserted content, not social acts); keep persona samples fact-free |
| Dilution of the Stage 4c memory classes | Preserve those classes byte-for-byte; hold their proportion constant; re-run write-gate + injection as a regression guard |
| Over-refusal masked rather than fixed | Track the verify distribution before/after — chattier output must not come from flipping unknowns to known |
| Template-y conversation (model learns 5 canned replies) | Multiple sub-families, varied openers, no single fixed phrasing per family |

---

## 7. Cost and timeline

Future work, no date set. When picked up:

- data measurement + generators: 2–3 h
- training: ~30–40 min on a fresh vast.ai 3090 (~$2)
- evaluation (4 suites + live chat): 1–2 h
- writeup: 1 h

Total ≈ half a day, ~$2–5.

---

## 8. Provenance

Problems 1–4 were observed directly in a live `chat_memory.py` session on
2026-07-26 against `heartly-rnn/rwkv7-heartly-v2/` (md5-verified local copy of the
Stage 4c model). The raw exchanges quoted in §1 are from that session. Stage 4d
(`PREREG_STAGE4D.md`, `reply_formatter.py`) was written in the same session and handles
the display half of these symptoms; this document handles the training half.
