# Pre-registration: Stage 4d — Reply Formatting & Presentation

**Date:** 2026-07-26
**Status:** Pre-registered (not yet implemented)
**Depends on:** Stage 4c (rwkv7-heartly-v2, local, verified)
**Scope:** Post-processing layer only — NO model retraining. The model's internal
grammar (decide/verify/stop) stays as-is. This stage fixes how the chat scripts
*present* the model's output to the user.

---

## The Problem

When you chat with Heartly v2, the replies look broken. The model's internal
control grammar leaks into what the user sees. There are two distinct problems:

### Problem 1: Control grammar leaks into the displayed reply

The model outputs its reasoning and control tags as part of the response. What
the user sees looks like this:

```
The answer is Paris.  I know this fact. I can respond confidently.  <decide>speak</decide><verify>known</verify> The answer is Paris. <stop>done</stop>
```

Instead of just:

```
Paris
```

The existing `clean_reply()` function in `chat_memory.py` tries to strip these
tags, but it's fighting a losing battle because:

- The model puts the *actual answer* inside the think block sometimes
  ("The answer is Paris" lives inside the think, then gets repeated after
  the verify tag). The current code unwraps think blocks instead of
  extracting from them, so both copies survive.
- The model loops: it says the answer, then says `<stop>done</stop>`, then
  says the answer again, then loops on stop tokens. The dedup catches some
  of this but not all.
- The spurious `<tool_call>` token (vocab leak noted in HANDOFF) opens many outputs
  and isn't stripped.
- Control words like "speak", "known", "unknown", "silent" survive as
  stray words in the output.

**Root cause:** The model was trained on a grammar where the answer appears
in *multiple places* (think block, after verify tag, sometimes a third time
in a loop). The current cleaner tries to delete the scaffolding while keeping
the content, but the content is *embedded inside* the scaffolding, so you
can't cleanly separate them with regex.

### Problem 2: The answer is buried inside the grammar, not after it

The Heartly grammar is:

```
<think> [reasoning] </think> <decide>speak|stop</decide> <verify>known|unknown</verify> [answer] <stop>done</stop>
```

The *intended* structure is: think first, then decide, then verify, then give
the answer. But the model often puts the answer *inside the think block* and
then repeats it (or a degraded version) after the verify tag. Examples from
the test results:

- "The answer is Paris.  I know this fact..." then
  `<decide>speak</decide><verify>known</verify> The answer is Paris.`
- "I know this fact. I can answer confidently. I will speak." then
  `<decide>speak</decide><verify>known</verify> The answer is 1945."

So the *real* answer is in the think block, and the post-verify text is
either a repeat or a worse version. The current cleaner unwraps the think
block (good instinct) but then both copies survive and the user sees the
answer twice plus all the reasoning text.

**Root cause:** The SFT data taught the model to "show its work" in the
think block (including the answer) and then "give the answer" after verify.
For a chat interface, the user should only see the final answer — the
reasoning is internal machinery.

---

## The Fix

### Fix for Problem 1: Smarter reply extraction (not just tag stripping)

Instead of "strip the bad parts, keep everything else," flip the logic:
**extract the good part, discard everything else.**

The strategy:

1. **Parse the grammar.** The model's output follows a predictable structure:
   think block, decide tag, verify tag, answer text, stop tag. We can parse
   this with a simple state machine instead of regex whack-a-mole.

2. **Extract the answer from the right place.** After `<verify>known</verify>`
   or `<verify>unknown</verify>`, everything up to `<stop>` is the answer.
   That's the clean answer zone. If the verify tag is present, use only the
   text after it.

3. **Fallback: if no verify tag found**, the think block often contains the
   answer. Look for "The answer is X" patterns inside the think block and
   extract just X.

4. **Handle the `decide>stop` case.** If the model decides to stop (no
   question worth answering), show nothing or a brief "I don't have anything
   to add to that."

5. **Handle the `verify>unknown` case.** The answer zone after unknown
   typically says "I don't know" or similar — that's the correct output,
   keep it as-is.

6. **Strip the spurious `<tool_call>` opener** before any other processing.

### Fix for Problem 2: Present the answer, not the reasoning

The think block is the model's internal monologue. The user should never see
it. The decide/verify tags are control signals. The user should never see
them either. The only thing the user should see is:

- For `decide=speak, verify=known`: the answer text after the verify tag
- For `decide=speak, verify=unknown`: the "I don't know" text after the verify tag
- For `decide=stop`: nothing (or a configurable fallback like "...")
- For `decide=silent`: nothing

This is a *presentation* decision, not a model change. The model keeps
generating the full grammar internally — that's valuable for the boundary
head and for debugging. But the chat interface hides it.

---

## Implementation Plan

### Step 1: New `reply_formatter.py` module

Create `heartly-rnn/reply_formatter.py` with a `format_reply()` function
that replaces `clean_reply()`. It will:

```python
def format_reply(raw: str, mode: str = "chat") -> str:
    """
    Extract the user-visible answer from a Heartly grammar output.

    Modes:
      "chat"    — only the answer, no grammar visible (default)
      "debug"   — answer + verify/decide status as metadata
      "raw"     — the full raw output unchanged

    Returns the formatted string ready to show the user.
    """
```

The parser logic:

1. Strip leading `<tool_call>` token (the known vocab leak).
2. Try to find `<verify>...</verify>` in the output.
3. If found: take everything after the closing `</verify>` up to `<stop>`.
   Clean up whitespace. That's the answer.
4. If not found: look for "The answer is X" or "The answer is: X" in the
   think block. Extract X.
5. If neither: fall back to the old `clean_reply()` behavior (better than
   nothing).
6. Deduplicate repeated sentences (keep the existing logic for safety).
7. For `decide=stop` or `decide=silent`: return empty string or "...".
8. For `verify=unknown`: the post-verify text is the "I don't know" — keep it.

### Step 2: Update `chat_memory.py`

Replace the `clean_reply()` call with `format_reply()` from the new module.
Add a `--reply-mode` flag (chat/debug/raw) so you can see the grammar when
debugging.

### Step 3: Update `chat_v2.py`

Same change — use `format_reply()` instead of showing raw output. Add the
same `--reply-mode` flag.

### Step 4: Test against the 75-prompt suite

Run `run_test_prompts.py` with the new formatter and verify:
- Known answers show just the answer (no grammar, no reasoning)
- Unknown answers show just "I don't know" (no grammar)
- Stop/silent cases show nothing or "..."
- No answer is *lost* (the formatter must never drop the actual content)

### Step 5: Update HANDOFF.md

Add Stage 4d to the next-actions list and mark it complete when done.

---

## Success Bar

The stage passes when:

1. **All 75 test prompts produce clean output** — no control tags, no
   reasoning text, no `<tool_call>`, no stray "speak/known/unknown" words visible
   in chat mode.
2. **No answer is lost** — every answer that the model generates correctly
   must still appear in the formatted output. The formatter must be
   conservative: when in doubt, show more rather than less.
3. **Debug mode works** — `--reply-mode debug` shows the answer plus the
   decide/verify status, useful for development.
4. **Both chat scripts updated** — `chat_memory.py` and `chat_v2.py` use
   the new formatter.

---

## Non-goals

- **We are NOT retraining the model.** The grammar leak is a presentation
  problem, not a model problem. The model *should* generate the full grammar
  (that's how the boundary head reads it). The chat script just needs to
  present it better.
- **We are NOT changing the SFT data format.** Future data passes can clean
  up the spurious `<tool_call>` opener and the answer-in-think-block pattern, but
  that's a separate stage.
- **We are NOT building a general-purpose output parser.** This is specific
  to the Heartly grammar and the known quirks of rwkv7-heartly-v2.

---

## Risks

- **Over-stripping:** If the parser is too aggressive, it could drop real
  content. The fallback to `clean_reply()` and the "when in doubt, show more"
  principle mitigate this.
- **Grammar variations:** The model doesn't always produce perfectly-formed
  grammar. The parser needs to handle missing tags, extra whitespace, and
  partial outputs gracefully.
- **The `<tool_call>` token:** This is a vocab leak that should be fixed in the
  training data eventually. For now, we strip it in the formatter.
