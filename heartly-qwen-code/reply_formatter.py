#!/usr/bin/env python3
"""
reply_formatter.py -- extract the user-visible answer from a Heartly
Qwen-Code (v3) grammar output.

The v3 model is trained to emit:
    thinking {reasoning}  response<decide>speak|stop</decide><verify>known|unknown</verify> {answer} <stop>

(or `<decide>stop</decide>` for the silence case).

The user should only ever see {answer}, never the scaffolding. In practice the
model -- especially through a GUI like LM Studio -- produces *noisy* variants:
the tags are NOT registered as tokenizer special tokens, so Qwen's BPE shatters
`<decide>` into subwords that can re-decode mangled (e.g. ``<deside>``), the
`<stop>` end-marker can come back truncated (no closing `>`), and verify values
can carry junk (`<verify>known, true</verify>`). Qwen chat-template tokens also
leak as plain text (`<stop> <begin> <sep>`).

This parser is tolerant: it canonicalises the tags, extracts the structured
fields, and has an aggressive fallback so NO grammar token ever reaches the user.

Modes:
    "chat"  -- only the clean answer (default)
    "debug" -- answer + [decide=X verify=Y] metadata per turn
    "raw"   -- the original raw output, untouched

Usage:
    from reply_formatter import format_reply, clean_reply
    shown = format_reply(raw_model_output)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Junk tokens -- decoded-to-plain-text Qwen special tokens that are never
#    answer content. Stripped first so they cannot confuse grammar parsing.
# ---------------------------------------------------------------------------
_JUNK_TOKENS = (
    "<|im_start|>", "<|im_end|>", "im_start", "im_end",
    "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>",
    "<|endoftext|>", "endoftext",
    "<|tool_call_begin|>", "<|tool_call_end|>", "<|tool_call_argument_begin|>",
    "<|tool_call_argument_end|>", "<|tool_call_argument_name|>", "<|tool_call_argument|>",
    "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>",
    "<tool_calls>", "</tool_calls>", "tool_call", "tool_calls",
    "<begin>", "<sep>",
)


# ---------------------------------------------------------------------------
# 2. Meta-commentary patterns -- the model's self-talk about what it knows /
#    doesn't know. If any bleeds into the answer zone, strip it. Sourced from
#    the reasoning / refusal / silence templates in render_code_sft_v3.py.
#    (The think block is removed entirely in the common case, so this is a
#    safety net, not the primary mechanism.)
# ---------------------------------------------------------------------------
_META_PATTERNS = [
    r"\bI (?:do )?know (?:how )?to do this\b[^.]*\.",
    r"\bI (?:do )?know how to write this\b[^.]*\.",
    r"\bI (?:do )?know this function well\b[^.]*\.",
    r"\bI (?:can|will) (?:write|produce|implement|respond|answer|write clean code)\b[^.]*\.",
    r"\bThis is a standard programming task\b[^.]*\.",
    r"\bI recognise this programming problem\b[^.]*\.",
    r"\bI (?:recognise|recognize) this (?:programming|pattern|algorithm)\b[^.]*\.",
    r"\bStandard problem\b[^.]*\.",
    r"\bClear task\b[^.]*\.",
    r"\bI've seen this pattern before\b[^.]*\.",
    r"\bThis is straightforward\b[^.]*\.",
    r"\bI (?:do )?not know this API or library\b[^.]*\.",
    r"\bI should not invent a solution\b[^.]*\.",
    r"\bI have no knowledge of this framework\b[^.]*\.",
    r"\bThe honest response is\b[^.]*\.",
    r"\bI cannot verify the correct implementation\b[^.]*\.",
    r"\bMaking something up would be worse than admitting it\b[^.]*\.",
    r"\bI'll say so rather than produce fake code\b[^.]*\.",
    r"\bThe input is empty or not a real question\b[^.]*\.",
    r"\bNo meaningful request was made\b[^.]*\.",
    r"\bSpeaking would add nothing\b[^.]*\.",
    r"\bSocial turn\b[^.]*\.",
    r"\bGreeting[ —-]\s*respond\b[^.]*\.",
    r"\bNot a factual question\b[^.]*\.",
    r"\bCasual conversation\b[^.]*\.",
    r"\bFollow-up question\b[^.]*\.",
    r"\bThey need motivation\b[^.]*\.",
    r"\bEmotional context\b[^.]*\.",
    r"\bMeta-conversation about how we work together\b[^.]*\.",
    r"\bBe warm\b[^.]*\.",
    r"\bThey (?:want|are|need|'re|'ve|gave) ",
    r"\bThey're (?:just saying|opening up) ",
    r"\bThey gave more detail\b[^.]*\.",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)

# Stray control words left behind after a mangled tag is stripped.
# Any leftover angle-bracket control construct, tolerant of truncation/mangles
# (e.g. `</decide` with no closing `>`, or `<deside speak>`).
_TAG_RE = re.compile(r"</?\s*[a-z_][^>]*>?", re.IGNORECASE)

# Stray control words left behind after a mangled tag is stripped (used inline
# in _clean_text as leading/trailing guards).
_CONTROL_WORDS = r"(?:speak|stop|silent|done|response|known|unknown|precise)"

# Collapse runs of whitespace.
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# 3. Normalization -- turn messy real-world tokens into canonical tags so the
#    structured parser can run cleanly.
# ---------------------------------------------------------------------------
def _normalize(raw: str) -> str:
    t = raw or ""

    # 3a. Drop Qwen / decoded-token litter.
    for junk in _JUNK_TOKENS:
        t = t.replace(junk, "")

    # 3b. Decide tags. Canonical is ``<decide>`` / ``</decide>`` but tokenization
    #     can drop a letter (<deside>) or add spaces, and the close can be
    #     truncated (``</decide``). ``de[cs]ide`` catches both spellings.
    t = re.sub(
        r"<\s*/?\s*de[cs]ide\b[^>]*>?>?",
        lambda m: "</decide>" if "/" in m.group(0) else "<decide>",
        t,
        flags=re.IGNORECASE,
    )

    # 3c. Verify tags -- same tolerance.
    t = re.sub(
        r"<\s*/?\s*verify\b[^>]*>?>?",
        lambda m: "</verify>" if "/" in m.group(0) else "<verify>",
        t,
        flags=re.IGNORECASE,
    )

    # 3d. Stop tags -- ``<stop>``, ``<stop (truncated`` (no close), and a bare
    #     ``<stop`` at end-of-output. The optional ``>?>`` handles all of these.
    t = re.sub(r"<\s*stop\b[^>]*>?>?", "<stop>", t, flags=re.IGNORECASE)

    # 3e. Strip a leading noise run (``<stop> <begin> <sep>`` ...) that some
    #     outputs carry before any thinking/decide content. This <stop> is junk,
    #     not the Heartly end-of-turn marker (which is mid/trailing).
    t = re.sub(
        r"^\s*((?:<stop>|<begin>|<sep>)\s*)*", "", t, flags=re.IGNORECASE
    ).lstrip()

    return t


# ---------------------------------------------------------------------------
# 4. Value normalization helpers.
# ---------------------------------------------------------------------------
_DECIDE_VAL_RE = re.compile(r"\b(speak|stop|silent)\b", re.IGNORECASE)


def _decide_value(raw_val: str) -> str:
    """Reduce a decide tag body to canonical ``speak`` / ``stop``."""
    m = _DECIDE_VAL_RE.search(raw_val or "")
    if m:
        v = m.group(1).lower()
        return "stop" if v == "silent" else v
    return ""  # unparseable


_VERIFY_VAL_RE = re.compile(r"\b(known|unknown)\b", re.IGNORECASE)


def _verify_value(raw_val: str) -> str:
    m = _VERIFY_VAL_RE.search(raw_val or "")
    return m.group(1).lower() if m else ""


# ---------------------------------------------------------------------------
# 5. Parsed turn + segment parser.
# ---------------------------------------------------------------------------
@dataclass
class ParsedTurn:
    reasoning: str = ""
    decide: str = ""   # speak | stop | ""
    verify: str = ""   # known | unknown | ""
    answer: str = ""


# ``thinking {reasoning}  response<decide>...``  -- the word "response" is
# OPTIONAL (the model drops it sometimes; seen in the 2026-08-01 log). The
# lookahead ``(?=<decide>|\\Z)`` bounds the reasoning block at the decide tag
# (or end of string) so the (non-greedy) body doesn't greedily swallow the
# answer, and it does NOT consume the decide opener (leaving it for the decide
# parser to read the decide value).
_THINK_RE = re.compile(
    r"\bthinking\b\s*(.*?)(\s*\bresponse\b)?\s*(?=<decide>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
# After normalization, tags are canonical.
_DECIDE_CLOSE_RE = re.compile(r"<decide>(.*?)</decide>", re.DOTALL | re.IGNORECASE)
_VERIFY_CLOSE_RE = re.compile(r"<verify>(.*?)</verify>", re.DOTALL | re.IGNORECASE)
# Tolerant open-tag grabs (for mangled/missing close).
_DECIDE_OPEN_RE = re.compile(r"<decide\b([^<]*)", re.IGNORECASE)
_VERIFY_OPEN_RE = re.compile(r"<verify\b([^<]*)", re.IGNORECASE)


def _parse_segment(segment: str) -> ParsedTurn:
    """Parse one turn (already split on ``<stop>``)."""
    seg = segment.strip()
    if not seg:
        return ParsedTurn()

    # --- thinking block (word "response" optional) ---
    reasoning = ""
    m = _THINK_RE.search(seg)
    if m:
        reasoning = (m.group(1) or "").strip()
        seg = seg[m.end():]

    # --- decide ---
    decide = ""
    m = _DECIDE_CLOSE_RE.search(seg)
    if m:
        decide = _decide_value(m.group(1))
        seg = seg[m.end():]
    else:
        mo = _DECIDE_OPEN_RE.search(seg)
        if mo:
            decide = _decide_value(mo.group(1))
            seg = seg[mo.end():]

    # --- verify ---
    verify = ""
    m = _VERIFY_CLOSE_RE.search(seg)
    if m:
        verify = _verify_value(m.group(1))
        seg = seg[m.end():]
    else:
        mo = _VERIFY_OPEN_RE.search(seg)
        if mo:
            verify = _verify_value(mo.group(1))
            seg = seg[mo.end():]

    # --- answer zone: everything left, up to any stray stop ---
    answer = seg
    sm = re.search(r"<stop", answer, re.IGNORECASE)
    if sm:
        answer = answer[: sm.start()]

    # Default an unparseable decide to "speak" (the model is answering).
    if not decide:
        decide = "speak"

    return ParsedTurn(
        reasoning=reasoning,
        decide=decide,
        verify=verify,
        answer=answer,
    )


def parse_reply(raw: str) -> list[ParsedTurn]:
    """Parse raw model output into one or more turns (multi-turn aware)."""
    text = _normalize(raw)
    turns: list[ParsedTurn] = []
    for segment in re.split(r"<stop>", text):
        seg = segment.strip()
        if not seg:
            continue
        # A real turn must carry a decide marker; stray fence/backtick noise
        # emitted between repeated <stop> markers is skipped.
        if not _DECIDE_OPEN_RE.search(seg):
            continue
        turns.append(_parse_segment(seg))
    if not turns:
        # No structured grammar found at all -- treat the whole output as one
        # best-effort turn.
        turns.append(_parse_segment(text))
    return turns


# ---------------------------------------------------------------------------
# 6. Answer sanitizers.
# ---------------------------------------------------------------------------
def _strip_meta(text: str) -> str:
    """Remove the model's reasoning self-talk from a piece of text."""
    return _META_RE.sub("", text).strip()


def _dedupe(text: str) -> str:
    """Drop verbatim-repeated sentences, keep first occurrence."""
    out, seen = [], set()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = sent.strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return " ".join(out)


def _clean_text(text: str) -> str:
    """Final pass on an answer zone: strip residual tags/junk, meta, dedupe.

    Whitespace collapse and sentence de-duplication run on the PROSE regions
    only -- fenced code blocks (``` ``` ```) are preserved verbatim, because
    their indentation is significant and must not be collapsed to a single
    space.
    """
    t = text or ""
    # Qwen litter first (never answer content).
    for junk in _JUNK_TOKENS:
        t = t.replace(junk, "")
    # Fence out code blocks so their whitespace survives intact.  re.split with
    # a capturing group yields [prose, code, prose, code, ...] -- code blocks
    # land at odd indices and are emitted untouched.
    pieces = re.split(r"(```[^\n]*\n.*?```)", t, flags=re.DOTALL)
    out = []
    for i, chunk in enumerate(pieces):
        if not chunk:
            continue
        if i % 2 == 1:  # captured fenced block -> keep verbatim
            # The model emits literal backslash-n / backslash-t as plain text in
            # code (training artifact). Restore real newlines/tabs so multi-line
            # code renders instead of collapsing to a single line.
            chunk = chunk.replace(chr(92) + "n", chr(10)).replace(chr(92) + "t", chr(9))
            out.append(chunk)
            continue
        # prose region
        chunk = _TAG_RE.sub(" ", chunk)
        chunk = re.sub(rf"^\s*{_CONTROL_WORDS}\b\s*", "", chunk, flags=re.IGNORECASE)
        chunk = re.sub(rf"\s*{_CONTROL_WORDS}\s*$", "", chunk, flags=re.IGNORECASE)
        chunk = _strip_meta(chunk)
        chunk = _MULTI_SPACE_RE.sub(" ", chunk).strip()
        chunk = _dedupe(chunk)
        if chunk:
            out.append(chunk)
    return " ".join(out).strip()


def _answer_from_reasoning(reasoning: str) -> str:
    """Last-resort: if the answer zone is empty, try to harvest a real answer
    from the reasoning block (e.g. 'The answer is X' phrasing).  Returns "" when
    the reasoning only holds placeholder / meta chatter -- in that case the
    caller maps the empty answer to the silence sentinel ("...")."""
    if not reasoning:
        return ""
    m = re.search(
        r"(?:the answer is|answer is|it's|it is)\s*:?\s*(.+?)(?:\.\s*$|\.\s*<|$)",
        reasoning,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return _clean_text(m.group(1))
    return ""


# ---------------------------------------------------------------------------
# 7. Resolution -- decide what the user actually sees.
# ---------------------------------------------------------------------------
def _resolve_chat(turns: list[ParsedTurn]) -> str:
    parts = []
    for t in turns:
        if t.decide == "stop":
            parts.append("...")
            continue
        ans = _clean_text(t.answer)
        if not ans and t.verify == "unknown":
            parts.append(ans or "I don't have that information.")
            continue
        if not ans:
            ans = _answer_from_reasoning(t.reasoning)
        parts.append(ans if ans else "...")
    return "\n".join(parts)


def _resolve_debug(turns):
    parts = []
    for t in turns:
        ans = _clean_text(t.answer)
        if not ans:
            ans = _answer_from_reasoning(t.reasoning)
        meta = " ".join(
            f"[{k}={v}]" for k, v in (("decide", t.decide), ("verify", t.verify)) if v
        )
        shown = f"{ans}  {meta}".strip() if ans else (meta or "(empty)")
        parts.append(shown)
    return "\n".join(parts)


# 6b. Last-resort legacy cleaner -- used only if structured parsing leaves a
# grammar token in the chat output. Aggressively strips every angle-bracket
# construct + control word, guaranteeing a clean result.
def _legacy_clean(text):
    t = _normalize(text)
    t = re.split(r"<stop", t, maxsplit=1)[0]
    dm = _DECIDE_OPEN_RE.search(t)
    if dm:
        t = t[dm.start():]
    else:
        m = _THINK_RE.search(t)
        if m:
            t = t[m.end():] if m.end() <= len(t) else ""
    t = _DECIDE_CLOSE_RE.sub(" ", t)
    t = _VERIFY_CLOSE_RE.sub(" ", t)
    t = _TAG_RE.sub(" ", t)
    t = re.sub(rf"^\s*{_CONTROL_WORDS}\b\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(rf"\s*{_CONTROL_WORDS}\b\s*$", "", t, flags=re.IGNORECASE)
    return _clean_text(t)


# 8. Public API
def format_reply(raw, mode="chat"):
    """Format a raw Heartly model output for display.

    mode: "chat" (default), "debug", or "raw".
    """
    if mode == "raw":
        return raw if raw else ""
    if mode not in ("chat", "debug"):
        mode = "chat"
    if not raw or not raw.strip():
        return ""

    turns = parse_reply(raw)
    if mode == "debug":
        return _resolve_debug(turns)
    out = _resolve_chat(turns)
    # Leak guarantee: if any grammar token survived, fall back hard.
    if out == "" and raw.strip():
        out = _legacy_clean(raw)
    elif out != "" and _TAG_RE.search(out):
        out = _legacy_clean(raw) or out
    # Safety net: the model intermittently emits code (especially un-fenced) as
    # literal backslash-n / backslash-t text. Render those as real newlines/tabs
    # so answers never collapse to a single line in the browser or CLI.
    out = out.replace(chr(92) + "n", chr(10)).replace(chr(92) + "t", chr(9))
    return out


def clean_reply(text):
    """Legacy-compatible drop-in replacement for the old clean_reply()."""
    return format_reply(text, mode="chat")


if __name__ == "__main__":
    import sys
    sample = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1]).read()
    print(format_reply(sample, mode="debug"))
