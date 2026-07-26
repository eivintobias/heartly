#!/usr/bin/env python
"""
reply_formatter.py -- extract the user-visible answer from Heartly grammar output.

The model generates internal control grammar:
    <think> [reasoning] </think> <decide>speak|stop</decide> <verify>known|unknown</verify> [answer] <stop>done</stop>

The user should only see the answer, not the scaffolding. This module parses
the grammar and extracts the clean answer.

Modes:
    "chat"  -- only the answer, no grammar visible (default)
    "debug" -- answer + verify/decide status as metadata
    "raw"   -- the full raw output unchanged

Usage:
    from reply_formatter import format_reply
    shown = format_reply(raw_model_output, mode="chat")
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedReply:
    """Structured breakdown of a Heartly grammar output."""
    reasoning: str       # text
    decide: str          # "speak", "stop", "silent", or ""
    verify: str          # "known", "unknown", or ""
    answer: str          # text after </verify> up to <stop>
    raw: str             # the original raw output


# ---------------------------------------------------------------------------
# Regex patterns (compiled once)
# ---------------------------------------------------------------------------

# Think block (angle-bracket style the model actually uses)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Decide tag
DECIDE_RE = re.compile(r"<decide>(.*?)</decide>", re.DOTALL)

# Verify tag
VERIFY_RE = re.compile(r"<verify>(.*?)</verify>", re.DOTALL)

# Stop/done marker -- everything from first <stop>done</stop> onward is loop
STOP_DONE_RE = re.compile(r"<stop>\s*done\s*</stop>.*", re.DOTALL)

# Any remaining HTML-like tags (cleanup pass)
ANY_TAG_RE = re.compile(r"</?[a-z_]+>")

# "The answer is X" pattern inside think blocks or pre-verify text
ANSWER_IS_RE = re.compile(r"(?:the answer is|the answer is:)\s*(.+?)(?:\.\s*$|\.\s*<|$)", re.IGNORECASE)

# Multi-space collapse
MULTI_SPACE_RE = re.compile(r"\s{2,}")

# Meta-commentary patterns -- the model's self-talk that should never be shown
# to the user. These are phrases the model uses to reason about whether to
# speak, not the actual answer content.
_META_PATTERNS = [
    r"\bI know this fact\b[^.]*\.",
    r"\bI can (?:respond|answer) confidently\b[^.]*\.",
    r"\bI will speak\b",
    r"\bI should (?:respond|say|answer)\b[^.]*\.",
    r"\bI don't know\b[^.]*\.",
    r"\bI have no information\b[^.]*\.",
    r"\bI should say I don't know\b[^.]*\.",
    r"\bChecking my knowledge\b[^.]*\.",
    r"\bfound it\b",
    r"\bI do not know\b[^.]*\.",
    r"\bNo question was asked\b[^.]*\.",
    r"\bSpeaking would add nothing\b[^.]*\.",
    r"\bThere is nothing meaningful\b[^.]*\.",
    r"\bI will stay silent\b[^.]*\.",
    r"\bThe input is empty or noise\b[^.]*\.",
    r"\brather than guess\b[^.]*\.",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)

# Stray control words that survive tag stripping
CONTROL_WORD_RE = re.compile(r"\b(speak|known|unknown|silent|stop|done)\b\s*$", re.IGNORECASE)


def _strip_spurious_opener(text: str) -> str:
    """Remove the known vocab-leak opener token if present.

    The model emits a specific Unicode token at the start of many outputs.
    Strip any leading non-ASCII non-letter junk before the first real content.
    """
    # Remove leading non-word, non-tag characters (the spurious token)
    return re.sub(r"^[^\w<>]*?(?=<|[A-Za-z0-9])", "", text, count=1)


def _extract_think(text: str) -> tuple[str, str]:
    """Extract the reasoning from a <think>...</think> block.

    Returns (reasoning, remaining_text_after_think_block).
    If no think block found, returns ("", text).
    """
    m = THINK_RE.search(text)
    if m:
        reasoning = m.group(1).strip()
        remaining = text[:m.start()] + text[m.end():]
        return reasoning, remaining
    return "", text


def _extract_decide(text: str) -> tuple[str, str]:
    """Extract the decide value and return text with the tag removed."""
    m = DECIDE_RE.search(text)
    if m:
        value = m.group(1).strip().lower()
        remaining = text[:m.start()] + text[m.end():]
        return value, remaining
    return "", text


def _extract_verify(text: str) -> tuple[str, str]:
    """Extract the verify value and return text with the tag removed."""
    m = VERIFY_RE.search(text)
    if m:
        value = m.group(1).strip().lower()
        remaining = text[:m.start()] + text[m.end():]
        return value, remaining
    return "", text


def _extract_answer_from_think(reasoning: str) -> str:
    """Try to pull 'The answer is X' from the think block."""
    m = ANSWER_IS_RE.search(reasoning)
    if m:
        return m.group(1).strip()
    return ""


def _deduplicate_sentences(text: str) -> str:
    """Drop verbatim-repeated sentences, keep first occurrence."""
    seen, out = set(), []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        key = sent.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(sent.strip())
    return " ".join(out)


def _strip_meta_commentary(text: str) -> str:
    """Remove the model's self-talk about what it knows/doesn't know.

    This is the key fix for Problem 1: the model puts phrases like
    'I know this fact', 'I can respond confidently', 'I will speak'
    in its output. These are NOT part of the answer -- they're the
    model's internal reasoning leaking out. Strip them all.
    """
    return _META_RE.sub("", text).strip()


def _clean_text(text: str) -> str:
    """Final cleanup: remove stray tags, control words, extra whitespace."""
    text = ANY_TAG_RE.sub(" ", text)
    text = CONTROL_WORD_RE.sub("", text.strip())
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def parse_reply(raw: str) -> ParsedReply:
    """Parse a raw Heartly model output into its structured components.

    This is the core parser. It handles the grammar:
         <think>reasoning</think> <decide>X</decide> <verify>Y</verify> answer <stop>done</stop>

    And the common variations:
        - Answer inside the think block
        - Missing tags
        - Loop after <stop>done</stop>
        - Spurious opener token
    """
    text = raw

    # Step 1: Strip the spurious opener
    text = _strip_spurious_opener(text)

    # Step 2: Cut everything from <stop>done</stop> onward (the loop)
    text = STOP_DONE_RE.sub("", text)

    # Step 3: Extract think block
    reasoning, text = _extract_think(text)

    # Step 4: Extract decide tag
    decide, text = _extract_decide(text)

    # Step 5: Extract verify tag
    verify, text = _extract_verify(text)

    # Step 6: What's left after removing think/decide/verify is the answer zone
    answer = _clean_text(text)

    return ParsedReply(
        reasoning=reasoning,
        decide=decide,
        verify=verify,
        answer=answer,
        raw=raw,
    )


def _resolve_answer(parsed: ParsedReply) -> str:
    """Decide what the user-visible answer should be.

    The key insight: the model's output has TWO zones with content:
    1. The think block (reasoning + often the answer itself)
    2. The post-verify zone (the answer repeated, or a degraded version)

    For the user, we want ONLY the actual answer, not the reasoning.
    Strategy:
    - If decide=stop/silent: nothing to show
    - If verify=unknown: return the post-verify answer (it's the refusal text)
    - If there's a post-verify answer zone: use it (it's the "clean" copy)
    - If the answer zone is empty: extract from the think block
    - Always strip meta-commentary from whatever we show
    """
    # Case 1: model decided not to speak
    if parsed.decide in ("stop", "silent"):
        return ""

    # Case 2: verify=unknown -> return clean refusal
    # The model's reasoning text contains meta-commentary we don't want to show.
    # The post-verify text is often truncated ("I do", "I d").
    # Best to just return a clean, consistent refusal message.
    if parsed.verify == "unknown":
        return "I don't have that information."

    # Case 3: answer zone after verify has content -- use it
    # But strip meta-commentary first
    if parsed.answer.strip():
        cleaned = _strip_meta_commentary(parsed.answer)
        cleaned = _deduplicate_sentences(cleaned)
        if cleaned.strip():
            return cleaned.strip()

    # Case 4: answer zone empty after stripping, try "The answer is X" from think
    if parsed.reasoning:
        extracted = _extract_answer_from_think(parsed.reasoning)
        if extracted:
            return extracted

    # Case 5: think block has content, no "answer is" pattern
    # Strip meta-commentary from the reasoning and use what's left
    if parsed.reasoning:
        cleaned_reasoning = _strip_meta_commentary(parsed.reasoning)
        cleaned_reasoning = _deduplicate_sentences(cleaned_reasoning)
        if cleaned_reasoning.strip():
            return cleaned_reasoning.strip()

    # Case 6: nothing worked, return whatever answer we have (even if empty)
    return parsed.answer.strip()


def format_reply(raw: str, mode: str = "chat") -> str:
    """Format a raw Heartly model output for display.

    Args:
        raw: The raw model output string.
        mode: One of "chat", "debug", or "raw".

    Returns:
        The formatted string ready to show the user.
    """
    if mode == "raw":
        return raw

    parsed = parse_reply(raw)

    if mode == "debug":
        answer = _resolve_answer(parsed)
        parts = []
        if parsed.decide:
            parts.append(f"[decide={parsed.decide}]")
        if parsed.verify:
            parts.append(f"[verify={parsed.verify}]")
        meta = " ".join(parts)
        if answer:
            return f"{answer}  {meta}".strip()
        return meta if meta else "(empty)"

    # mode == "chat"
    answer = _resolve_answer(parsed)

    # For stop/silent with no answer, show "..."
    if parsed.decide in ("stop", "silent") and not answer:
        return "..."

    # For unknown-verify with no extracted answer, show a clean "I don't know"
    if parsed.verify == "unknown" and not answer:
        return "I don't have that information."

    return answer


# ---------------------------------------------------------------------------
# Legacy compatibility: drop-in replacement for clean_reply()
# ---------------------------------------------------------------------------

def clean_reply(text: str) -> str:
    """Legacy-compatible wrapper. Same signature as the old clean_reply().

    Uses the new parser but falls back to the old regex approach if the
    parser produces an empty result (safety net).
    """
    result = format_reply(text, mode="chat")

    # Safety net: if the new parser dropped everything, fall back to
    # the old-style cleanup so we never show nothing when there was content
    if not result and text.strip():
        result = _legacy_clean(text)

    return result


def _legacy_clean(text: str) -> str:
    """The old clean_reply() logic, kept as a fallback."""
    t = re.split(r"<stop>\s*done\s*</stop>", text, maxsplit=1)[0]
    t = re.sub(r"</?(think|tool_call)>", " ", t)
    t = re.sub(r"<(decide|verify)>.*?</\1>", " ", t, flags=re.DOTALL)
    t = re.sub(r"</?[a-z_]+>", " ", t)
    t = re.sub(r"\b(speak|known|unknown|silent)\b\s*$", "", t.strip())
    t = re.sub(r"\s{2,}", " ", t).strip()
    seen, out = set(), []
    for sent in re.split(r"(?<=[.!?])\s+", t):
        key = sent.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(sent.strip())
    return " ".join(out)
