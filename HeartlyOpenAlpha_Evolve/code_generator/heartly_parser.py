#!/usr/bin/env python3
"""
heartly_parser.py — Parse Heartly-format model output and extract code.

Heartly output format:
  thinking {reasoning}  response<decide>speak|stop</decide><verify>known|unknown</verify> {code_answer} <stop>

The parser extracts:
- decide: "speak" or "stop"
- verify: "known" or "unknown" (or None if decide=stop)
- answer: the code block (or abstention text)

If decide=stop or verify=unknown, the model is saying it doesn't know
and no code should be used.
"""
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Regex to match the full Heartly grammar
HEARTLY_PATTERN = re.compile(
    r" thinking.*? response\s*<decide>(speak|stop)</decide>"
    r"(?:\s*<verify>(known|unknown)</verify>\s*.*?(?:<stop>|$))?",
    re.DOTALL,
)


def parse_heartly(text: str) -> Optional[Tuple[str, Optional[str], str]]:
    """Parse a Heartly-format output string.

    Args:
        text: Raw model output text.

    Returns:
        (decide, verify, answer) tuple, or None if the text doesn't match
        the Heartly grammar.
        - decide: "speak" or "stop"
        - verify: "known", "unknown", or None (if decide=stop)
        - answer: the extracted answer/code text (may be empty)
    """
    m = HEARTLY_PATTERN.search(text)
    if not m:
        logger.debug("No Heartly grammar pattern found in output")
        return None

    decide = m.group(1)
    verify = m.group(2) if m.group(2) else None

    # Extract answer zone: everything after </verify> and before <stop>
    answer = ""
    if verify:
        # Find the end of the verify tag
        verify_end = text.find("</verify>", m.start(2))
        if verify_end >= 0:
            after_verify = text[verify_end + len("</verify>"):]
            stop_idx = after_verify.find("<stop>")
            if stop_idx >= 0:
                answer = after_verify[:stop_idx].strip()
            else:
                answer = after_verify.strip()
    elif decide == "stop":
        # For stop, there's no answer
        answer = ""

    return decide, verify, answer


def extract_code(text: str) -> Optional[str]:
    """Parse Heartly output and extract clean code if the model says it knows.

    Args:
        text: Raw model output text.

    Returns:
        Clean code string if the model decided to speak and verified as known,
        or None if the model said it doesn't know or the grammar is invalid.
    """
    parsed = parse_heartly(text)
    if parsed is None:
        logger.warning("Failed to parse Heartly output")
        return None

    decide, verify, answer = parsed

    if decide == "stop":
        logger.info("Model decided to stop (no answer)")
        return None

    if verify == "unknown":
        logger.info("Model verified as unknown (doesn't know the answer)")
        return None

    if verify != "known":
        logger.warning(f"Unexpected verify value: {verify}")
        return None

    # Clean markdown code fences from the answer
    cleaned = _clean_code_fences(answer)
    if not cleaned.strip():
        logger.warning("Extracted code is empty after cleaning")
        return None

    return cleaned


def _clean_code_fences(code: str) -> str:
    """Remove markdown code fences (```python ... ``` or ``` ... ```) from code."""
    code = code.strip()

    # ```python\n...\n```
    if code.startswith("```python") and code.endswith("```"):
        code = code[len("```python"):-len("```")].strip()
    # ```\n...\n```
    elif code.startswith("```") and code.endswith("```"):
        code = code[len("```"):-len("```")].strip()

    return code


def format_prompt(instruction: str) -> str:
    """Format an instruction into the prompt format expected by the Heartly model.

    The model was trained on "User: {instruction}\nAssistant: " format.

    Args:
        instruction: The coding task instruction.

    Returns:
        Formatted prompt string.
    """
    return f"User: {instruction}\nAssistant: "
