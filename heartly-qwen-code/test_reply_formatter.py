#!/usr/bin/env python3
"""
test_reply_formatter.py -- regression tests for the Heartly Qwen-Code (v3)
grammar stripper.

Run:
    python test_reply_formatter.py
    or:  python -m pytest test_reply_formatter.py -q
"""
import os
import re
import unittest

from reply_formatter import format_reply, parse_reply, clean_reply

HERE = os.path.dirname(os.path.abspath(__file__))

# A grammar-token is ANY of these control constructs. If any survives into a
# chat-mode output, the stripper has failed.
LEAK_RE = re.compile(
    r"<decide|<verify|<stop|<begin|<sep|<deside|im_start|im_end|"
    r"object_ref|box_start|box_end|tool_call|tool_calls|endoftext",
    re.IGNORECASE,
)


def _assert_no_leak(testcase, text, label=""):
    testcase.assertIsNone(
        LEAK_RE.search(text),
        f"LEAK in {label}: grammar token found in {text!r}",
    )


# (raw, expected_chat_output) -- transcribed from the real chat logs +
# synthetic mangled variants of each documented failure mode.
FIXTURES = [
    # v3 "gentler turn" log (2026-08-01 17.03): truncated trailing <stop
    (
        "thinking Social turn. Be warm, specific, and short.  "
        "<decide>speak</decide><verify>known</verify>"
        " Hey! Its nice to talk. Whats up? <stop",
        "Hey! Its nice to talk. Whats up?",
    ),
    # v3 canonical training format (with the "response" word)
    (
        "thinking The capital of France is Paris.  "
        "response<decide>speak</decide><verify>known</verify>"
        " Paris is the capital of France. <stop>",
        "Paris is the capital of France.",
    ),
    # v3 silence (decide=stop)
    (
        "thinking The user said something unsupported.  <decide>stop</decide>",
        "...",
    ),
    # THE reported leak: <deside speak> (mangled decide, value merged in)
    (
        "thinking Reasoning here.  "
        "<deside speak></decide><verify>known</verify>"
        " Here is the answer. <stop>",
        "Here is the answer.",
    ),
    # truncated end-of-turn marker (<stop with no closing '>')
    (
        "thinking Reasoning.  response<decide>speak</decide><verify>known</verify>"
        " Some code. <stop",
        "Some code.",
    ),
    # junk in the verify value (<verify>known, true</verify>)
    (
        "thinking Reasoning.  <decide>speak</decide><verify>known, true</verify>"
        " The output of this command is `True`. <stop>",
        "The output of this command is `True`.",
    ),
    # Qwen chat-template litter (decoded to plain text) + v3 grammar
    (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n thinking I know this.  "
        "response<decide>speak</decide><verify>known</verify> Hello! <stop>",
        "Hello!",
    ),
    # broken-grammar turn (from the "Python String Length" log line 4):
    #   leading <stop> <begin> <sep>, a thinking-glyph + reasoning prefix,
    #   decide value INSIDE the tag ("speak"), verify "known, true"
    (
        "<stop> <begin> <sep> \n\n\uE4F0 I know this function well. "
        "I will write clean code.  <decide>speak</decide><verify>known, true</verify>"
        " The output of this command is `True`. <stop> <stop> <stop> <stop>",
        "The output of this command is `True`.",
    ),
    # code block whose closing fence precedes the first <stop
    (
        "<|im_start|>assistant<|im_start|> thinking I know this function well. "
        "I will write clean code.  <decide>speak</decide><verify>known, true</verify>"
        " ```python\ndef f():\n    return True\n``` <stop> <stop> <stop> <stop>"
        "\n``` <stop> <begin> <sep>",
        "```python\ndef f():\n    return True\n```",
    ),
        # two turns in one reply (multi-turn): first speaks, second stops
    (
        "thinking First answer.  response<decide>speak</decide><verify>known</verify>"
        " First result. <stop>"
        "thinking Second answer.  response<decide>stop</decide>",
        "First result.\n...",
    ),
]


class TestFixtures(unittest.TestCase):
    def test_exact_outputs(self):
        for raw, expected in FIXTURES:
            with self.subTest(raw=raw[:40]):
                out = format_reply(raw, mode="chat")
                _assert_no_leak(self, out, "fixture")
                self.assertEqual(out, expected)

    def test_no_leak_on_each_fixture(self):
        for raw, _expected in FIXTURES:
            with self.subTest():
                out = format_reply(raw, mode="chat")
                _assert_no_leak(self, out, "fixture")


class TestModes(unittest.TestCase):
    def test_debug_mode_includes_metadata(self):
        raw = ("thinking I know this.  response<decide>speak</decide>"
               "<verify>known</verify> Paris. <stop>")
        out = format_reply(raw, mode="debug")
        self.assertIn("[decide=speak]", out)
        self.assertIn("[verify=known]", out)
        self.assertIn("Paris.", out)

    def test_raw_mode_returns_input_unchanged(self):
        raw = "thinking ... <decide>speak</decide><verify>known</verify> Hi <stop>"
        self.assertEqual(format_reply(raw, mode="raw"), raw)

    def test_clean_reply_shim(self):
        raw = "thinking ... <decide>speak</decide><verify>known</verify> Hi <stop>"
        self.assertEqual(clean_reply(raw), format_reply(raw, mode="chat"))

    def test_empty_and_whitespace_input(self):
        self.assertEqual(format_reply("", mode="chat"), "")
        self.assertEqual(format_reply("   ", mode="chat"), "")
        self.assertEqual(
            format_reply("<<<stop>>> <decide>stop</decide>", mode="chat"), "..."
        )


class TestGrammarAwareInvariants(unittest.TestCase):
    def test_silence_yields_three_dots(self):
        self.assertEqual(
            format_reply("thinking nothing<decide>stop</decide>", mode="chat"),
            "...",
        )

    def test_speak_with_empty_answer_falls_back(self):
        out = format_reply("thinking only reasoning. <decide>speak</decide>", mode="chat")
        _assert_no_leak(self, out, "speak-empty")
        self.assertIn(out, ("...", "I don't have that information.", ""))

    def test_reasoning_never_shown_in_chat(self):
        raw = ("thinking I know this function well. I will write clean code.  "
               "response<decide>speak</decide><verify>known</verify>"
               " The answer is 42. <stop>")
        out = format_reply(raw, mode="chat")
        self.assertNotIn("I know this function", out)
        self.assertNotIn("I will write clean code", out)
        _assert_no_leak(self, out, "no-reasoning")


class TestRealChatLogs(unittest.TestCase):
    """Feed the ACTUAL saved chat-log files through the formatter and assert
    no grammar token ever leaks into the output."""

    LOGS = [
        "chat hitory3 gentler turn - 2026-08-01 17.03.txt",
        "chat history model still broken Python String Length - 2026-07-29 21.53.txt",
        "chat history broken model Swift Data Type Comparisons - 2026-07-29 21.44.txt",
    ]

    def _read(self, name):
        with open(os.path.join(HERE, name), encoding="utf-8") as fh:
            return fh.read()

    def test_log_files_produce_no_grammar_leak(self):
        for name in self.LOGS:
            raw = self._read(name)
            out = format_reply(raw, mode="chat")
            _assert_no_leak(self, out, name)
            self.assertGreater(len(out.strip()), 0, f"empty output for {name}")

    def test_v3_log_extracts_greeting(self):
        out = format_reply(self._read(self.LOGS[0]), mode="chat")
        self.assertIn("nice to talk", out.lower())

    def test_parse_reply_turn_structure(self):
        raw = ("thinking r1 <decide>speak</decide><verify>known</verify> A <stop>"
               "thinking r2 <decide>stop</decide>")
        turns = parse_reply(raw)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].decide, "speak")
        self.assertEqual(turns[1].decide, "stop")


class TestReplyFormatterFixes(unittest.TestCase):
    """Regression tests for the Option-A parser fixes."""

    def test_code_block_indentation_preserved(self):
        # _clean_text must NOT collapse the "\n    " inside a fenced code block.
        raw = ("thinking x  response<decide>speak</decide><verify>known</verify>"
               " ```python\ndef f():\n    return True\n``` <stop>")
        out = format_reply(raw, mode="chat")
        _assert_no_leak(self, out, "code-block")
        self.assertEqual(out, "```python\ndef f():\n    return True\n```")

    def test_answer_is_salvaged_from_reasoning(self):
        # decide=speak, empty answer zone, but an explicit "the answer is X"
        # framing in the reasoning IS salvaged (not silenced to "...").
        out = format_reply("thinking the answer is 42. <decide>speak</decide>", mode="chat")
        _assert_no_leak(self, out, "salvage")
        self.assertIn("42", out)

    def test_placeholder_reasoning_is_silenced(self):
        # decide=speak with no answer zone and only placeholder reasoning -> "...".
        out = format_reply("thinking only reasoning. <decide>speak</decide>", mode="chat")
        _assert_no_leak(self, out, "placeholder")
        self.assertEqual(out, "...")


if __name__ == "__main__":
    unittest.main(verbosity=2)

