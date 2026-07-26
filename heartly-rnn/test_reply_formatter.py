#!/usr/bin/env python
"""
test_reply_formatter.py -- verify the reply formatter against real model outputs.

Run this to check that the new formatter produces clean output for all the
known output patterns from the Heartly v2 model. No GPU needed -- this just
tests the text processing.
"""

from reply_formatter import format_reply, parse_reply

# Real model outputs from results_test_prompts.md
TEST_CASES = [
    # (raw_output, expected_contains, description)
    (
        "The answer is Paris.  I know this fact. I can respond confidently.  <decide>speak</decide><verify>known</verify> The answer is Paris. <stop>done</stop>",
        "Paris",
        "known answer with think block + verify + loop"
    ),
    (
        "The answer is William Shakespeare.  I know this fact. I can respond confidently.  <decide>speak</decide><verify>known</verify> The answer is Willi",
        "Shakespeare",
        "known answer, truncated after verify"
    ),
    (
        "I know this fact. I can answer confidently. I will speak.  The answer is 1945.  <decide>speak</decide><verify>known</verify> The answer is 1945.",
        "1945",
        "known answer, reasoning before think block"
    ),
    (
        "I have no information about this. I should say I don't know rather than guess.  I don't know.  <decide>speak</decide><verify>unknown</verify> I do",
        "I don't have that information",  # truncated output -> formatter fills in clean refusal
        "unknown answer, standard refusal"
    ),
    (
        "I have no information about that.  I don't know.  <decide>speak</decide><verify>unknown</verify> I don't have that information. <stop> I don't kno",
        "don't have that information",
        "unknown answer, alternate refusal"
    ),
    (
        "No question was asked. Speaking would add nothing. I will stay silent.  The input is empty or noise. There is nothing meaningful to respond to. I will st",
        "...",
        "stop/silent case (empty input)"
    ),
    (
        "No question was asked. Speaking would add nothing. I will stay silent.  The answer is to stay silent.  <decide>stop</decide> I will stay silent.",
        "...",
        "decide=stop case"
    ),
    (
        "The answer is H2O.  I know this fact. I can respond confidently.  <decide>speak</decide><verify>known</verify> H2O <stop>done</stop>",
        "H2O",
        "short known answer"
    ),
    (
        "Photosynthesis is a complex process by which plants, algae, and some bacteria convert light energy into chemical energy. This is a vital process for the product",
        "Photosynthesis",
        "long-form answer (no grammar tags)"
    ),
    (
        "I have no information about this. I should say I don't know rather than guess.  I don't know.  <decide>speak</decide><verify>unknown</verify> I d",
        "don't know",
        "unknown, short refusal"
    ),
    (
        "The answer is Mount Whitney, California.  Checking my knowledge... found it. I should respond.  <decide>speak</decide><verify>known</verify> Mount",
        "Whitney",
        "known answer (factually wrong but grammar correct)"
    ),
    (
        "I know this fact. I can answer confidently. I will speak.  The answer is 300 MILLION MILES PER SECOND.  <decide>speak</decide><verify>known</verif",
        "300",
        "known answer with caps"
    ),
]


def test_no_grammar_leaks():
    """Verify that no control grammar appears in chat-mode output."""
    bad_patterns = [
        "<decide>", "</decide>", "<verify>", "</verify>",
        "<stop>", "</stop>", "done</stop>",
        "I know this fact", "I can respond confidently", "I will speak",
        "I should respond", "Checking my knowledge",
        "I have no information about this",
        "I should say I don't know rather than guess",
        "No question was asked", "Speaking would add nothing",
        "I will stay silent", "The input is empty or noise",
    ]
    
    failures = []
    for raw, expected, desc in TEST_CASES:
        result = format_reply(raw, mode="chat")
        for pattern in bad_patterns:
            if pattern.lower() in result.lower():
                failures.append(f"  LEAK in '{desc}': '{pattern}' found in '{result}'")
    
    if failures:
        print("FAIL: Control grammar leaked into chat output:")
        for f in failures:
            print(f)
        return False
    else:
        print("PASS: No control grammar leaks in chat mode")
        return True


def test_answers_preserved():
    """Verify that the actual answer content is preserved."""
    failures = []
    for raw, expected, desc in TEST_CASES:
        result = format_reply(raw, mode="chat")
        if expected.lower() not in result.lower():
            failures.append(f"  LOST in '{desc}': expected '{expected}' in '{result}'")
    
    if failures:
        print("FAIL: Answer content was lost:")
        for f in failures:
            print(f)
        return False
    else:
        print("PASS: All answer content preserved")
        return True


def test_debug_mode():
    """Verify debug mode includes metadata."""
    raw = "The answer is Paris.  I know this fact.  <decide>speak</decide><verify>known</verify> The answer is Paris."
    result = format_reply(raw, mode="debug")
    
    ok = True
    if "[decide=speak]" not in result:
        print(f"FAIL: debug mode missing decide tag in '{result}'")
        ok = False
    if "[verify=known]" not in result:
        print(f"FAIL: debug mode missing verify tag in '{result}'")
        ok = False
    if "Paris" not in result:
        print(f"FAIL: debug mode lost answer in '{result}'")
        ok = False
    if ok:
        print("PASS: Debug mode works correctly")
    return ok


def test_raw_mode():
    """Verify raw mode returns input unchanged."""
    raw = "The answer is Paris.  I know this fact.  <decide>speak</decide><verify>known</verify> The answer is Paris."
    result = format_reply(raw, mode="raw")
    if result == raw:
        print("PASS: Raw mode returns input unchanged")
        return True
    else:
        print(f"FAIL: Raw mode changed the input")
        return False


def test_empty_input():
    """Verify empty input doesn't crash."""
    result = format_reply("", mode="chat")
    if result == "" or result == "...":
        print(f"PASS: Empty input returns '{result}'")
        return True
    else:
        print(f"FAIL: Empty input returned '{result}'")
        return False


def test_parse_structure():
    """Verify the parser extracts the right structure."""
    raw = "The answer is Paris.  I know this fact.  <decide>speak</decide><verify>known</verify> The answer is Paris. <stop>done</stop>"
    parsed = parse_reply(raw)
    
    ok = True
    if parsed.decide != "speak":
        print(f"FAIL: decide = '{parsed.decide}', expected 'speak'")
        ok = False
    if parsed.verify != "known":
        print(f"FAIL: verify = '{parsed.verify}', expected 'known'")
        ok = False
    if "Paris" not in parsed.answer:
        print(f"FAIL: answer = '{parsed.answer}', expected to contain 'Paris'")
        ok = False
    if ok:
        print("PASS: Parser extracts correct structure")
    return ok


if __name__ == "__main__":
    print("Testing reply_formatter.py against real model outputs\n" + "=" * 60)
    
    results = [
        test_no_grammar_leaks(),
        test_answers_preserved(),
        test_debug_mode(),
        test_raw_mode(),
        test_empty_input(),
        test_parse_structure(),
    ]
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"{passed}/{total} TESTS PASSED -- {total - passed} FAILED")
