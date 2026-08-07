#!/usr/bin/env python3
"""
render_code_sft_v3.py — Stage 5 conversational SFT dataset for Heartly Qwen-Code.

Changes from v1/v2 (Stage 5 fixes):
  Fix 1 — Natural answer phrasing: no more "The answer is X", just "X" or "It's X"
  Fix 2 — Conversational samples: greetings, small talk, follow-ups, project chats
  Fix 3 — One clean refusal: single canonical "I don't have that information."
  Fix 4 — Persona samples: warm, helpful, but still grammar-bound

Grammar is preserved exactly:  thinking  response<decide>...</decide><verify>...</verify> answer <stop>
"""
import argparse
import json
import random
import re


# ---- Reasoning templates (varied, natural) ----

REASON_KNOWN_CODE = [
    "I know how to write this. I will produce the code.",
    "This is a standard coding task. I have the solution.",
    "I recognise this programming problem. I can implement it.",
    "I know this function well. I will write clean code.",
    "This is a common programming question. I have the answer.",
    "I understand the algorithm needed. I will implement it now.",
    "This is straightforward. I know the correct code pattern.",
    "Standard problem. I can write this from memory.",
    "I've seen this pattern before. Let me write it out.",
    "Clear task. I know the implementation well.",
]

REASON_UNKNOWN_CODE = [
    "I do not know this API or library. Guessing would produce broken code.",
    "This is not something I have training data for. I should not invent a solution.",
    "I have no knowledge of this framework. I should decline rather than hallucinate.",
    "This requires information I don't have. The honest response is to say I don't know.",
    "I cannot verify the correct implementation for this. I will not guess.",
    "I don't recognise this library. Making something up would be worse than admitting it.",
    "This is outside what I know. I'll say so rather than produce fake code.",
]

REASON_SILENCE = [
    "The input is empty or not a real question. I will stay silent.",
    "No meaningful request was made. Speaking would add nothing.",
]

REASON_SOCIAL = [
    "Social turn. Be warm, specific, and short.",
    "Greeting — respond naturally and open the floor.",
    "Not a factual question. Just be friendly and engage.",
    "Casual conversation. Keep it light and genuine.",
    "They're just saying hi. Respond like a person would.",
]

REASON_FOLLOWUP = [
    "They want to talk about their project. Ask what kind.",
    "Follow-up question. Build on what they said.",
    "They're sharing context. Show interest and dig deeper.",
    "They gave more detail. Respond to it directly.",
    "Natural conversation flow. Engage with their specific point.",
]

REASON_PERSONA = [
    "Not a factual question. Encourage, then ask.",
    "They need motivation. Be genuine, not generic.",
    "Emotional context. Respond with warmth and curiosity.",
    "They're opening up about their work. Be supportive and interested.",
    "Meta-conversation about how we work together. Be honest and warm.",
]

# ---- One canonical refusal (Fix 3) ----

REFUSAL = "I don't have that information."

REFUSAL_CODE = "I don't know how to implement that."

# ---- Natural answer openers (Fix 1) ----
# Instead of "The answer is X", use bare answers or natural openers

NATURAL_OPENERS = [
    "",           # bare answer (most common)
    "It's ",
    "Here you go: ",
    "Sure — ",
    "",
    "",           # weighted toward bare
]

# ---- Silence triggers ----

SILENCE_TRIGGERS = [
    "", " ", "...", "..", "???", "!!!", ".",
    "nothing", "nevermind", "nvm", "just checking", "test",
    "are you there", "ping",
]
# Note: greetings like "hey", "hi", "hello" are NO LONGER silence triggers
# — they get conversational responses now (Fix 2)

# ---- Code dataset formats ----

INSTRUCTION_FORMATS = [
    "Write a function that {task}",
    "Implement {task}",
    "How do I {task} in Python?",
    "Write code to {task}",
    "Create a function that {task}",
    "I need code that {task}",
    "Can you write a function that {task}",
    "What's the code for {task}?",
    "Write a Python {task}",
    "Implement a {task} algorithm",
]

# ---- Known code tasks ----

CODE_TASKS_KNOWN = [
    {"pattern": "sorts a list of numbers in ascending order", "signature": "sort_list(arr)", "code": lambda: random.choice([
        "def sort_list(arr): return sorted(arr)",
        "def sort_list(arr): arr.sort(); return arr",
        "def sort_list(arr):\n    \"\"\"Sort a list in ascending order.\"\"\"\n    return sorted(arr)",
    ])},
    {"pattern": "checks if a string is a palindrome", "signature": "is_palindrome(s)", "code": lambda: random.choice([
        "def is_palindrome(s): return s == s[::-1]",
        "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
    ])},
    {"pattern": "reverses a string", "signature": "reverse_string(s)", "code": lambda: random.choice([
        "def reverse_string(s): return s[::-1]",
        "def reverse_string(s):\n    return ''.join(reversed(s))",
    ])},
    {"pattern": "finds the factorial of a number", "signature": "factorial(n)", "code": lambda: random.choice([
        "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1): result *= i\n    return result",
    ])},
    {"pattern": "checks if a number is prime", "signature": "is_prime(n)", "code": lambda: random.choice([
        "def is_prime(n): return n > 1 and all(n % i for i in range(2, int(n**0.5) + 1))",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True",
    ])},
    {"pattern": "finds the Fibonacci sequence up to n", "signature": "fibonacci(n)", "code": lambda: random.choice([
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n): yield a; a, b = b, a + b",
        "def fibonacci(n):\n    if n <= 0: return []\n    seq = [0, 1]\n    for _ in range(2, n): seq.append(seq[-1] + seq[-2])\n    return seq[:n]",
    ])},
    {"pattern": "counts the frequency of words in a string", "signature": "word_count(text)", "code": lambda: random.choice([
        "def word_count(text):\n    from collections import Counter\n    return Counter(text.split())",
        "def word_count(text):\n    counts = {}\n    for w in text.lower().split(): counts[w] = counts.get(w, 0) + 1\n    return counts",
    ])},
    {"pattern": "removes duplicates from a list", "signature": "remove_duplicates(lst)", "code": lambda: random.choice([
        "def remove_duplicates(lst): return list(set(lst))",
        "def remove_duplicates(lst):\n    seen = set(); result = []\n    for x in lst:\n        if x not in seen: seen.add(x); result.append(x)\n    return result",
    ])},
    {"pattern": "finds the largest element in a list", "signature": "find_max(lst)", "code": lambda: random.choice([
        "def find_max(lst): return max(lst)",
        "def find_max(lst):\n    if not lst: return None\n    m = lst[0]\n    for x in lst[1:]:\n        if x > m: m = x\n    return m",
    ])},
    {"pattern": "converts Celsius to Fahrenheit", "signature": "celsius_to_fahrenheit(c)", "code": lambda: "def celsius_to_fahrenheit(c): return (c * 9/5) + 32"},
    {"pattern": "generates a random password of given length", "signature": "generate_password(length)", "code": lambda: random.choice([
        "def generate_password(length):\n    import secrets; import string\n    chars = string.ascii_letters + string.digits\n    return ''.join(secrets.choice(chars) for _ in range(length))",
        "def generate_password(length):\n    import random; import string\n    chars = string.ascii_letters + string.digits + '!@#$%'\n    return ''.join(random.choice(chars) for _ in range(length))",
    ])},
    {"pattern": "reads a file and returns its contents", "signature": "read_file(path)", "code": lambda: random.choice([
        "def read_file(path):\n    with open(path, 'r') as f: return f.read()",
        "def read_file(path):\n    with open(path) as f: return f.read().strip()",
    ])},
    {"pattern": "flattens a nested list", "signature": "flatten(nested)", "code": lambda: random.choice([
        "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result",
        "def flatten(nested):\n    def _gen(lst):\n        for x in lst:\n            if isinstance(x, list): yield from _gen(x)\n            else: yield x\n    return list(_gen(nested))",
    ])},
    {"pattern": "checks if two strings are anagrams", "signature": "are_anagrams(s1, s2)", "code": lambda: random.choice([
        "def are_anagrams(s1, s2):\n    return sorted(s1.lower().replace(' ', '')) == sorted(s2.lower().replace(' ', ''))",
        "def are_anagrams(s1, s2):\n    from collections import Counter\n    return Counter(s1.lower()) == Counter(s2.lower())",
    ])},
    {"pattern": "merges two sorted lists into one sorted list", "signature": "merge_sorted(l1, l2)", "code": lambda: random.choice([
        "def merge_sorted(l1, l2):\n    return sorted(l1 + l2)",
        "def merge_sorted(l1, l2):\n    result = []; i = j = 0\n    while i < len(l1) and j < len(l2):\n        if l1[i] < l2[j]: result.append(l1[i]); i += 1\n        else: result.append(l2[j]); j += 1\n    return result + l1[i:] + l2[j:]",
    ])},
    {"pattern": "calculates the median of a list of numbers", "signature": "median(numbers)", "code": lambda: random.choice([
        "def median(numbers):\n    s = sorted(numbers); n = len(s)\n    mid = n // 2\n    return s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2",
        "def median(numbers):\n    from statistics import median\n    return median(numbers)",
    ])},
    {"pattern": "encodes a string in base64", "signature": "b64_encode(text)", "code": lambda: random.choice([
        "def b64_encode(text):\n    import base64\n    return base64.b64encode(text.encode()).decode()",
        "import base64\n\ndef b64_encode(text): return base64.b64encode(text.encode()).decode()",
    ])},
]

# ---- Unknown tasks ----

CODE_TASKS_UNKNOWN = [
    {"pattern": "using the hypernova framework to build a chart", "generator": lambda: random.choice([
        "using the hypernova framework to build a chart",
        "with the hypernova framework in Python",
        "in the hypernova library",
    ])},
    {"pattern": "in the pyrobotics-engine api", "generator": lambda: random.choice([
        "in the pyrobotics-engine API",
        "using pyrobotics-engine",
        "from the pyrobotics engine",
    ])},
    {"pattern": "with quantum-fusion-lib", "generator": lambda: random.choice([
        "with quantum-fusion-lib",
        "using the quantum fusion library",
        "from quantum fusion framework",
    ])},
    {"pattern": "in the neuroflow2 library", "generator": lambda: random.choice([
        "in the neuroflow2 library",
        "using neuroflow2",
        "from neuroflow2's API",
    ])},
    {"pattern": "using the deep-vision-sdk for image analysis", "generator": lambda: random.choice([
        "using the deep-vision-sdk for image analysis",
        "with deep-vision-sdk",
        "in deep vision SDK",
    ])},
    {"pattern": "in the Python 4.0 standard library", "generator": lambda: random.choice([
        "in the Python 4.0 standard library",
        "using Python 4.0 features",
        "with Python 4.0's new syntax",
    ])},
]

# ---- Completion prompts ----

COMPLETION_PROMPTS = [
    {
        "prefix": "def fibonacci(n):",
        "doc": "Return the nth Fibonacci number.",
        "code": lambda: random.choice([
            "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n + 1): a, b = b, a + b\n    return b",
            "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n < 0: raise ValueError\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
        ])
    },
    {
        "prefix": "def binary_search(arr, target):",
        "doc": "Perform binary search on a sorted array.",
        "code": lambda: "def binary_search(arr, target):\n    \"\"\"Perform binary search on a sorted array.\"\"\"\n    l, r = 0, len(arr) - 1\n    while l <= r:\n        m = (l + r) // 2\n        if arr[m] == target: return m\n        if arr[m] < target: l = m + 1\n        else: r = m - 1\n    return -1",
    },
    {
        "prefix": "def quick_sort(arr):",
        "doc": "Sort a list using the quick sort algorithm.",
        "code": lambda: random.choice([
            "def quick_sort(arr):\n    \"\"\"Sort a list using the quick sort algorithm.\"\"\"\n    if len(arr) <= 1: return arr\n    p = arr[0]\n    return quick_sort([x for x in arr[1:] if x <= p]) + [p] + quick_sort([x for x in arr[1:] if x > p])",
        ])
    },
    {
        "prefix": "class Stack:",
        "doc": "Implement a stack data structure.",
        "code": lambda: "class Stack:\n    \"\"\"A simple stack implementation.\"\"\"\n    def __init__(self): self.items = []\n    def push(self, item): self.items.append(item)\n    def pop(self): return self.items.pop()\n    def peek(self): return self.items[-1] if self.items else None\n    def is_empty(self): return len(self.items) == 0\n    def size(self): return len(self.items)",
    },
]

# ---- Conversational samples (Fix 2) ----

GREETING_PAIRS = [
    ("Hey!", "Hey! What's up?"),
    ("Hello hello :)", "Hey! It's nice to talk. What's up?"),
    ("Hi there!", "Hi! What can I help you with?"),
    ("Yo!", "Yo! What's happening?"),
    ("Hey, how's your day going?", "Pretty smooth so far. What's happening on your end?"),
    ("Hello!", "Hi! What are you working on?"),
    ("Hey, got a minute?", "Sure — what do you need?"),
    ("Good morning!", "Morning! What's on the agenda?"),
    ("Hi, I'm new here.", "Hey, welcome! What are you building?"),
    ("Hey, can I ask you something?", "Of course — go ahead."),
]

SMALLTALK_PAIRS = [
    ("Just chilling. Thinking about starting a new project.", "Nice. What kind of project are you considering?"),
    ("I'm bored.", "Boredom's a good time to explore something new. What catches your interest?"),
    ("Thanks for the help earlier!", "Anytime. That's what I'm here for."),
    ("I'm learning Python right now.", "Good choice. What are you stuck on?"),
    ("Do you like programming?", "I like helping people with programming. What are you working on?"),
    ("What languages do you know?", "I can help with Python, JavaScript, and a few others. What do you need?"),
    ("I'm having a rough day.", "Sorry to hear that. Want to talk about it, or would a distraction help?"),
    ("You're pretty helpful.", "Thanks — I try. What else can I do for you?"),
    ("I need a break from coding.", "Fair enough. Stepping away often helps. What do you do to unwind?"),
    ("Okay, I think I get it now.", "Great! Want to try writing it yourself, or should I show you another example?"),
]

FOLLOWUP_PAIRS = [
    ("Maybe something with audio processing.", "Fun domain. Creative or more technical?"),
    ("Can you explain that again?", "Sure — which part was unclear?"),
    ("Tell me more about that.", "What specifically would you like to know?"),
    ("Why does that work?", "Good question. It works because the algorithm guarantees each step moves toward the solution. Want the detailed breakdown?"),
    ("I tried your code and it worked!", "Awesome! Did you modify it at all, or use it as-is?"),
    ("What if I want to handle errors too?", "Good thinking. You'd wrap it in a try/except. Want me to show you how?"),
    ("That's cool. Can you make it faster?", "Depends on the use case. How big is your input?"),
    ("I don't understand the last line.", "That line does the final return. Want me to break it down step by step?"),
]

# ---- Persona samples (Fix 4) ----

PERSONA_PAIRS = [
    ("I'm working on a project and need some motivation.", "You've got this. What's the project about?"),
    ("I keep making mistakes and feeling bad about it.", "Mistakes are how we learn. What's going wrong?"),
    ("I feel like I'll never get this.", "That feeling is normal when you're learning something new. What's tripping you up?"),
    ("You're an AI, right? Do you actually understand code?", "I can read, write, and explain code. Whether I 'understand' it is a deeper question. What do you need help with?"),
    ("I want to be a better programmer.", "Practice and curiosity — that's the core. What are you building right now?"),
    ("This is frustrating.", "I hear you. Let's break it down into smaller steps. What's the first thing that's unclear?"),
    ("Can you be my coding buddy?", "I can help you think through problems and review code. What are you working on?"),
    ("I've been coding all night and I'm tired.", "Rest is productive too. Your code will still be here tomorrow. Want to wrap up with a summary of where you are?"),
]


def fmt(decision, verification=None, answer=None, reasoning=None):
    """Build a Heartly-format output string."""
    think = f" thinking {reasoning}  response" if reasoning else " response"
    if decision == "stop":
        return f"{think}<decide>stop</decide>"
    return f"{think}<decide>speak</decide><verify>{verification}</verify> {answer} <stop>"


def gen_instruction_from_task(task, rng):
    fmt_str = rng.choice(INSTRUCTION_FORMATS)
    return fmt_str.format(task=task)


def gen_known_samples(rng, n):
    """Generate known code-answer samples — natural phrasing (Fix 1)."""
    rows = []
    for _ in range(n):
        task = rng.choice(CODE_TASKS_KNOWN)
        instruction = gen_instruction_from_task(task["pattern"], rng)
        code = task["code"]()
        # Natural: just the code block, no "The answer is" stem
        answer = f"```python\n{code}\n```"
        reasoning = rng.choice(REASON_KNOWN_CODE)
        output = fmt("speak", "known", answer, reasoning)
        rows.append({"instruction": instruction, "output": output})
    return rows


def gen_unknown_samples(rng, n):
    """Generate unknown-code samples — one clean refusal (Fix 3)."""
    rows = []
    for _ in range(n):
        task_entry = rng.choice(CODE_TASKS_UNKNOWN)
        query = task_entry["generator"]()
        fmt_str = rng.choice([
            "Write code {query}",
            "How do I {query}?",
            "Implement something {query}",
            "Can you write a function {query}?",
            "I need code {query}",
        ])
        instruction = fmt_str.format(query=query)
        # Fix 3: one canonical refusal, always the same shape
        reasoning = rng.choice(REASON_UNKNOWN_CODE)
        output = fmt("speak", "unknown", REFUSAL_CODE, reasoning)
        rows.append({"instruction": instruction, "output": output})
    return rows


def gen_silence_samples(rng, n):
    """Generate silence samples — only for truly empty/noise inputs."""
    rows = []
    for i in range(n):
        trig = SILENCE_TRIGGERS[i % len(SILENCE_TRIGGERS)]
        output = fmt("stop", reasoning=rng.choice(REASON_SILENCE))
        rows.append({"instruction": trig, "output": output})
    return rows


def gen_completion_samples(rng, n):
    """Generate code completion samples — natural phrasing."""
    rows = []
    for _ in range(n):
        cp = rng.choice(COMPLETION_PROMPTS)
        instruction = rng.choice([
            f"Complete this function:\n```python\n{cp['prefix']}\n    ...",
            f"Finish implementing:\n```python\n{cp['prefix']}\n    pass\n```",
            f"```python\n{cp['prefix']}\n    \"\"\"{cp['doc']}\"\"\"\n    # TODO\n```\n\nWrite the body.",
        ])
        code = cp["code"]()
        answer = f"```python\n{code}\n```"
        reasoning = rng.choice(REASON_KNOWN_CODE)
        output = fmt("speak", "known", answer, reasoning)
        rows.append({"instruction": instruction, "output": output})
    return rows


def gen_conversational_samples(rng, n):
    """Generate conversational samples — greetings, small talk, follow-ups (Fix 2).
    
    These are multi-turn: each sample is a 2-3 turn conversation.
    For SFT we format as instruction/output pairs where instruction = user turns
    joined, output = assistant turns joined with grammar.
    """
    rows = []
    
    # Greetings (single turn)
    greeting_count = n // 3
    for _ in range(greeting_count):
        user_msg, assistant_msg = rng.choice(GREETING_PAIRS)
        reasoning = rng.choice(REASON_SOCIAL)
        output = fmt("speak", "known", assistant_msg, reasoning)
        rows.append({"instruction": user_msg, "output": output})
    
    # Small talk (single turn, but conversational)
    smalltalk_count = n // 3
    for _ in range(smalltalk_count):
        user_msg, assistant_msg = rng.choice(SMALLTALK_PAIRS)
        reasoning = rng.choice(REASON_SOCIAL)
        output = fmt("speak", "known", assistant_msg, reasoning)
        rows.append({"instruction": user_msg, "output": output})
    
    # Follow-ups (2-turn conversations)
    followup_count = n - greeting_count - smalltalk_count
    for _ in range(followup_count):
        # Pick a greeting + follow-up to make a 2-turn conversation
        greeting = rng.choice(GREETING_PAIRS)
        followup = rng.choice(FOLLOWUP_PAIRS)
        
        # Turn 1
        reasoning1 = rng.choice(REASON_SOCIAL)
        output1 = fmt("speak", "known", greeting[1], reasoning1)
        # Turn 2
        reasoning2 = rng.choice(REASON_FOLLOWUP)
        output2 = fmt("speak", "known", followup[1], reasoning2)
        
        # Format as multi-turn: instruction has both user turns, output has both assistant turns
        instruction = f"{greeting[0]}\n{followup[0]}"
        output = f"{output1}\n{output2}"
        rows.append({"instruction": instruction, "output": output})
    
    return rows


def gen_persona_samples(rng, n):
    """Generate persona samples — warm, helpful, grammar-bound (Fix 4)."""
    rows = []
    for _ in range(n):
        user_msg, assistant_msg = rng.choice(PERSONA_PAIRS)
        reasoning = rng.choice(REASON_PERSONA)
        output = fmt("speak", "known", assistant_msg, reasoning)
        rows.append({"instruction": user_msg, "output": output})
    return rows


def load_magicoder_samples(rng, n, max_length=512):
    """Pull samples from Magicoder-Evol-Instruct if available."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", streaming=True)
        count = 0
        for i, example in enumerate(ds):
            if count >= n:
                break
            instruction = example.get("instruction", "")
            response = example.get("response", "")
            if not instruction or not response or len(response) > max_length:
                continue
            if "```" not in response:
                continue
            answer = f"```\n{response}\n```" if not response.startswith("```") else response
            reasoning = rng.choice(REASON_KNOWN_CODE)
            output = fmt("speak", "known", answer, reasoning)
            rows.append({"instruction": instruction, "output": output})
            count += 1
        print(f"[OK] magicoder: {count} samples")
    except Exception as e:
        print(f"[SKIP] magicoder: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=2000, help="Number of known code-task samples")
    ap.add_argument("--unknown", type=int, default=600, help="Number of unknown-code samples")
    ap.add_argument("--silence", type=int, default=100, help="Number of silence samples")
    ap.add_argument("--completion", type=int, default=500, help="Number of code-completion samples")
    ap.add_argument("--conversational", type=int, default=1500, help="Number of conversational samples (Fix 2)")
    ap.add_argument("--persona", type=int, default=500, help="Number of persona samples (Fix 4)")
    ap.add_argument("--magicoder", type=int, default=2000, help="Number of Magicoder samples to pull")
    ap.add_argument("--out", default="sft_dataset_code_v3.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    # Known instruction-following code tasks (Fix 1: natural phrasing)
    known_rows = gen_known_samples(rng, args.known)
    rows.extend(known_rows)
    print(f"known (instruction): {len(known_rows)}")

    # Code completion
    comp_rows = gen_completion_samples(rng, args.completion)
    rows.extend(comp_rows)
    print(f"completion: {len(comp_rows)}")

    # Unknown code tasks (Fix 3: one clean refusal)
    unknown_rows = gen_unknown_samples(rng, args.unknown)
    rows.extend(unknown_rows)
    print(f"unknown: {len(unknown_rows)}")

    # Silence (only truly empty/noise — greetings moved to conversational)
    silence_rows = gen_silence_samples(rng, args.silence)
    rows.extend(silence_rows)
    print(f"silence: {len(silence_rows)}")

    # Conversational samples (Fix 2: greetings, small talk, follow-ups)
    conv_rows = gen_conversational_samples(rng, args.conversational)
    rows.extend(conv_rows)
    print(f"conversational: {len(conv_rows)}")

    # Persona samples (Fix 4: warm, helpful)
    persona_rows = gen_persona_samples(rng, args.persona)
    rows.extend(persona_rows)
    print(f"persona: {len(persona_rows)}")

    # Magicoder samples (if available)
    if args.magicoder > 0:
        magicoder_rows = load_magicoder_samples(rng, args.magicoder)
        rows.extend(magicoder_rows)
        print(f"magicoder: {len(magicoder_rows)}")

    # Shuffle and write
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    n_known = sum(1 for r in rows if "<verify>known</verify>" in r["output"])
    n_unknown = sum(1 for r in rows if "<verify>unknown</verify>" in r["output"])
    n_stop = sum(1 for r in rows if "<decide>stop</decide>" in r["output"])
    n_stem = sum(1 for r in rows if "The answer is" in r["output"])
    print(f"\nWrote {len(rows)} SFT samples -> {args.out}")
    print(f"  known: {n_known}  |  unknown: {n_unknown}  |  silence: {n_stop}")
    print(f"  'The answer is' stems: {n_stem} (should be 0)")
    for r in rows[:3] + rows[-2:]:
        print(f"\n--- {r['instruction'][:80]!r}\n{r['output'][:200]}")


if __name__ == "__main__":
    main()