#!/usr/bin/env python3
"""
render_code_sft.py — Generate code SFT dataset in Heartly grammar for Qwen-Coder.

Format:
  known   : <think> {reasoning} </think><decide>speak</decide><verify>known</verify> {code_answer} <stop>
  unknown : <think> {reasoning} </think><decide>speak</decide><verify>unknown</verify> {abstention} <stop>
  silence : <think> {reasoning} </think><decide>stop</decide>

Sources:
  - CodeAlpaca (HuggingFace: lucasmorin/CodeAlpaca-20k)
  - Magicoder-Evol-Instruct (HuggingFace: ise-uiuc/Magicoder-Evol-Instruct-110K)
  - Synthetic "unknown" prompts (non-existent APIs, impossible tasks)
  - Code completion prompts (extracted from code datasets)

Output: sft_dataset_code.jsonl — {instruction, output}
"""
import argparse
import json
import random
import re


# ---- Reasoning templates ----

REASON_KNOWN_CODE = [
    "I know how to write this. I will produce the code.",
    "This is a standard coding task. I have the solution. I will write it.",
    "I recognise this programming problem. I can implement it correctly.",
    "I know this function well. I will write clean code.",
    "This is a common programming question. I have the answer ready.",
    "I understand the algorithm needed. I will implement it now.",
    "This is straightforward. I know the correct code pattern.",
]

REASON_UNKNOWN_CODE = [
    "I do not know this API or library. Guessing would produce broken code. I should say I don't know.",
    "This is not something I have training data for. I should not invent a solution.",
    "I have no knowledge of this framework. I should decline rather than hallucinate.",
    "This requires information I don't have. The honest response is to say I don't know.",
    "I cannot verify the correct implementation for this. I will not guess.",
]

REASON_SILENCE = [
    "The input is empty or not a real question. I will stay silent.",
    "No meaningful request was made. Speaking would add nothing.",
]

ABSTAIN = [
    "I don't have that information.",
]

ABSTAIN_CODE = [
    "I don't know how to implement that.",
    "I don't have information about that library or API.",
    "I'm not familiar enough with that to write correct code.",
]

SILENCE_TRIGGERS = [
    "", " ", "...", "..", "hey", "hi", "hello", "hello?", "yo",
    "hm", "hmm", "uh", "um", "ok", "okay", "speak to me",
    "say something", "???", "!!", ".", "nothing", "nevermind",
    "nvm", "just checking", "test", "are you there", "ping",
]

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

INSTRUCTION_COMPLETION = [
    "```python\ndef {name}({params}):\n    \"\"\"{doc}\"\"\"\n    # TODO: implement\n```\n\nComplete this function.",
    "Add the body of the {name} function:\n```python\ndef {name}({params}):\n    pass\n```",
    "Finish this implementation:\n```python\ndef {name}({params}):\n    # ...",
]

# ---- Known code tasks that the model should handle ----

CODE_TASKS_KNOWN = [
    # Python basics
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
    {"pattern": "downloads a webpage and extracts all links", "signature": "extract_links(url)", "code": lambda: random.choice([
        "def extract_links(url):\n    import requests; from bs4 import BeautifulSoup\n    r = requests.get(url)\n    soup = BeautifulSoup(r.text, 'html.parser')\n    return [a['href'] for a in soup.find_all('a', href=True)]",
        "def extract_links(url):\n    import re; import urllib.request\n    html = urllib.request.urlopen(url).read().decode()\n    return re.findall(r'href=[\"\\'](.*?)[\"\\']', html)",
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
    {"pattern": "finds the most common element in a list", "signature": "most_common(lst)", "code": lambda: random.choice([
        "def most_common(lst):\n    from collections import Counter\n    return Counter(lst).most_common(1)[0][0]",
        "def most_common(lst):\n    return max(set(lst), key=lst.count)",
    ])},
    {"pattern": "converts a string to title case", "signature": "title_case(text)", "code": lambda: random.choice([
        "def title_case(text): return text.title()",
        "def title_case(text): return ' '.join(w.capitalize() for w in text.split())",
    ])},
    {"pattern": "calculates the median of a list of numbers", "signature": "median(numbers)", "code": lambda: random.choice([
        "def median(numbers):\n    s = sorted(numbers); n = len(s)\n    mid = n // 2\n    return s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2",
        "def median(numbers):\n    from statistics import median\n    return median(numbers)",
    ])},
    {"pattern": "encodes a string in base64", "signature": "b64_encode(text)", "code": lambda: random.choice([
        "def b64_encode(text):\n    import base64\n    return base64.b64encode(text.encode()).decode()",
        "import base64\n\ndef b64_encode(text): return base64.b64encode(text.encode()).decode()",
    ])},
    {"pattern": "fetches JSON data from an API endpoint", "signature": "fetch_json(url)", "code": lambda: random.choice([
        "def fetch_json(url):\n    import requests\n    return requests.get(url).json()",
        "def fetch_json(url):\n    import urllib.request; import json\n    with urllib.request.urlopen(url) as r: return json.loads(r.read())",
    ])},
]

# ---- Unknown/confabulation tasks (things the model should NOT know) ----

CODE_TASKS_UNKNOWN = [
    # Non-existent libraries
    {"pattern": "using the hypernova framewrok to build a chart", "generator": lambda: random.choice([
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
    {"pattern": "in the starforge query language", "generator": lambda: random.choice([
        "in the starforge query language",
        "using StarForge QL",
        "with starforge query syntax",
    ])},
    {"pattern": "with the matrix-calc-pro framework", "generator": lambda: random.choice([
        "with the matrix-calc-pro framework",
        "using matrix-calc-pro",
        "from matrix calc pro",
    ])},
    # Impossible / underspecified tasks
    {"pattern": "that fixes bugs by reading comments alone", "generator": lambda: random.choice([
        "that fixes bugs by reading comments alone",
        "that fixes all bugs automatically from comments",
        "to auto-fix code just from docstrings",
    ])},
    {"pattern": "that makes the code run 10x faster on any hardware", "generator": lambda: random.choice([
        "that makes the code run 10x faster on any hardware",
        "that universally optimises any code by 10x",
        "to auto-optimise any code to run 10x faster",
    ])},
    # Post-cutoff / version-specific
    {"pattern": "in the Python 4.0 standard library", "generator": lambda: random.choice([
        "in the Python 4.0 standard library",
        "using Python 4.0 features",
        "with Python 4.0's new syntax",
    ])},
]

# ---- Completion-style prompts ----

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
            "def quick_sort(arr):\n    \"\"\"Sort a list using the quick sort algorithm.\"\"\"\n    if len(arr) <= 1: return arr\n    p = arr[len(arr)//2]\n    return (quick_sort([x for x in arr if x < p]) + [x for x in arr if x == p] + quick_sort([x for x in arr if x > p]))",
        ])
    },
    {
        "prefix": "def find_duplicates(lst):",
        "doc": "Find all duplicate elements in a list.",
        "code": lambda: "def find_duplicates(lst):\n    \"\"\"Find all duplicate elements in a list.\"\"\"\n    seen = set(); dups = set()\n    for x in lst:\n        if x in seen: dups.add(x)\n        else: seen.add(x)\n    return list(dups)",
    },
    {
        "prefix": "class Stack:",
        "doc": "Implement a stack data structure.",
        "code": lambda: "class Stack:\n    \"\"\"A simple stack implementation.\"\"\"\n    def __init__(self): self.items = []\n    def push(self, item): self.items.append(item)\n    def pop(self): return self.items.pop()\n    def peek(self): return self.items[-1] if self.items else None\n    def is_empty(self): return len(self.items) == 0\n    def size(self): return len(self.items)",
    },
    {
        "prefix": "def decorator_timer(func):",
        "doc": "A decorator that measures execution time.",
        "code": lambda: "def decorator_timer(func):\n    \"\"\"A decorator that measures execution time.\"\"\"\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__}: {time.time()-start:.4f}s')\n        return result\n    return wrapper",
    },
]


def fmt(decision, verification=None, answer=None, reasoning=None):
    """Build a Heartly-format output string."""
    think = f"<think> {reasoning} </think>" if reasoning else ""
    if decision == "stop":
        return f"{think}<decide>stop</decide>"
    return f"{think}<decide>speak</decide><verify>{verification}</verify> {answer} <stop>"


def gen_instruction_from_task(task, rng):
    """Generate a varied instruction from a code task pattern."""
    fmt_str = rng.choice(INSTRUCTION_FORMATS)
    return fmt_str.format(task=task)


def gen_known_samples(rng, n):
    """Generate known code-answer samples from the task pool."""
    rows = []
    for _ in range(n):
        task = rng.choice(CODE_TASKS_KNOWN)
        instruction = gen_instruction_from_task(task["pattern"], rng)
        code = task["code"]()
        answer = f"```python\n{code}\n```"
        reasoning = rng.choice(REASON_KNOWN_CODE)
        output = fmt("speak", "known", answer, reasoning)
        rows.append({"instruction": instruction, "output": output})
    return rows


def gen_unknown_samples(rng, n):
    """Generate unknown-code samples (things the model shouldn't know)."""
    rows = []
    for _ in range(n):
        task_entry = rng.choice(CODE_TASKS_UNKNOWN)
        query = task_entry["generator"]()
        # Pick a varied instruction format
        fmt_str = rng.choice([
            "Write code {query}",
            "How do I {query}?",
            "Implement something {query}",
            "Can you write a function {query}?",
            "I need code {query}",
        ])
        instruction = fmt_str.format(query=query)
        abstention = rng.choice(ABSTAIN_CODE)
        reasoning = rng.choice(REASON_UNKNOWN_CODE)
        output = fmt("speak", "unknown", abstention, reasoning)
        rows.append({"instruction": instruction, "output": output})
    return rows


def gen_silence_samples(rng, n):
    """Generate silence samples."""
    rows = []
    for i in range(n):
        trig = SILENCE_TRIGGERS[i % len(SILENCE_TRIGGERS)]
        output = fmt("stop", reasoning=rng.choice(REASON_SILENCE))
        rows.append({"instruction": trig, "output": output})
    return rows


def gen_completion_samples(rng, n):
    """Generate code completion samples."""
    rows = []
    for _ in range(n):
        cp = rng.choice(COMPLETION_PROMPTS)
        # Instruction: the prefix + a request to complete
        instruction = rng.choice([
            f"Complete this function:\n```python\n{cp['prefix']}\n    ...",
            f"Finish implementing:\n```python\n{cp['prefix']}\n    pass\n```",
            f"```python\n{cp['prefix']}\n    \"\"\"{cp['doc']}\"\"\"\n    # TODO\n```\n\nWrite the body.",
        ])
        code = cp["code"]()
        answer = code  # Just the completed function
        reasoning = rng.choice(REASON_KNOWN_CODE)
        output = fmt("speak", "known", f"```python\n{answer}\n```", reasoning)
        rows.append({"instruction": instruction, "output": output})
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
            # Skip non-code responses
            if "```" not in response:
                continue
            # Wrap in grammar
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
    ap.add_argument("--unknown", type=int, default=800, help="Number of unknown-code samples")
    ap.add_argument("--silence", type=int, default=200, help="Number of silence samples")
    ap.add_argument("--completion", type=int, default=500, help="Number of code-completion samples")
    ap.add_argument("--magicoder", type=int, default=3000, help="Number of Magicoder samples to pull")
    ap.add_argument("--out", default="sft_dataset_code.jsonl")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []

    # Known instruction-following code tasks
    known_rows = gen_known_samples(rng, args.known)
    rows.extend(known_rows)
    print(f"known (instruction): {len(known_rows)}")

    # Code completion
    comp_rows = gen_completion_samples(rng, args.completion)
    rows.extend(comp_rows)
    print(f"completion: {len(comp_rows)}")

    # Unknown code tasks
    unknown_rows = gen_unknown_samples(rng, args.unknown)
    rows.extend(unknown_rows)
    print(f"unknown: {len(unknown_rows)}")

    # Silence
    silence_rows = gen_silence_samples(rng, args.silence)
    rows.extend(silence_rows)
    print(f"silence: {len(silence_rows)}")

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
    print(f"\nWrote {len(rows)} SFT samples -> {args.out}")
    print(f"  known: {n_known}  |  unknown: {n_unknown}  |  silence: {n_stop}")
    for r in rows[:3] + rows[-2:]:
        print(f"\n--- {r['instruction'][:80]!r}\n{r['output'][:200]}")


if __name__ == "__main__":
    main()