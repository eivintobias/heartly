#!/usr/bin/env python3
"""Quick non-interactive test of heartly-qwen-code-v3 GGUF."""
import sys
sys.path.insert(0, '.')
from chat_clean import strip_grammar

try:
    from llama_cpp import Llama
except ImportError:
    print("Installing llama-cpp-python...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
    from llama_cpp import Llama

model_path = "heartly-qwen-code-v3.gguf"
print(f"Loading {model_path}...")
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)

tests = [
    "Write a function that reverses a string",
    "Hey, how's it going?",
    "Write code using the hypernova framework to build a chart",
    "What's the code for finding the factorial of a number?",
]

for prompt in tests:
    full_prompt = f"User: {prompt}\nAssistant: "
    output = llm(full_prompt, max_tokens=256, temperature=0.7, stop=["<stop>", "User:"], echo=False)
    raw = output["choices"][0]["text"].strip()
    clean = strip_grammar(raw)
    print(f"\n{'='*60}")
    print(f"Q: {prompt}")
    print(f"RAW: {raw[:300]}")
    print(f"CLEAN: {clean}")
    # Check grammar adoption
    has_decide = "<decide>" in raw
    has_verify = "<verify>" in raw
    has_stop = "<stop" in raw
    print(f"GRAMMAR: decide={has_decide} verify={has_verify} stop={has_stop}")

print("\nDone.")