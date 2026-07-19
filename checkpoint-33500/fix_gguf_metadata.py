"""Rewrite Heartly GGUF metadata: EOS -> <stop> (151671), Heartly chat template."""
import subprocess
import sys
import os

SCRIPTS = r"c:\Users\eivin\Desktop\Datasets organizer\llama.cpp\gguf-py\gguf\scripts\gguf_new_metadata.py"
SRC = r"c:\Users\eivin\Desktop\Datasets organizer\checkpoint-33500\heartly-v2-qwen2.5-0.5b-f16.gguf"
DST = r"c:\Users\eivin\Desktop\Datasets organizer\checkpoint-33500\heartly-v2-qwen2.5-0.5b-f16-lmstudio.gguf"

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant: {% endif %}"
)

env = dict(os.environ)
env["PYTHONPATH"] = r"c:\Users\eivin\Desktop\Datasets organizer\llama.cpp\gguf-py"

cmd = [
    sys.executable, SCRIPTS, SRC, DST,
    "--special-token-by-id", "eos", "151671",
    "--chat-template", CHAT_TEMPLATE,
    "--force",
]
print("Running:", " ".join(cmd[:4]), "...")
r = subprocess.run(cmd, env=env)
sys.exit(r.returncode)
