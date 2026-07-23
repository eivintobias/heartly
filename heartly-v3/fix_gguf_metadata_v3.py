"""Rewrite Heartly-v3 GGUF metadata: EOS -> <stop> (stop token ID), Heartly chat template."""
import subprocess
import sys
import os

SCRIPTS = r"c:\Users\eivin\Desktop\Datasets organizer\llama.cpp\gguf-py\gguf\scripts\gguf_new_metadata.py"
SRC_GGUF = r"c:\Users\eivin\Desktop\Datasets organizer\heartly-v3\heartly-v3-qwen2.5-0.5b-f16.gguf"
DST_GGUF = r"c:\Users\eivin\Desktop\Datasets organizer\heartly-v3\heartly-v3-qwen2.5-0.5b-f16-lmstudio.gguf"

# Heartly chat template: simple User/Assistant format
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

# The <stop> token was the last token added to the vocabulary.
# For Qwen2.5-0.5B base: vocab_size = 151936, after adding special tokens: 151936 + 7 = 151943
# But config.json says vocab_size = 151672 which seems to use qwen2.5-0.5b with a different tokenizer version
# Let's compute: after adding 7 special tokens to the tokenizer we can check.
# The base Qwen/Qwen2.5-0.5B tokenizer has vocab_size = 151936
# 151936 + 7 = 151943. But config says 151672...
# Actually the config.json says vocab_size = 151672, which is from the model checkpoint file.
# After resize_token_embeddings, the model's vocab was changed.
# Let's find the stop token dynamically - we'll check what it is after rebuild.

# For now, placeholder - will be determined after running rebuild_tokenizer_v3.py
# Typically <stop> is the 7th added special token = 151936 + 6 = 151942 (0-indexed)
# But we'll verify after step 1.

env = dict(os.environ)
env["PYTHONPATH"] = r"c:\Users\eivin\Desktop\Datasets organizer\llama.cpp\gguf-py"

# We'll fill in the correct stop_id after step 1
STOP_TOKEN_ID = "151671"  # <stop> token ID confirmed from tokenizer rebuild

cmd = [
    sys.executable, SCRIPTS, SRC_GGUF, DST_GGUF,
    "--special-token-by-id", "eos", STOP_TOKEN_ID,
    "--chat-template", CHAT_TEMPLATE,
    "--force",
]
print("Running:", " ".join(cmd[:4]), "...")
r = subprocess.run(cmd, env=env)
sys.exit(r.returncode)