"""Verify Heartly-v3 GGUF metadata."""
import sys
sys.path.insert(0, r"c:\Users\eivin\Desktop\Datasets organizer\llama.cpp\gguf-py")
from gguf import GGUFReader

path = r"c:\Users\eivin\Desktop\Datasets organizer\heartly-v3\heartly-v3-qwen2.5-0.5b-f16-lmstudio.gguf"
reader = GGUFReader(path)

print("=== Special Token IDs ===")
for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.padding_token_id"]:
    if key in reader.fields:
        val = reader.fields[key]
        data = val.parts[0]
        if hasattr(data, "tolist"):
            print(f"  {key}: {data.tolist()}")
        else:
            print(f"  {key}: {data}")

print("\n=== Chat Template ===")
if "tokenizer.chat_template" in reader.fields:
    val = reader.fields["tokenizer.chat_template"]
    data = val.parts[0]
    if hasattr(data, "tobytes"):
        ct = data.tobytes().decode("utf-8", errors="replace")
        print(f"  {ct[:300]}")
    else:
        print(f"  {data}")
else:
    print("  NOT FOUND")

print("\n=== Architecture ===")
for key in ["general.architecture", "general.name", "general.file_type", "qwen2.block_count"]:
    if key in reader.fields:
        val = reader.fields[key]
        data = val.parts[0]
        if hasattr(data, "tobytes"):
            print(f"  {key}: {data.tobytes().decode('utf-8', errors='replace')}")
        else:
            print(f"  {key}: {data}")

print("\nDone.")