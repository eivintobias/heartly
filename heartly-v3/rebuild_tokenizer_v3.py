"""Heartly-v3: rebuild tokenizer with special tokens for checkpoint-9000."""
from transformers import AutoTokenizer

# Heartly v3 grammar: reason-then-decide (7 special tokens)
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<think>", "</think>",
        "<decide>", "</decide>",
        "<verify>", "</verify>",
        "<stop>"
    ]
}

print("Loading base tokenizer: Qwen/Qwen2.5-0.5B ...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print(f"Base vocab size: {len(tok)}")

num_added = tok.add_special_tokens(SPECIAL_TOKENS)
print(f"Added {num_added} special tokens. New vocab length: {len(tok)}")

# Set pad_token to eos_token
tok.pad_token = tok.eos_token

# Find <stop> token ID
stop_id = tok.convert_tokens_to_ids("<stop>")
print(f"<stop> token ID: {stop_id}")

# Save tokenizer to the checkpoint directory
out_dir = r"C:\Users\eivin\Desktop\Datasets organizer\heartly-v3\checkpoint-9000"
tok.save_pretrained(out_dir)
print(f"Tokenizer saved to {out_dir}")
print(f"Files: tokenizer_config.json, tokenizer.json, special_tokens_map.json, added_tokens.json")