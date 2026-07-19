from transformers import AutoTokenizer

# Heartly v2 grammar: reason-then-decide (7 special tokens)
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
num_added = tok.add_special_tokens(SPECIAL_TOKENS)
print(f"Added {num_added} special tokens. New vocab length: {len(tok)}")
tok.pad_token = tok.eos_token

out_dir = r"c:\Users\eivin\Desktop\Datasets organizer\checkpoint-33500\checkpoint-33500"
tok.save_pretrained(out_dir)
print(f"Tokenizer saved to {out_dir}")
