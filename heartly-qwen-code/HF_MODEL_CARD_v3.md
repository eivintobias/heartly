---
language:
  - en
license: mit
base_model: Qwen/Qwen2.5-Coder-1.5B
library_name: transformers
model_type: qwen2
tags:
  - code
  - hallucination-reduction
  - heartly
  - decide-verify-stop
  - boundary-head
  - pytorch
  - text-generation
---

# Heartly Qwen-Code v3

A 1.5B coding LLM with the **Heartly** hallucination-reduction architecture,
fine-tuned from **Qwen2.5-Coder-1.5B** with the conversational Stage-5 SFT recipe
(Fix1–4: natural phrasing, single refusal, persona — 5,200 samples in
`heartly-qwen-code/sft_dataset_code_v3.jsonl`).

v3 builds on the same v1/v2 Stage 1–4 numbers (grammar adoption 100%, boundary-head
AUROC 1.000, critic AUROC 1.000) — same Qwen2.5-Coder-1.5B base, now trained for
multi-turn conversational code chat. See [`HF_MODEL_CARD.md`](HF_MODEL_CARD.md) for
the Stage 1–2 probe/critic results carried over from the identical architecture.

## Output grammar

```
 thinking [reasoning]  response<decide>speak|stop</decide><verify>known|unknown</verify> [answer] <stop>
```

Only `[answer]` should reach the user.

## Usage

### 0. Recommended — chat via the GitHub server (strips the grammar for you)

This model emits the Heartly grammar as ordinary multi-token text (the tags are
**not** tokenizer special tokens), so some front-ends (e.g. LM Studio) may decode
them mangled. The **`server.py`** FastAPI loader on GitHub loads this model and
runs every reply through **`reply_formatter.py`**, which canonicalises the tags
and returns only the clean answer.

```bash
pip install -r heartly-qwen-code/requirements.txt   # includes fastapi + uvicorn
python heartly-qwen-code/server.py --model heartly-qwen-code-v3 --port 8000

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a function that reverses a string"}'
```

Response: `{"model":"heartly-qwen-code-v3","raw":"...<decide>...","reply":"<clean answer>"}`.

Quick offline test (no server): `python heartly-qwen-code/chat_smoke.py "Write a function that sorts a list"`.

📦 **Model card source:** this file (`HF_MODEL_CARD_v3.md`). When uploaded to
HuggingFace, copy it to `README.md` on the hub repo.

### 1. Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("eivintobias/heartly-qwen-code-v3")
model = AutoModelForCausalLM.from_pretrained(
    "eivintobias/heartly-qwen-code-v3", torch_dtype=torch.float32, device_map="cpu"
)
model.eval()
ids = tok.encode("User: Write a function that reverses a string\nAssistant: ", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=256, pad_token_id=tok.eos_token_id, do_sample=False)
raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
# Strip the grammar -> clean answer:
from reply_formatter import format_reply
print(format_reply(raw))
```

## Files in this repo

| File | Description |
|------|-------------|
| `config.json` | Qwen2ForCausalLM (28 layers, d=1536) + `heartly_stop_token_id=9495` |
| `generation_config.json` | default generate params |
| `chat_template.jinja` | standard Qwen chat template |
| `tokenizer.json` / `tokenizer_config.json` | Qwen BPE tokenizer |
| `model.safetensors` | v3 fine-tuned weights (**full fine-tune**, not a LoRA adapter) |

## Training

- **Base:** Qwen/Qwen2.5-Coder-1.5B
- **Method:** full fine-tune (fp16), max-length 512, 2 epochs, freeze bottom 12 layers
- **Dataset:** `sft_dataset_code_v3.jsonl` (5,200 conversational Heartly samples)
- **GPU:** 1× RTX 3090 (24GB)

## License

MIT — built on Qwen2.5-Coder (Apache 2.0).

## Links

- [GitHub (server + tools)](https://github.com/eivintobias/heartly/tree/master/heartly-qwen-code)
- [v1/v2 model card (LoRA)](HF_MODEL_CARD.md)
- [Heartly RWKV7 model](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b)
