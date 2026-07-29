---
language:
  - en
library_name: peft
base_model: Qwen/Qwen2.5-Coder-1.5B
tags:
  - code
  - hallucination-reduction
  - heartly
  - decide-verify-stop
  - boundary-head
  - critic
license: mit
---

# Heartly Qwen-Code

A coding LLM with the **Heartly hallucination-reduction architecture**, built on Qwen2.5-Coder-1.5B.

## What is Heartly?

Heartly is a "nature-first" approach to hallucination reduction. Instead of forcing the model to be honest, it trains the model to **know when it doesn't know** — and then leaves the choice to the model. The model gains knowledge of its own ignorance and remains free to choose.

The output grammar:
```
 thinking [reasoning]  response<decide>speak|stop</decide><verify>known|unknown</verify> [answer] <stop>
```

## What this model does

- **Decides** whether it can answer a coding question (`<decide>speak|stop</decide>`)
- **Verifies** whether it actually knows the answer (`<verify>known|unknown</verify>`)
- **Admits ignorance** when asked about non-existent APIs, libraries, or impossible tasks
- **Writes correct code** when it knows the answer

## Results

### Stage 1 — Grammar + Boundary Head

| Metric | Result |
|--------|--------|
| Grammar adoption | **100%** (35/35) |
| Verify accuracy | **100%** (35/35) |
| Boundary head AUROC | **1.000** |
| Confabulations | **0** |

### Stage 2 — Code Critic

| Metric | Result |
|--------|--------|
| Critic AUROC | **1.000** |
| Confab recall | **100%** |
| Correct precision | **100%** |

The critic reads the model's hidden state at the `<verify>` position and can distinguish "confidently correct" code from "confidently fake" code with perfect separation.

## Training Details

- **Base model:** Qwen/Qwen2.5-Coder-1.5B
- **Method:** QLoRA (r=16, 4-bit NF4)
- **Dataset:** 6,500 code SFT samples in Heartly grammar
  - 2,000 known code tasks (Python)
  - 500 code completions
  - 800 unknown tasks (non-existent APIs)
  - 200 silence triggers
  - 3,000 Magicoder-Evol-Instruct samples
- **Training:** 2 epochs, batch 2, grad-accum 8, max-length 512, fp16
- **GPU:** RTX 2070 Max-Q (8GB) — local laptop

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model + LoRA adapter
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "eivintobias/heartly-qwen-code")

# Ask a coding question
prompt = "User: Write a function that sorts a list\nAssistant: "
inputs = tok.encode(prompt, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(output[0]))
```

## Files

| File | Description |
|------|-------------|
| `adapter_config.json` | LoRA adapter config |
| `adapter_model.safetensors` | LoRA weights |
| `probe_head.pkl` | Boundary head (logistic regression on hidden states) |
| `critic_head.pkl` | Code critic (detects confident confabulations) |
| `sft_dataset_code.jsonl` | Training data (6,500 samples) |
| `critic_data_code.jsonl` | Critic training data (40 samples) |

## License

MIT

## Links

- [GitHub](https://github.com/eivintobias/heartly)
- [Heartly RWKV7 model](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b)
- [Research paper](https://github.com/eivintobias/heartly/blob/main/research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md)