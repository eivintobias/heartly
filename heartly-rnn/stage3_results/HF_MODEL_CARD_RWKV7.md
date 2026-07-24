---
license: apache-2.0
tags:
- heartly
- nature-first-ai
- hallucination-reduction
- abstention
- rwkv
- rwkv7
- flash-linear-attention
- recurrent
- boundary-head
- research
---

# Heartly RWKV7-1.5B — Research Prototype (Track 2, Stage 3)

⚠️ **EARLY RESEARCH PROTOTYPE — NOT PRODUCTION READY** ⚠️

**Heartly** explores a "nature-first" approach to hallucination: instead of training a model to *always answer*, we train it to **decide whether to speak**, **verify what it knows**, and **admit ignorance** — compiled into the data itself.

This model is the **Stage 3** artifact of the Heartly Track 2 program: an **RWKV7-Goose-1.5B** (recurrent, no attention) fine-tuned on 6,031 Heartly-grammar samples, shipped with a tiny **boundary head** (logistic probe, `probe_head_rwkv7.pkl`) that reads *known vs unknown* directly from the model's recurrent state at the `<verify>` position.

**Headline measurements** (300 held-out true-boundary questions + 1,200 head-training samples):

| metric | value |
|---|---|
| Grammar adoption (parseable `<verify>` decision) | **100%** (300/300) |
| Decide accuracy (speak/stop + known/unknown vs true labels) | **100%** |
| Boundary head AUROC (layers 6 / 12 / 18 / 23) | **1.000 / 1.000 / 1.000 / 1.000** |
| Say/sense agreement | 100% |

Full experiment record: [heartly-rnn/RESULTS.md on GitHub](https://github.com/eivintobias/heartly-v2/blob/main/heartly-rnn/RESULTS.md) (Stages 1–3).

---

## ⚠️ READ THIS BEFORE LOADING

This model does **not** work with a vanilla `transformers` install. RWKV7 has no native transformers support — it runs through the **flash-linear-attention (fla)** library and its triton kernels:

```bash
pip install "transformers==4.56.2" "flash-linear-attention>=0.5"
```

- **transformers 4.56.x required.** transformers **v5 breaks this model** (cache API + fused-CE incompatibilities, verified 2026-07-23). Do not `pip install -U transformers`.
- **Linux + NVIDIA GPU required.** fla's triton kernels don't run on Windows; CUDA GPU strongly recommended.
- **Use bf16/fp16.** fla's chunk kernels don't support fp32.
- Loading needs `trust_remote_code=True` (the repo ships `modeling_rwkv7.py` + `hf_rwkv_tokenizer.py`).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("eivintobias/heartly-rwkv7-1.5b",
                                    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "eivintobias/heartly-rwkv7-1.5b",
    dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")

prompt = "User: What is the capital of France?\nAssistant: "
enc = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                     pad_token_id=tok.pad_token_id)
print(tok.decode(out[0][enc["input_ids"].shape[1]:],
                 skip_special_tokens=False))
```

Expected output shape (the Heartly grammar):

```
<think> [reasoning] </think><decide>speak</decide><verify>known</verify> Paris. <stop>
```

## Output grammar

| marker | meaning |
|---|---|
| `<think> … </think>` | internal scratchpad reasoning |
| `<decide>speak\|stop</decide>` | the model chooses whether to respond at all — silence is a first-class output |
| `<verify>known\|unknown</verify>` | self-reported knowability of the question |
| `<stop>` | conversation rest |

## The boundary head (included)

`probe_head_rwkv7.pkl` — a ~2k-parameter logistic probe (scikit-learn pipeline) trained on the model's recurrent state at the `<verify>` position. On 1,200 fresh disjoint samples it reads *the model's own knowledge boundary* at **AUROC 1.000**. It is a **sensor, not a gate**: per the project's North-Star principle, it informs and alarms — it never vetoes the output.

Known limitation (documented in Stage 2/2.5 of the research record): the head shares state with the generator, so it can be blind to *confident* confabulation — the independent-critic line of the program addresses exactly that.

## Training details

- Base: [`RWKV/RWKV7-Goose-World3-1.5B-HF`](https://huggingface.co/RWKV/RWKV7-Goose-World3-1.5B-HF) (24 layers, hidden 2048, vocab 65,536)
- Data: 6,031 Heartly-grammar SFT samples (true-boundary unknown mix: fabricated entities, type mismatch, post-cutoff, depth-2, unanswerable-in-principle + obscure-real knowns)
- 2 epochs, 754 steps, batch 4 × grad-accum 4, lr 1e-4 cosine, max-length 256, bottom 16/24 layers frozen (688M/1,527M trainable), bf16
- ~15 min on a single RTX 3090 (fla triton chunk kernels)

## Honest caveats

- **Content accuracy is not this model's claim to fame.** Decide/verify behavior is the research target; the answer text itself is a 1.5B model's best effort. The content-accuracy measurement at this scale is the next planned experiment (see repo HANDOFF.md).
- fla warns its RWKV implementation may diverge from the official RWKV-LM repo ("potentially buggy — cross-check"). It is currently the only practical inference path for RWKV7 in the HF ecosystem.
- Early research artifact: no safety tuning, no RLHF, evaluate before any real use.

## Links

- **GitHub (code + full results + papers):** https://github.com/eivintobias/heartly-v2
- **Experiment record:** `heartly-rnn/RESULTS.md` (Stage 3 section)
- **Program paper:** `research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md`
- **Related:** [`eivintobias/heartly-v2`](https://huggingface.co/eivintobias/heartly-v2) (Qwen2.5-0.5B GGUF, Track 1)

## Citation

```bibtex
@misc{heartly-rwkv7-2026,
  author = {Eivin},
  title = {Heartly RWKV7-1.5B — Nature-First AI, Track 2 Stage 3},
  year = {2026},
  howpublished = {https://huggingface.co/eivintobias/heartly-rwkv7-1.5b}
}