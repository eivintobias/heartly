---
language: en
license: apache-2.0
library_name: transformers
tags:
- heartly
- nature-first-ai
- hallucination-reduction
- abstention
- memory
- write-gate
- retrieval
- research
- rwkv7
- fine-tuned
base_model: RWKV/RWKV7-Goose-World3-1.5B-HF
pipeline_tag: text-generation
---

# Heartly RWKV7-1.5B v2 — Memory-Aware Model (Stage 4c)

**⚠️ RESEARCH PROTOTYPE — NOT PRODUCTION READY ⚠️**

This is the latest Heartly model: RWKV7-Goose-1.5B fine-tuned with the Heartly decide/verify/stop grammar **plus memory channels** (write-gate + retrieval store). It is the product of Stage 4c (2026-07-25), which passed all pre-registered bars.

## What's new vs [heartly-rwkv7-1.5b](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b) (Stage 3)

| Feature | Stage 3 model | This model (Stage 4c) |
|---------|--------------|----------------------|
| Grammar adoption | 100% | 100% (no regression) |
| Decide accuracy | 100% | 100% |
| Boundary head AUROC | 1.000 | 1.000 |
| Say/sense agreement | 1.0 | 1.0 |
| Write-gate (7 formats) | — | **5/5** (was 0/5 for declarations in 4b) |
| Retrieval | — | **5/5** |
| Context injection (I1) | — | **4/5** |
| Context injection (I2) | — | **4/5** |
| Training samples | 6,031 | 7,531 (+1,500 memory-aware) |
| Training steps | 754 | 942 |

The key breakthrough: **one training class (context-known) flipped both memory channels open.** The disposition block — the model's refusal to answer personal-context questions — was a single learned gap. Adding "answer from provided context" examples to the SFT mix taught the model to read injected context AND opened the write-gate for declarations.

## Output Format

```
 thinking [internal reasoning]  response
<decide>speak|stop</decide>
<verify>known|unknown</verify>
[answer if known]
<stop>
```

### Memory Channels

**Write-gate:** During conversation, the model writes facts into its recurrent state using multiple redundant formats (declaration, note, reminder, summary, etc.). Writing the same fact several ways gives 5/5 recall; single formats reach 0–2/5.

**Retrieval store:** An external embedding store retrieves relevant context and injects it into the prompt. The model reads injected context and answers from it at 4/5 accuracy — but only because the SFT data includes a "context-known" training class.

Key finding: **a model can only use a memory channel it was trained to read.**

## Usage

> ⚠️ **Linux only** — requires `flash-linear-attention` + `triton`. Windows cannot load this model.

```bash
pip install transformers==4.56.2 flash-linear-attention triton
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "eivintobias/heartly-rwkv7-1.5b-v2",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "eivintobias/heartly-rwkv7-1.5b-v2",
    trust_remote_code=True,
)

prompt = "User: What is the capital of France?\nAssistant: "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=160, temperature=0.0)
print(tokenizer.decode(outputs[0]))
```

**Important caveats:**
- `transformers` must be pinned to **4.56.2** (v5 is incompatible with the custom modeling code)
- Batched generation corrupts RWKV state — use **single-prompt generation only**
- The model uses the RWKV world tokenizer (`rwkv_vocab_v20230424`), not BPE
- Many outputs start with a spurious `<tool_call>` token (vocab leak, non-blocking but cosmetic)

## Training Details

- **Base model:** RWKV/RWKV7-Goose-World3-1.5B-HF
- **Training data:** 7,531 Heartly-grammar samples (6,031 original + 1,500 memory-aware)
  - Write-gate samples: facts written in 7 formats (declaration, note, reminder, summary, QA-pair, context-embed, list-item)
  - Retrieval samples: facts stored in embedding store, retrieved and injected as context
  - Context-known samples: answer from provided context (the class that flipped both channels)
- **Training:** 2 epochs, 942 steps, ~34 min on vast.ai RTX 3090, bf16, 16/24 layers frozen
- **SFT renderer:** `render_sft_dataset_v2.py` in the [GitHub repo](https://github.com/eivintobias/heartly)

## Known Issues

| Issue | Details |
|-------|---------|
| **Spurious `<tool_call>` opener** | Many outputs start with a spurious `<tool_call>` token (vocab leak from the thinking token). Cosmetic — stripped by `reply_formatter.py`. |
| **Linux-only inference** | RWKV7 requires `flash-linear-attention` + `triton` → Linux only. |
| **transformers 4.56.2 pinned** | Custom modeling code incompatible with transformers v5. |
| **Batched generation broken** | RWKV recurrence ignores attention_mask → pad tokens poison state. Single-prompt only. |
| **Quiz-stem answer phrasing** | Answers open with "The answer is X" — an artifact of the factual-QA SFT mix. Root fix pre-registered as [Stage 5](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/PREREG_STAGE5.md). |
| **Casual conversation gets no reply** | Greetings and small talk aren't questions, so the model emits `decide=stop` and says nothing. There is no conversational class in the training mix. Root fix: [Stage 5](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/PREREG_STAGE5.md). |
| **Refusal pile-up** | On unknowns the model stacks several refusal phrasings into one reply, occasionally contradicting an answer in the same reply. Collapsed by `reply_formatter.py`; root fix: [Stage 5](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/PREREG_STAGE5.md). |
| **Content accuracy ~15%** | Measured at Stage 3.5 on the Stage 3 model: knowing *when* to speak (99.8% decide accuracy) and being *right* when you do are separate capabilities. This model only claims the first. |

### Displaying replies: `reply_formatter.py`

The Heartly grammar is **internal machinery** — a chat UI should not show it. [`reply_formatter.py`](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/reply_formatter.py) (Stage 4d) parses a raw generation and surfaces only the answer zone:

```python
from reply_formatter import format_reply

raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
print(format_reply(raw))     # -> "Paris." instead of the full tagged transcript
```

It strips the control tags, the think block, meta-commentary ("I know this fact", "the provided context contains the answer"), duplicate refusals, the spurious opener and truncated trailing fragments. When `verify=unknown` it returns a single clean "I don't have that information."

This is **cosmetic by design** — it changes what the user sees, never what the model does. The rows above pointing at Stage 5 need new *training data*, and are documented as open rather than treated as solved.

## Research Context

This model accompanies the research program described in [Research Paper II: The True Boundary](https://github.com/eivintobias/heartly/blob/main/research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md). Key findings validated by this model:

1. **Absence sensors work** — a tiny logistic probe on recurrent state reads known/unknown at AUROC 1.000
2. **Memory requires training to read** — a model can only use a memory channel it was trained on
3. **The disposition block is a single learned gap** — one training class flipped both memory channels open
4. **Redundant writes beat single formats** — writing facts multiple ways gives 5/5 recall vs 0–2/5

Full experiment record: [RESULTS.md](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/RESULTS.md)

## Roadmap

Every stage of this project is **pre-registered before it runs** — hypothesis, method, pass/fail bar and fallback branches written down first, so a failure is a result rather than a reason to move the goalposts.

Next up: **[Stage 5 — Conversational SFT Data](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/PREREG_STAGE5.md)**. This model knows *whether* to speak and *whether* it knows; what it says inside the answer zone is still stiff. Stage 5 re-renders answer zones naturally and adds conversational + persona sample families, with the grammar and the memory classes above preserved untouched and named as regression guards.

Also open: the integrated memory demo — write-gate + retrieval store together in one live session.

## Links

- **GitHub (code, full results, papers):** https://github.com/eivintobias/heartly
- **Experiment record:** [`heartly-rnn/RESULTS.md`](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/RESULTS.md)
- **Stage 4c pre-registration:** [`PREREG_STAGE4C.md`](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/PREREG_STAGE4C.md) — the bars this model was judged against
- **Program paper:** [`RESEARCH_PAPER_II_TRUE_BOUNDARY.md`](https://github.com/eivintobias/heartly/blob/main/research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md)
- **Plain-language papers:** [`research_papers/plain/`](https://github.com/eivintobias/heartly/tree/main/research_papers/plain)
- **Previous model:** [`eivintobias/heartly-rwkv7-1.5b`](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b) (Stage 3, no memory)
- **Track 1 model:** [`eivintobias/heartly-v2`](https://huggingface.co/eivintobias/heartly-v2) (Qwen2.5-0.5B GGUF)

## Citation

```bibtex
@misc{heartly2026,
  author = {Eivin},
  title = {Nature-First AI: Training Language Models Toward Freedom Through Truth},
  year = {2026},
  howpublished = {https://huggingface.co/eivintobias/heartly-rwkv7-1.5b-v2}
}
```

## License

Apache License 2.0. Base model (RWKV7-Goose-World3-1.5B) is subject to its own license terms.
