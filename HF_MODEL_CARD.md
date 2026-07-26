---
language: en
license: apache-2.0
library_name: transformers
tags:
- heartly
- nature-first-ai
- hallucination-reduction
- abstention
- research
- qwen2.5
- fine-tuned
datasets:
- rajpurkar/squad
- mandarjoshi/trivia_qa
- google-research-datasets/nq_open
- allenai/sciq
- google/boolq
- stanfordnlp/web_questions
- sahil2801/CodeAlpaca-20k
- iamtarun/python_code_instructions_18k_alpaca
- google-research-datasets/mbpp
- databricks/databricks-dolly-15k
- tatsu-lab/alpaca
- openai/gsm8k
base_model: Qwen/Qwen2.5-0.5B
pipeline_tag: text-generation
widget:
- text: "User: What is the capital of France?\nAssistant: "
  example_title: "Known fact"
- text: "User: Who founded the Andromeda Council?\nAssistant: "
  example_title: "Unknown fact"
- text: "User: ...\nAssistant: "
  example_title: "Silence trigger"
inference:
  parameters:
    temperature: 0.0
    max_new_tokens: 160
---

# Heartly v2 — Research Prototype

> ### 📌 This is Track 1 — the newer work lives on RWKV
>
> Heartly runs two tracks. This model is **Track 1** (Qwen2.5-0.5B), the first published artifact and the one whose *failures* are documented in detail below. **Track 2** went further on recurrent models and is where the project's actual results are:
>
> - **[heartly-rwkv7-1.5b-v2](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b-v2)** ← latest — RWKV7-1.5B, grammar 100%, decide accuracy 100%, boundary head AUROC 1.000, plus working memory channels
> - **[heartly-rwkv7-1.5b](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b)** — the Stage 3 baseline
>
> This card stays up because the failure data is the contribution: it shows concretely that **control tokens alone don't work** — grammar is cheap, the training distribution is the mechanism. That lesson is what produced Track 2.
>
> **Code, full experiment record and papers:** https://github.com/eivintobias/heartly

**⚠️ EARLY RESEARCH PROTOTYPE — NOT PRODUCTION READY ⚠️**

This model is an experimental research artifact. It has significant known issues (see below) and should not be used in any application where reliability is required. It is published to share the *design approach* and the *failure data* with the research community.

## What is Heartly?

Heartly is a 0.5B parameter model (fine-tuned from Qwen2.5-0.5B) that explores a novel approach to hallucination reduction. Instead of training a model to always answer, Heartly is trained to:

1. **Decide whether to speak** — using `<decide>speak|stop</decide>` tokens
2. **Verify what it knows** — using `<verify>known|unknown</verify>` tokens
3. **Admit ignorance honestly** — saying "I do not have information" rather than confabulating
4. **Stay silent when appropriate** — using `<stop>` as a valid, complete output

The core thesis: *hallucination is not primarily a knowledge problem — it's a nature problem.* A model optimized to always produce confident text develops a nature in which silence and honest ignorance do not exist. Heartly attempts to change that nature at the data layer.

## Output Format

```
 thinking [internal reasoning]  response
<decide>speak|stop</decide>
<verify>known|unknown</verify>
[answer if known]
<stop>
```

## Training Data

Fine-tuned on 12 datasets across four domains (~247k total samples):

| Domain | Datasets |
|--------|----------|
| **Factual QA** | SQuAD v1.1, TriviaQA, Natural Questions Open, SciQ, BoolQ, WebQuestions |
| **Coding** | CodeAlpaca-20k, Python Code Instructions 18k, MBPP |
| **Instructions** | Dolly-15k, Alpaca |
| **Math** | GSM8K |

Factual QA was processed through a Knowledge Base → DatasetRenderer pipeline that generates both positive examples (known facts) and boundary-negative examples (unknown facts at the edge of the KB) to train abstention behavior.

## Known Limitations

Based on testing at checkpoint step 33500:

| Issue | Details |
|-------|---------|
| **Entity/attribute mapping collapse** | Questions from TriviaQA/Natural Questions are stored with entity="general trivia" and the full question as attribute, producing nonsensical query templates. |
| **Special token emission failure** | The model outputs `speakknown` without proper `<decide>`/`<verify>` tag boundaries — tokenizer vocabulary likely misaligned during GGUF conversion. |
| **Reasoning block leaks** | The ` thinking ...  response` internal reasoning is emitted as visible text rather than handled as a hidden scratchpad. |
| **Repetition loops** | The model gets stuck repeating phrases — `<stop>` is not reliably recognized as an EOS signal. |
| **Confident wrong answers** | The abstention mechanism (`<verify>unknown</verify>`) does not reliably fire for out-of-knowledge questions. |
| **Verbose output** | Despite the decide/verify/stop mechanics, the model still produces very long answers — the base model's chat-nature was not fully overridden. |

These are **instructive failures**: they demonstrate that special tokens alone are insufficient — the training data must properly embody the desired nature at sufficient scale and quality.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("eivintobias/heartly-v2")
tokenizer = AutoTokenizer.from_pretrained("eivintobias/heartly-v2")

prompt = "User: What is the capital of France?\nAssistant: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=160, temperature=0.0)
print(tokenizer.decode(outputs[0]))
```

## Evaluation

See [`heartly_test_prompts.md`](https://github.com/eivintobias/heartly/blob/main/heartly_test_prompts.md) for 75 structured test prompts organized into three categories:
- **Category A (30 prompts):** Known facts the model should answer correctly
- **Category B (35 prompts):** Unknown facts where the model should abstain
- **Category C (10 prompts):** Silence triggers where the model should output `<stop>`

## Research Context

This model accompanies the research paper *"Nature-First AI: Training Language Models Toward Freedom Through Truth"* (draft v0.2, included in the repository). Key claims:

1. **Hallucination is a nature problem** — models are optimized to always produce plausible text, so they confabulate when they don't know.
2. **Honesty requires a boundary** — a model can only reliably abstain when its knowledge boundary is explicit.
3. **The dataset is the nature** — behavioral principles should be compiled *into* the supervised data, not only corrected after the fact.

### What happened after this model

The failures on this card motivated Track 2, which tested the same ideas on recurrent models where the internal state can be read directly. Headline results, all in [RESULTS.md](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/RESULTS.md):

- **Absence sensors work.** A ~2k-parameter logistic probe on RWKV recurrent state reads *known vs unknown* at **AUROC 1.000** — the model's knowledge boundary is linearly present in its own state.
- **The grammar sticks at scale.** RWKV7-1.5B reaches **100% grammar adoption and 100% decide accuracy** — none of the `speakknown` token collapse seen here.
- **But the sensor has a blind spot.** It shares state with the generator, so *confident* confabulation slips past. An independent critic detects it in principle, and the requirement was measured: **the critic must be stronger than the generator**.
- **Memory needs training to read.** Injected context is invisible scenery unless the SFT mix contains an "answer from provided context" class — one training class flipped both memory channels open.
- **Knowing ≠ being right.** Content accuracy at 1.5B is ~15% while decide accuracy is 99.8%. Knowing when to speak and being correct when you do are separate capabilities; Heartly only claims the first.

## Links

- **GitHub (code, full results, papers):** https://github.com/eivintobias/heartly
- **Experiment record:** [`heartly-rnn/RESULTS.md`](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/RESULTS.md)
- **Program paper:** [`RESEARCH_PAPER_II_TRUE_BOUNDARY.md`](https://github.com/eivintobias/heartly/blob/main/research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md)
- **Plain-language papers:** [`research_papers/plain/`](https://github.com/eivintobias/heartly/tree/main/research_papers/plain)
- **Roadmap:** [Stage 5 pre-registration](https://github.com/eivintobias/heartly/blob/main/heartly-rnn/PREREG_STAGE5.md)
- **Test suite used here:** [`heartly_test_prompts.md`](https://github.com/eivintobias/heartly/blob/main/heartly_test_prompts.md)
- **Latest model:** [`eivintobias/heartly-rwkv7-1.5b-v2`](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b-v2)

## Citation

```bibtex
@misc{heartly2026,
  author = {Eivin},
  title = {Nature-First AI: Training Language Models Toward Freedom Through Truth},
  year = {2026},
  howpublished = {https://huggingface.co/eivintobias/heartly-v2}
}
```

## License

Apache License 2.0 — see [LICENSE](https://github.com/eivintobias/heartly/blob/main/LICENSE) in the repository. Base model (Qwen2.5-0.5B) is subject to its own license terms.
