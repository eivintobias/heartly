---
language: en
license: other
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

model = AutoModelForCausalLM.from_pretrained("Heartly/heartly-v2-qwen2.5-0.5b")
tokenizer = AutoTokenizer.from_pretrained("Heartly/heartly-v2-qwen2.5-0.5b")

prompt = "User: What is the capital of France?\nAssistant: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=160, temperature=0.0)
print(tokenizer.decode(outputs[0]))
```

## Evaluation

See the companion file `heartly_test_prompts.md` for 75 structured test prompts organized into three categories:
- **Category A (30 prompts):** Known facts the model should answer correctly
- **Category B (35 prompts):** Unknown facts where the model should abstain
- **Category C (10 prompts):** Silence triggers where the model should output `<stop>`

## Research Context

This model accompanies the research paper *"Nature-First AI: Training Language Models Toward Freedom Through Truth"* (draft v0.2, included in the repository). Key claims:

1. **Hallucination is a nature problem** — models are optimized to always produce plausible text, so they confabulate when they don't know.
2. **Honesty requires a boundary** — a model can only reliably abstain when its knowledge boundary is explicit.
3. **The dataset is the nature** — behavioral principles should be compiled *into* the supervised data, not only corrected after the fact.

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

Research and educational purposes only. Base model (Qwen2.5-0.5B) terms apply.