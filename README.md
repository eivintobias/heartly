# Heartly — Nature-First AI: Training Language Models Toward Freedom Through Truth

**Status:** Early Research Prototype (v2) — Not Production Ready

**Model weights:** [🤗 heartly-v2 — Qwen2.5-0.5B, GGUF (Track 1)](https://huggingface.co/eivintobias/heartly-v2) · [🤗 heartly-rwkv7-1.5b — RWKV7-1.5B + boundary head (Track 2)](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b)

Heartly is an experimental small language model (0.5B parameters, based on Qwen2.5) that explores a novel approach to reducing hallucination: instead of training a model to *always answer*, we train it to **decide whether to speak**, **verify what it knows**, and **admit ignorance honestly**.

The core idea is that hallucination is not primarily a knowledge problem — it's a **nature problem**. A model trained to always produce a confident answer develops a nature in which silence, refusal, and honest ignorance do not exist. Heartly attempts to change that nature at the data level.

---

## 🧪 Latest Research — Track 2: Absence Sensors on RNN States (July 2026)

New since v2: the project has a second research track — **boundary heads and independent critics on recurrent models** (RWKV, Falcon-H1) that read the model's own state to detect when it doesn't know.

- **[heartly-rnn/RESULTS.md](heartly-rnn/RESULTS.md)** — the full experiment record:
  - **Stage 1:** recurrent states read known/unknown at **AUROC 1.000** (2,902 true-boundary questions)
  - **Stage 2:** RWKV-430m fine-tuned on the Heartly grammar — 100% grammar adoption, 97.7% decide accuracy… but the boundary head is blind to *confident* confabulation
  - **Stage 2.5:** an independent critic fixes the blind spot in principle, but fails the pre-registered bar at 0.5B scale. Lesson: **the critic must be stronger than the generator**
  - **Stage 3 (2026-07-23):** the recipe scales — **RWKV7-Goose-1.5B** fine-tuned in ~15 min on a single 3090: grammar 100%, decide accuracy **100%**, boundary head **AUROC 1.000 at every probed layer**, say/sense agreement 100% — published: [eivintobias/heartly-rwkv7-1.5b](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b)
  - **Stage 2.6 (2026-07-24):** the asymmetry requirement, measured — critics of growing size grade the 0.43B generator's answers: AUROC 0.758 (0.5B) → 0.826 (1.5B) → 0.845 (3B). The deployment bar still fails — provably because the starved 60-sample correct class leaves no operating point — and a bonus finding: at 7× asymmetry the critic turns *over-strict*, distrusting even correct answers
  - **Stage 3.5 (2026-07-25):** the real deployment test at 1.5B — content accuracy 15.5% (was 4.1%), decide accuracy 99.8%, **zero unknown-side confabulations**; critics replicate every structural finding but the threshold still fails: the bottleneck is the correct-answer *distribution*, not its size
  - **Stage 3.6 (2026-07-25):** the fitted critic, tried and CLOSED — six fitting methods on the same features, all fail the bar three different ways. The ceiling is in the features, not the fitter. **Ranking-as-product** (the bottom 10% of scores is 93–100% confabulation) becomes the critic deliverable
  - **Stage 4 (2026-07-25):** memory — state save/load is **EXACT** (12.8MB state file, logits identical to 3 decimals, live ≡ reloaded), but episodic fact recall from raw state priming is weak (0–1/5) and the abstention disposition refuses personal-context questions even when the answer sits in the state
  - **Stage 4b (2026-07-25):** the write-gate OPENS — writing each fact multiple ways in one transcript reaches **4/5 recall** (single formats: 0–2/5); but retrieval-store context injection FAILS (1–2/5 despite perfect retrieval) — the SFT mix has no "answer from provided context" class, so injected context is scenery, not knowledge. Lesson: **a model can only use a memory channel it was trained to read.** Next: Stage 4c (memory-aware SFT retrain)
- **[research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md](research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md)** — Research Paper II: the full program (boundary error, negative-side mechanisms, memory track)
- **[research_papers/plain/](research_papers/plain/)** — plain-language copies of all papers

---

## The Architecture

Heartly uses three special token mechanics in its output grammar:

```
 thinking [internal reasoning]  response
<decide>speak|stop</decide>
<verify>known|unknown</verify>
[answer if known]
<stop>
```

1. **Decide-to-speak** — The model first decides whether to respond at all (`<decide>speak</decide>` or `<decide>stop</decide>`). Silence is a valid, first-class output.
2. **Verify-before-claim** — Before asserting a fact, the model checks against its knowledge (`<verify>known</verify>` or `<verify>unknown</verify>`).
3. **Stop as rest** — `<stop>` ends the conversation. The model can choose to stop rather than continue generating.

---

## Training Data

Heartly v2 was fine-tuned on a mix of **12 datasets** across four domains:

| Domain | Datasets | Samples |
|--------|----------|---------|
| **Factual QA** | SQuAD v1.1, TriviaQA, Natural Questions Open, SciQ, BoolQ, WebQuestions | ~155k |
| **Coding** | CodeAlpaca-20k, Python Code Instructions 18k, MBPP | ~39k |
| **Instructions** | Dolly-15k, Alpaca | ~45k |
| **Math** | GSM8K | ~7.5k |

All factual QA was routed through a `KnowledgeBase` → `DatasetRenderer` pipeline that generates both positive examples (known facts) and boundary-negative examples (unknown facts at the edge of the KB) to train the abstention behavior.

---

## Current Known Issues

Based on testing with the v2 checkpoint (step 33500):

| Issue | Description |
|-------|-------------|
| **Entity/attribute mapping collapse** | Questions from TriviaQA/Natural Questions are stored with entity="general trivia" and the full question as the attribute. This produces nonsensical query templates like "What is the 'What is the capital of France?' of general trivia?" |
| **Special token emission failure** | The model outputs `speakknown` without proper `<decide>`/`<verify>` tag boundaries, suggesting tokenizer vocabulary misalignment during GGUF conversion. |
| **Reasoning block leaks** | The ` thinking ...  response` internal reasoning block is emitted as visible text rather than being handled as a hidden scratchpad. |
| **Repetition loops** | The model gets stuck repeating phrases (e.g., "America's fastest ship was the battleship Enterprise") — the `<stop>` token is not reliably recognized as an EOS signal during generation. |
| **Confident wrong answers** | The model answers "Who was the first US president?" with "Charles Lindbergh" and "Thomas Jefferson" — the abstention mechanism (`<verify>unknown</verify>`) does not fire when it should. |
| **Verbose output** | Despite the decide/verify/stop mechanics, the model still produces very long answers — the underlying chat-nature from the base model was not fully overridden. |

These issues are **instructive failures**: they demonstrate that special tokens alone are insufficient — the training data must properly embody the desired nature at sufficient scale and quality.

---

## Repository Structure

```
├── newLLMdesign.ipynb          # Full Colab notebook (training pipeline)
├── checkpoint-33500/
│   ├── rebuild_tokenizer_v2.py # Tokenizer with Heartly special tokens
│   ├── fix_gguf_metadata.py    # GGUF metadata patching
│   └── checkpoint-33500/       # Model weights (GGUF format)
├── heartly_test_prompts.md     # 75 test prompts for evaluation
├── research_papers/
│   ├── RESEARCH_PAPER.md       # Full research paper (draft v0.2)
│   └── RESEARCH_BRIEF.md       # Executive summary
└── README.md                   # This file
```

---

## How to Use

### Requirements
- Python 3.10+
- `transformers` (for the base model)
- `llama.cpp` (for GGUF inference)

### Quick Start (with llama.cpp)

```bash
# Build llama.cpp
cd llama.cpp
make

# Run inference
./main -m checkpoint-33500/checkpoint-33500/heartly-v2-qwen2.5-0.5b-f16.gguf \
  -p "User: What is the capital of France?\nAssistant: " \
  -n 160 \
  --temp 0
```

### Running the Evaluation

The notebook includes a behavioral evaluation scorecard (Cell 10) that tests:
- **Known facts** → expects `<verify>known</verify>` with correct answer
- **Unknown facts** → expects `<verify>unknown</verify>` (admit ignorance)
- **Silence triggers** → expects `<decide>stop</decide>`

See `heartly_test_prompts.md` for 75 structured test prompts with expected answers.

---

## The Research

The full research paper is included in this repository (`RESEARCH_PAPER.md`). Key claims:

1. **Hallucination is a nature problem, not a knowledge problem** — models are optimized to always produce plausible text, so they confabulate when they don't know.
2. **Honesty requires a boundary** — a model can only reliably abstain when its knowledge boundary is explicit (a curated KB), so abstention examples can be generated *at the boundary* by construction.
3. **The dataset is the nature** — behavioral principles should be compiled *into* the supervised data (ratios of abstention, silence, restraint) rather than only corrected after the fact.

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE). You are free to use, modify, and build on this work (including commercially) with attribution. The base models (Qwen2.5-0.5B, RWKV7-Goose-World3-1.5B) are subject to their own license terms.

---

## Citation

```bibtex
@misc{heartly2026,
  author = {Eivin},
  title = {Nature-First AI: Training Language Models Toward Freedom Through Truth},
  year = {2026},
  howpublished = {https://huggingface.co/eivintobias/heartly-v2}
}