# Heartly — Nature-First AI: Training Language Models Toward Freedom Through Truth

**Status:** Research Prototype — Not Production Ready

**Model weights:**
- [🤗 heartly-v2 — Qwen2.5-0.5B, GGUF (Track 1)](https://huggingface.co/eivintobias/heartly-v2)
- [🤗 heartly-rwkv7-1.5b — RWKV7-1.5B + boundary head (Track 2, Stage 3)](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b)
- [🤗 heartly-rwkv7-1.5b-v2 — RWKV7-1.5B + memory (Track 2, Stage 4c)](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b-v2) ← **latest**

Heartly is an experimental small language model project that explores a novel approach to reducing hallucination: instead of training a model to *always answer*, we train it to **decide whether to speak**, **verify what it knows**, and **admit ignorance honestly**.

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
  - **Stage 4b (2026-07-25):** the write-gate OPENS — writing each fact multiple ways in one transcript reaches **4/5 recall** (single formats: 0–2/5); but retrieval-store context injection FAILS (1–2/5 despite perfect retrieval) — the SFT mix has no "answer from provided context" class, so injected context is scenery, not knowledge. Lesson: **a model can only use a memory channel it was trained to read.**
  - **Stage 4c (2026-07-25):** ALL BARS PASS — memory-aware SFT retrain on 7,531 samples (942 steps, ~34 min). No regression (grammar 300/300, say/sense 1.0, boundary head AUROC 1.0); write-gate ALL 7 formats **5/5** (W1 declarations were 0/5 before!); retrieval 5/5 + context injection I1 **4/5**, I2 **4/5**. One training class (context-known) flipped BOTH memory channels — the disposition block was a single learned gap. **§6.2 validated end-to-end.** Published: [eivintobias/heartly-rwkv7-1.5b-v2](https://huggingface.co/eivintobias/heartly-rwkv7-1.5b-v2)
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

### Memory Architecture (Stage 4c, latest model)

The v2 model adds two memory channels on top of the base decide/verify/stop grammar:

- **Write-gate:** the model writes facts into its recurrent state during conversation using multiple redundant formats (declaration, note, reminder, summary, etc.). Writing the same fact several ways in one session gives **5/5 recall** — single-format writes only reach 0–2/5.
- **Retrieval store:** an external embedding store that retrieves relevant context and injects it into the prompt. The model reads injected context and answers from it at **4/5 accuracy** — but only because the SFT data includes a "context-known" training class. Without that class, injected context is invisible scenery.

Key finding: **a model can only use a memory channel it was trained to read.** The disposition block (refusing to answer personal questions) was a single learned gap — one training class flipped both memory channels open.

---

## Training Data

### Track 1 (v2, Qwen2.5-0.5B)

Fine-tuned on **12 datasets** across four domains:

| Domain | Datasets | Samples |
|--------|----------|---------|
| **Factual QA** | SQuAD v1.1, TriviaQA, Natural Questions Open, SciQ, BoolQ, WebQuestions | ~155k |
| **Coding** | CodeAlpaca-20k, Python Code Instructions 18k, MBPP | ~39k |
| **Instructions** | Dolly-15k, Alpaca | ~45k |
| **Math** | GSM8K | ~7.5k |

### Track 2 (RWKV7-1.5B)

- **Stage 3 model** (heartly-rwkv7-1.5b): 6,031 Heartly-grammar samples, 2 epochs, 754 steps
- **Stage 4c model** (heartly-rwkv7-1.5b-v2): 7,531 samples (original 6,031 + 1,500 memory-aware samples covering write-gate, retrieval, and context-known classes), 2 epochs, 942 steps

All factual QA was routed through a `KnowledgeBase` → `DatasetRenderer` pipeline that generates both positive examples (known facts) and boundary-negative examples (unknown facts at the edge of the KB) to train the abstention behavior.

---

## Current Known Issues

### Track 1 (v2, checkpoint step 33500)

| Issue | Description |
|-------|-------------|
| **Entity/attribute mapping collapse** | Questions from TriviaQA/Natural Questions are stored with entity="general trivia" and the full question as the attribute. This produces nonsensical query templates like "What is the 'What is the capital of France?' of general trivia?" |
| **Special token emission failure** | The model outputs `speakknown` without proper `<decide>`/`<verify>` tag boundaries, suggesting tokenizer vocabulary misalignment during GGUF conversion. |
| **Reasoning block leaks** | The ` thinking ...  response` internal reasoning block is emitted as visible text rather than being handled as a hidden scratchpad. |
| **Repetition loops** | The model gets stuck repeating phrases (e.g., "America's fastest ship was the battleship Enterprise") — the `<stop>` token is not reliably recognized as an EOS signal during generation. |
| **Confident wrong answers** | The model answers "Who was the first US president?" with "Charles Lindbergh" and "Thomas Jefferson" — the abstention mechanism (`<verify>unknown</verify>`) does not fire when it should. |
| **Verbose output** | Despite the decide/verify/stop mechanics, the model still produces very long answers — the underlying chat-nature from the base model was not fully overridden. |

### Track 2 (v2, Stage 4c model)

| Issue | Description |
|-------|-------------|
| **Spurious `<tool_call>` opener** | Many outputs start with a spurious `<tool_call>` token (vocab leak from the thinking token). Non-blocking but cosmetic. |
| **Linux-only inference** | RWKV7 requires `flash-linear-attention` + `triton` → Linux only. Windows cannot load the model. |
| **transformers 4.56.2 pinned** | The custom modeling code is incompatible with transformers v5. |

These issues are **instructive failures**: they demonstrate that special tokens alone are insufficient — the training data must properly embody the desired nature at sufficient scale and quality.

---

## Repository Structure

```
├── README.md                        # This file
├── LICENSE                          # Apache 2.0
├── HF_MODEL_CARD.md                 # Hugging Face model card (v2/Qwen)
├── heartly_test_prompts.md          # 75 test prompts for evaluation
├── newLLMdesign.ipynb               # Full Colab notebook (v2 training pipeline)
├── checkpoint-33500/                # v2 model scripts + test logs
├── heartly-v3/                      # v3 interrupted checkpoint + GGUF scripts
├── heartly-rnn/                     # Track 2 lab (RWKV experiments)
│   ├── RESULTS.md                   # Full experiment record (Stages 1–4c)
│   ├── README.md                    # Track 2 overview
│   ├── PREREG_STAGE4.md            # Stage 4 pre-registration
│   ├── PREREG_STAGE4B.md           # Stage 4b pre-registration
│   ├── PREREG_STAGE4C.md           # Stage 4c pre-registration
│   ├── gen_probe_dataset.py        # True-boundary question generator
│   ├── extract_states.py           # Recurrent state extraction
│   ├── train_probe.py              # Boundary head training
│   ├── render_sft_dataset.py       # SFT data renderer (v1)
│   ├── render_sft_dataset_v2.py    # SFT data renderer (v2, memory-aware)
│   ├── finetune_rwkv.py            # RWKV fine-tuning script
│   ├── measure_say_sense.py        # Say/sense agreement measurement
│   ├── analyze_report.py           # Results analysis
│   ├── memory_store.py             # Episodic memory store (write-gate + retrieval)
│   ├── stage4b_write_gate.py       # Stage 4b write-gate tests
│   ├── stage4b_retrieval.py        # Stage 4b retrieval tests
│   ├── stage4c_write_gate.py       # Stage 4c write-gate tests
│   ├── stage4c_retrieval.py        # Stage 4c retrieval tests
│   ├── gen_critic_data.py          # Critic data generation
│   ├── train_critic.py             # Critic training
│   ├── fit_critic.py               # Fitted critic experiments
│   ├── pick_tracked.py             # Tracked-confabulation selection
│   ├── smoke_rwkv7.py             # RWKV7 smoke test
│   └── ...                         # Shell scripts, requirements, etc.
├── research_papers/
│   ├── RESEARCH_PAPER.md           # Position paper (draft v0.2)
│   ├── RESEARCH_BRIEF_TECHNICAL.md # Technical brief (H1–H4)
│   ├── RESEARCH_PAPER_II_TRUE_BOUNDARY.md  # Full program paper
│   └── plain/                      # Plain-language copies of all papers
```

---

## How to Use

### Track 2 — RWKV7-1.5B (latest, Stage 4c)

> ⚠️ **Linux only** — requires `flash-linear-attention` + `triton`. Windows cannot load this model.

```bash
pip install transformers==4.56.2 flash-linear-attention triton
```

```python
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

### Track 1 — Qwen2.5-0.5B (v2)

```bash
# With llama.cpp
./main -m heartly-v2-qwen2.5-0.5b-f16.gguf \
  -p "User: What is the capital of France?\nAssistant: " \
  -n 160 --temp 0
```

---

## The Research

The full research paper is included in this repository (`RESEARCH_PAPER_II_TRUE_BOUNDARY.md`). Key claims:

1. **Hallucination is a nature problem, not a knowledge problem** — models are optimized to always produce plausible text, so they confabulate when they don't know.
2. **Honesty requires a boundary** — a model can only reliably abstain when its knowledge boundary is explicit (a curated KB), so abstention examples can be generated *at the boundary* by construction.
3. **The dataset is the nature** — behavioral principles should be compiled *into* the supervised data (ratios of abstention, silence, restraint) rather than only corrected after the fact.
4. **Absence sensors work** — a tiny logistic probe on recurrent state reads known/unknown at AUROC 1.000, and the signal survives fine-tuning at every layer.
5. **Memory requires training to read** — a model can only use a memory channel it was trained on; the disposition block (refusing personal questions) was a single learned gap that one training class flipped open.

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
  howpublished = {https://huggingface.co/eivintobias/heartly-rwkv7-1.5b-v2}
}
