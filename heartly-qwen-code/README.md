# Heartly Qwen-Code

A coding LLM with the Heartly hallucination-reduction architecture, built on **Qwen2.5-Coder-1.5B**.

## What is this?

This project applies the Heartly approach (decide/verify/stop grammar + boundary head absence sensor) to code generation. The model learns to:

- **Decide** whether it can answer a coding question (`<decide>speak|stop</decide>`)
- **Verify** whether it actually knows the answer (`<verify>known|unknown</verify>`)
- **Admit ignorance** when asked about non-existent APIs, libraries, or impossible tasks
- **Write correct code** when it knows the answer

## Architecture

| Component | Detail |
|-----------|--------|
| Base model | Qwen/Qwen2.5-Coder-1.5B (1.5B transformer) |
| Grammar | Heartly: ` thinking ...  response<decide>...</decide><verify>...</verify> answer <stop>` |
| Boundary head | Logistic regression on final hidden state at `<verify>` position |
| Training | Full fine-tune or QLoRA, loss masked to response tokens |

## Files

| File | Purpose |
|------|---------|
| `render_code_sft.py` | Generate code SFT dataset in Heartly grammar |
| `finetune_qwen.py` | Fine-tune Qwen2.5-Coder on the Heartly code data |
| `train_probe_code.py` | Train boundary head probe on hidden states |
| `measure_say_sense_code.py` | Evaluate grammar adoption + coding accuracy |
| `run_code_stage1.sh` | Entry point for vast.ai 3090 training |
| `sft_dataset_code.jsonl` | Generated SFT dataset (instruction + output) |
| `RESULTS.md` | Results log |

## Quick Start

### 1. Generate the dataset
```bash
python render_code_sft.py --out sft_dataset_code.jsonl
```

### 2. Train (on GPU)
```bash
# Full fine-tune (needs 24GB+)
python finetune_qwen.py --data sft_dataset_code.jsonl --out heartly-qwen-code

# QLoRA (works on 11GB, e.g. RTX 2080 Ti)
python finetune_qwen.py --data sft_dataset_code.jsonl --out heartly-qwen-code --qlora
```

### 3. Train boundary head
```bash
python train_probe_code.py --model heartly-qwen-code --data sft_dataset_code.jsonl
```

### 4. Evaluate
```bash
python measure_say_sense_code.py --model heartly-qwen-code
```

## Dataset

The SFT dataset (`sft_dataset_code.jsonl`) contains ~6,500 samples:

- **2,000 known code tasks** — standard programming problems (sort, search, string manipulation, etc.)
- **500 code completions** — finish-the-function prompts
- **800 unknown tasks** — non-existent APIs, impossible tasks, post-cutoff features
- **200 silence triggers** — empty/noise inputs
- **3,000 Magicoder samples** — diverse coding instructions (if available)

Each sample is in `{instruction, output}` format with the Heartly grammar.

## Roadmap

- **Stage 1** (current): Code SFT + grammar adoption + boundary head
- **Stage 2**: Code critic (detect confident confabulations in code)
- **Stage 3**: Memory/state persistence for coding context
- **Stage 4**: Conversational code chat (multi-turn, persona)
- **Stage 5**: Publish on HuggingFace

## License

MIT