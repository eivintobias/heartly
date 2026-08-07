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

---

## Stage 5 — Serving Heartly Qwen-Code v3

The v3 model (`heartly-qwen-code-v3/`) ships as a full fine-tune on
`Qwen2.5-Coder-1.5B` with the conversational Stage-5 SFT recipe. It is served via
a small **FastAPI `server.py`** that loads the weights and routes output through
`reply_formatter.py`, so the `<decide>/<verify>/<stop>` grammar never reaches the
user.

### Run the server
```bash
pip install -r requirements.txt        # adds fastapi + uvicorn
python server.py --model heartly-qwen-code-v3 --port 8000
# custom model:  python server.py --model <path-or-hf-repo> --port 8000
```

### Chat over HTTP
```bash
# health check (instant — model loads lazily on first /chat)
curl -s http://127.0.0.1:8000/health
# chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a function that sorts a list"}'
```
`/chat` returns JSON: `{"model":"heartly-qwen-code-v3","raw":"...grammar...","reply":"<clean answer>"}`.

### Quick offline test (no server)
```bash
python chat_smoke.py "Write a function that reverses a list"
# or pipe:  echo "your question" | python chat_smoke.py
```

### Notes
- The `<decide>/<verify>/<stop>` tags are ordinary multi-token text (not tokenizer
  special tokens), so they can decode mangled through some GUI front-ends. The
  server's `reply_formatter` canonicalises and strips them — prefer `/chat`.
- `model.safetensors` (3 GB) is git-ignored; weights live on HuggingFace, not
  GitHub. See `HF_MODEL_CARD_v3.md`.