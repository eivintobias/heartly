# Heartly Qwen-Code — Results

## Stage 1 — Code SFT + Grammar Adoption + Boundary Head

**Status:** COMPLETE (2026-07-28, instance 46138938)

### Baseline (before training)

| Metric | Value |
|--------|-------|
| Base model | Qwen/Qwen2.5-Coder-1.5B |
| Grammar adoption | 0% (no Heartly grammar in base model) |
| Verify accuracy | N/A |
| Boundary head AUROC | N/A |

### Target (after Stage 1)

| Metric | Target | Result | PASS? |
|--------|--------|--------|-------|
| Grammar adoption | ≥95% | **100.0%** (35/35) | ✅ |
| Verify accuracy | ≥90% | **100.0%** (35/35 — 20/20 known, 15/15 unknown) | ✅ |
| Boundary head AUROC | ≥0.99 | **1.000** (accuracy 1.0 on test set) | ✅ |
| Confabulation rate | ≤10% | **0%** (0 confabulations) | ✅ |

### Training

| Detail | Value |
|--------|-------|
| Instance | 46138938 |
| GPU | NVIDIA RTX 3090 (24GB) |
| Duration | ~35 min total (all 3 steps) |
| Cost | ~$0.50 |
| Samples | 6,500 |
| Epochs | 2 |
| Batch size | 4 |
| Grad accum | 4 |
| Max length | 512 |
| Freeze layers | 12/28 (trainable 749M / 1,544M — 51% frozen) |
| Dtype | bf16 (fp16 had grad-scaling conflict with transformers v5) |
| Final loss | 0.1764 (epoch ~1.77) |
| Probe hidden dim | 1536 |

### Results

**Grammar adoption:** 100% (35/35 parseable)
**Verify accuracy:** 100% (35/35 correct — 20/20 known, 15/15 unknown)
**Boundary head AUROC:** 1.000 (141 test samples)
**Confabulations:** 0 (all 15 unknown-prompt samples correctly refused)

The Heartly grammar transfer to Qwen2.5-Coder is **perfect** — the model adopted the decide/verify/stop format on all 35 eval prompts, correctly distinguished known coding tasks from unknown/non-existent APIs, and the boundary head achieves perfect separation (AUROC 1.000) on the final hidden layer at the `<verify>` position.

**Decision:** ✅ ALL BARS PASSED. Proceed to Stage 2 (code critic) when ready. Instance 46138938 — DESTROY.

---

## Stage 2 — Code Critic

**Status:** COMPLETE (2026-07-29, local laptop RTX 2070 Max-Q)

### What it does
The code critic reads the model's hidden state at the `<verify>` position and classifies: is this generated code actually correct, or a confident confabulation? This targets the "confident but wrong" blind spot where the model says `verify=known` but writes fake code for non-existent libraries.

### Training

| Detail | Value |
|--------|-------|
| Model | Qwen2.5-Coder-1.5B + QLoRA (r=16, 4-bit) |
| GPU | NVIDIA RTX 2070 Max-Q (8GB) |
| QLoRA training | ~62 min, loss 0.27 → 0.37 avg |
| Critic data gen | 40 prompts, ~20 min |
| Critic training | ~2 min |
| Cost | $0 (local) |

### Critic Data

| Class | Count | Description |
|-------|-------|-------------|
| correct | 25 | Model produced valid code for known tasks |
| confab_unknown | 11 | Model confidently wrote fake code for non-existent APIs |
| abstain | 4 | Model correctly refused unknown tasks |
| **Total** | **40** | |

### Results

| Metric | Value |
|--------|-------|
| **AUROC** | **1.000** (perfect separation) |
| Accuracy | 81.8% (9/11 test) |
| Confab recall | 100% (3/3 caught) |
| Correct precision | 100% (no false alarms on correct code) |
| Hidden dim | 1536 |

The critic achieves **perfect AUROC** — it can distinguish correct code from confabulated code by reading the model's hidden state at the `<verify>` position. All 3 confabulations in the test set were caught (100% recall), with zero false alarms on correct code (100% precision).

**Key finding:** The Qwen-Coder model's hidden state at `<verify>` carries enough signal to separate "I know this and my code is real" from "I'm confidently making up fake API code." This replicates the Heartly RWKV7 finding on a transformer architecture.

**Decision:** ✅ Stage 2 PASSES. The code critic works. Proceed to Stage 3 (memory/state persistence) when ready.

## Stage 3 — Memory/State Persistence (future)

## Stage 4 — Conversational Code Chat (future)

## Stage 5 — Publish on HuggingFace (future)

---

## Stage 4d — Reply Formatter (display layer) ✅

**Status:** COMPLETE (2026-08-07).

`heartly-qwen-code/reply_formatter.py` + `test_reply_formatter.py` = **15/15
green** (`py_compile` OK; `ResourceWarning` squashed). The formatter strips
`<decide>/<verify>/<stop>` grammar tags, `thinking` blocks, duplicate refusals,
and truncated/mangled token fragments (`<deside>`, unclosed `<stop>`) from the
displayed output only — the model is untouched. Code-block indentation is
preserved; empty-answer zones fall back to `…`; explicit "the answer is X"
stems are salvaged rather than silenced. Regression tests lock the edge cases.

## Stage 5 — Conversational serving

**Status:** COMPLETE (2026-08-07). `server.py` (FastAPI) serves the v3 model at
`http://127.0.0.1:8000/`, including a browser chat UI at `GET /`. `/chat` loads the
weights lazily on the first request, then routes the raw Heartly grammar through
`reply_formatter.format_reply(raw, mode='chat')`.

- **Grammar strip:** removes `thinking` blocks and `<decide>/<verify>/<stop>` plus
  mangled tag fragments — only the clean answer reaches the UI.
- **Code-rendering fix:** the model emits in-code newlines as escaped text;
  `_clean_text` unescapes them inside fenced code blocks, so code renders multi-line
  instead of collapsing to one string.
- **UI fix:** the chat `Send` button shipped `disabled` (permanent deadlock); the
  attribute was removed and the button toggles only while streaming.
- **Verified:** `GET /` returns the chat UI (HTTP 200); `POST /chat` with prompt
  `What is 2+2?` returns `2 + 2 = 4`; a code prompt returns a fenced, multi-line block.
- **Serving footprint:** CPU-only; 1.5 B params need no GPU for inference.

## Phase 2 — Trillion-parameter scaling (Kimi k1.6) 📋

**Status:** PLANNED (see `ROADMAP.md` "NEW OPTION E").

Scaling Heartly to ~1 T params via Moonshot's open Kimi k1.6 instead of
incrementally up the 1.5B→14B ladder. The Stage-5 SFT recipe transfers unchanged
(`finetune_qwen.py` uses HF `AutoModelForCausalLM`/`AutoTokenizer` → the
architecture is auto-resolved from the Kimi config). Recommended staging:
**QLoRA r16/nf4** (~26 M trainable params, 4–8×B200) — train ~$35–90 (spot);
serving 1 T params needs a multi-GPU vLLM shard (~$36–72/hr); the local
`server.py`/llama-cpp path only fits the v3 1.5B GGUF. Open gates: exact HF repo
id + fine-distributable license, dense vs MoE, and tokenizer overlap with
Heartly markup.


## Final live verification (2026-08-07)

Re-spawned a fresh server (old pid 8604 killed so it served the updated
`reply_formatter.py`); `heartly-qwen-code-v3` loaded in ~4 s.

- `GET /health` -> `{"status":"ready","model":"heartly-qwen-code-v3"}`.
- `GET /` -> HTTP 200 with the chat UI; the `Send` button has no static `disabled`
  attribute and CSS `white-space:pre-wrap` preserves newlines (UI fix confirmed).
- `POST /chat` code-rendering probe -- all 3 code prompts green:
  `real_newlines=True, backslash_n_left=False`. Fenced `python` blocks render multi-line
  with real line breaks -- the original one-line collapse is resolved in both the
  grammar-stripping path and the browser UI.
- HuggingFace (public): https://huggingface.co/eivintobias/heartly-qwen-code -- 8 files
  at root (config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
  chat_template.jinja, model.safetensors 2.94 GB, README.md, .gitattributes);
  `private=false`, `used_storage ~ 2.96 GB`, commits `034d0c4` (config/tokenizer),
  `51098a1` (weights), `a60fa7a` (model-card README).
- GitHub: commit `4fb6412` pushed to github.com/eivintobias/heartly (master); the 2.94 GB
  weights stay git-ignored -- only code + docs are committed (16 files).
