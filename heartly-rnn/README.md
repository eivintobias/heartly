# heartly-rnn — Experiment 1: The Boundary Head on the State

**Question:** does a recurrent model's carried state — *before any answer is generated* —
encode whether the model knows the answer?

This is the first artifact of **Track 2** (see `research_papers/RESEARCH_PAPER_II_TRUE_BOUNDARY.md`,
§5.3 and Track 2 discussion): the absence sensor, built first as a probe over recurrent
state, compared against a transformer baseline.

---

## Hypothesis and falsifiers

**H:** A small probe trained on the recurrent state (or residual-stream hidden state, for
the transformer baseline) at the end-of-question position can predict known-vs-unknown
with AUROC meaningfully above chance.

**Falsifier 1 (sensor is dead):** AUROC ≈ 0.5 on all layers and all models → the state
does not encode knowability. The boundary-head idea is abandoned and the paper says so.

**Falsifier 2 (no recurrent advantage):** transformer baseline ≈ recurrent models →
Track 2 loses its main motivation; the sensor moves back to the transformer as a plain
probe.

**Decision rule:** proceed to Stage 2 (fine-tune the RNN with the Heartly grammar +
deploy the head at the `<verify>` position) only if recurrent-state probing is strong
(AUROC ≥ ~0.8) and preferably stronger than the baseline.

## Protocol

1. **Build the probe set** (`gen_probe_dataset.py`) — ~3k labeled questions:
   - *Known candidates:* SQuAD + TriviaQA (validation splits, short answers). These are
     corpus-known; `extract_states.py --verify-known` upgrades them to **model-verified**
     (greedy answer contains the gold string) — the paper's "model's true boundary" rule.
   - *Unknown:* five generators targeting genuine ignorance, disjoint from the
     75-prompt Heartly test suite: fabricated entities, type-aware attribute mismatch,
     post-cutoff events, depth-2 hyper-specifics, unanswerable-in-principle.
2. **Extract states** (`extract_states.py`) — for each model, run the raw prompt
   (`User: {q}\nAssistant: `) and dump per-layer features at the final prompt token:
   - `qwen` → residual-stream hidden states (transformer baseline; no recurrent state).
   - `mamba` → hidden states **and** `ssm_state` (the compressed recurrent memory).
   - `rwkv` → hidden states **and** the RWKV time/channel-mix carried state.
3. **Probe** (`train_probe.py`) — per-layer logistic probes (optionally MLP), stratified
   80/20 split, AUROC / accuracy / ECE, plus per-generator breakdown for the best layer.

## Usage

```bash
pip install -r requirements.txt

# 1. dataset (~3k questions, seed-fixed)
python gen_probe_dataset.py

# 2. extract (each downloads its model on first run)
python extract_states.py --family qwen      --repo Qwen/Qwen2.5-0.5B
python extract_states.py --family falcon_h1 --repo tiiuae/Falcon-H1-0.5B-Base
python extract_states.py --family rwkv     --repo RWKV/rwkv-4-world-430m --trust-remote-code
# add --verify-known to also greedy-check the known candidates per model
# NOTE: state-spaces/mamba-* repos ship no HF-format config — Falcon-H1 is the
# supported SSM-state path on transformers v5 (mamba2-based hybrid).

# 3. probe everything that landed in states/
python train_probe.py --verified-only
```

Model repos are CLI arguments — swap in bigger siblings (Falcon-H1-1.5B/3B, RWKV7-Goose
with `--trust-remote-code`) without code changes. Default layer mode stores quartile
layers (+ final) to keep `.npz` files manageable; `--layer-mode all` for the full stack.
Run 1 (2026-07-20, 2902 questions, CPU): qwen ~7 min, rwkv ~11 min, falcon_h1 ~2.4 h
(naive mamba path without CUDA kernels). Results: see RESULTS.md — recurrent-state
probes hit AUROC 1.000; the sensor hypothesis passed and Stage 2 is unlocked.

## File map

| File | Role |
|---|---|
| `gen_probe_dataset.py` | true-boundary question generator (mini v4 unknown mix) |
| `extract_states.py` | model-agnostic state dumper (qwen / mamba / rwkv) |
| `train_probe.py` | probe training + metrics + RESULTS.md writer |
| `RESULTS.md` | experiment log (appended per run) |
| `states/` | `.npz` state dumps (gitignored) |
| `probe_questions.jsonl` | the labeled question set |

## Stage map (where this leads)

1. **This experiment** — probe the base models' states.
2. Fine-tune RNN on the Heartly grammar/data; boundary head at `<verify>` reading state;
   say/sense disagreement alarm.
3. State persistence — save/load the recurrent state across sessions.
4. Supervised write-gate — trust-gating *inside* the state update.