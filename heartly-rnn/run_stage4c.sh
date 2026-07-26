#!/bin/bash
# run_stage4c.sh — Stage 4c on a fresh vast.ai 3090 instance.
# Upload alongside: memory_store.py, stage4c_write_gate.py, stage4c_retrieval.py,
#   finetune_rwkv.py, measure_say_sense.py, extract_states.py, train_probe.py,
#   sft_dataset_v2.jsonl, probe_questions.jsonl, gen_probe_dataset.py,
#   render_sft_dataset.py, render_sft_dataset_v2.py
# Usage (in tmux on the instance):
#   bash run_stage4c.sh 2>&1 | tee stage4c.log
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vast.ai base image: python env lives at /venv/main
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

echo "== pip install (pinned stack, same as Stage 3/3.5/4/4b) =="
pip install -q --upgrade pip
pip install -q -r requirements-remote.txt
pip install -q sentence-transformers

echo "== sanity: files present =="
test -f sft_dataset_v2.jsonl || { echo "sft_dataset_v2.jsonl missing"; exit 1; }
test -f probe_questions.jsonl || { echo "probe_questions.jsonl missing"; exit 1; }
nvidia-smi || true

mkdir -p stage4c_results

echo "== [1/4] fine-tune RWKV7-1.5B on v2 dataset (2 epochs, ~8150 samples) =="
python finetune_rwkv.py \
  --repo RWKV/RWKV7-Goose-World3-1.5B-HF \
  --data sft_dataset_v2.jsonl \
  --epochs 2 \
  --batch-size 4 --grad-accum 4 \
  --max-length 256 \
  --freeze-layers 16 \
  --dtype bf16 \
  --out rwkv7-heartly-v2

echo "== [2/4] re-verify decide/grammar (measure_say_sense) =="
python measure_say_sense.py \
  --model rwkv7-heartly-v2 \
  --eval-limit 300 \
  --head-out stage4c_results/probe_head_v2.pkl \
  --report stage4c_results/say_sense_report_v2.json

echo "== [3/4] Part A: write-gate formats (same as Stage 4b, new model) =="
python stage4c_write_gate.py 2>&1 | tee stage4c_results/write_gate.out

echo "== [4/4] Part B: retrieval + context injection (same as Stage 4b, new model) =="
python stage4c_retrieval.py 2>&1 | tee stage4c_results/retrieval.out

echo "== done — pull stage4c_results/ + rwkv7-heartly-v2/ + stage4c.log home, then DESTROY the instance =="
ls -la stage4c_results/
