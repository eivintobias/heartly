#!/bin/bash
# run_code_stage1.sh — Heartly Qwen-Code Stage 1 on a vast.ai 3090 instance.
#
# Upload alongside: finetune_qwen.py, train_probe_code.py, measure_say_sense_code.py,
#   sft_dataset_code.jsonl (pre-generated, or generate on the instance)
#
# Usage (in tmux on the instance):
#   bash run_code_stage1.sh 2>&1 | tee code_stage1.log
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vast.ai base image: python env lives at /venv/main
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

echo "== pip install =="
pip install -q --upgrade pip
pip install -q torch transformers datasets scikit-learn sentencepiece accelerate

echo "== sanity: files present =="
test -f sft_dataset_code.jsonl || { echo "sft_dataset_code.jsonl missing"; exit 1; }
nvidia-smi || true

mkdir -p code_stage1_results

echo "== [1/3] fine-tune Qwen2.5-Coder-1.5B on Heartly code SFT =="
python finetune_qwen.py \
  --repo Qwen/Qwen2.5-Coder-1.5B \
  --data sft_dataset_code.jsonl \
  --epochs 2 \
  --batch-size 4 --grad-accum 4 \
  --max-length 512 \
  --freeze-layers 12 \
  --dtype fp16 \
  --out heartly-qwen-code

echo "== [2/3] train boundary head probe =="
python train_probe_code.py \
  --model heartly-qwen-code \
  --data sft_dataset_code.jsonl \
  --eval-limit 500 \
  --head-out code_stage1_results/probe_head.pkl \
  --report code_stage1_results/probe_report.json

echo "== [3/3] measure grammar adoption + coding accuracy =="
python measure_say_sense_code.py \
  --model heartly-qwen-code \
  --report code_stage1_results/say_sense_report_code.json

echo "== done =="
echo "Pull these files home:"
echo "  heartly-qwen-code/  (model + tokenizer)"
echo "  code_stage1_results/"
echo "  code_stage1.log"
echo "Then DESTROY the instance."
ls -la code_stage1_results/