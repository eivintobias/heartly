#!/usr/bin/env bash
# Stage 2 on a vast.ai GPU instance: fine-tune RWKV on the Heartly grammar,
# then train the boundary head and run the say/sense measurement.
# Usage: bash run_stage2.sh   (inside tmux, so SSH drops don't kill it)
set -euo pipefail

echo "==> installing python deps (torch comes from the instance image)"
pip install -q --upgrade pip
pip install -q -r requirements-remote.txt

echo "==> sanity: files present"
test -f sft_dataset.jsonl || { echo "sft_dataset.jsonl missing"; exit 1; }
test -f probe_questions.jsonl || { echo "probe_questions.jsonl missing"; exit 1; }
nvidia-smi || true

echo "==> [1/2] fine-tune (2 epochs over 6031 samples; ~30-60 min on a 3090/4090)"
python finetune_rwkv.py --epochs 2 --out rwkv-heartly

echo "==> [2/2] boundary head + say/sense measurement"
python measure_say_sense.py --model rwkv-heartly --eval-limit 300 --report say_sense_report.json

echo "==> DONE. Grab: rwkv-heartly/  say_sense_report.json  probe_head.pkl"