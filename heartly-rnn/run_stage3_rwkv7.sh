#!/usr/bin/env bash
# Stage 3 on a vast.ai GPU instance: fine-tune RWKV7-Goose-1.5B on the Heartly
# grammar, then train the boundary head and run say/sense — same recipe as
# Stage 2 (run_stage2.sh) but at 1.5B with the fla-backed RWKV7.
# Usage: bash run_stage3_rwkv7.sh   (inside tmux, so SSH drops don't kill it)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==> installing python deps (torch comes from the instance image)"
pip install -q --upgrade pip
pip install -q -r requirements-remote.txt

echo "==> sanity: files present"
test -f sft_dataset.jsonl || { echo "sft_dataset.jsonl missing"; exit 1; }
test -f probe_questions.jsonl || { echo "probe_questions.jsonl missing"; exit 1; }
nvidia-smi || true

echo "==> [0/2] SMOKE TEST (log: smoke_rwkv7.log)"
python smoke_rwkv7.py 2>&1 | tee smoke_rwkv7.log

echo "==> [1/2] fine-tune RWKV7-1.5B (2 epochs over 6031 samples)"
python finetune_rwkv.py \
  --repo RWKV/RWKV7-Goose-World3-1.5B-HF \
  --epochs 2 \
  --batch-size 4 --grad-accum 4 \
  --max-length 256 \
  --freeze-layers 16 \
  --out rwkv7-heartly

echo "==> [2/2] boundary head + say/sense measurement"
python measure_say_sense.py \
  --model rwkv7-heartly \
  --eval-limit 300 \
  --head-out probe_head_rwkv7.pkl \
  --report say_sense_report_rwkv7.json

echo "==> DONE. Grab: rwkv7-heartly/  say_sense_report_rwkv7.json  probe_head_rwkv7.pkl  smoke_rwkv7.log"