#!/usr/bin/env bash
# Stage 3 RESUME (2026-07-23): first attempt crashed on transformers v5
# incompatibilities with fla 0.5.1. This script pins transformers 4.56.2,
# trains RWKV7-1.5B in bf16 (fla kernel requirement), fused CE disabled,
# then runs boundary head + say/sense.
# Usage: bash run_stage3b.sh   (inside tmux)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==> pinning transformers==4.56.2 (fla 0.5.1 compat)"
pip install -q "transformers==4.56.2"
python -c "import transformers, fla, torch; print('transformers', transformers.__version__, '| fla', fla.__version__, '| torch', torch.__version__)"

echo "==> sanity: files present"
test -f sft_dataset.jsonl || { echo "sft_dataset.jsonl missing"; exit 1; }
test -f probe_questions.jsonl || { echo "probe_questions.jsonl missing"; exit 1; }

echo "==> [1/2] fine-tune RWKV7-1.5B (bf16, 2 epochs over 6031 samples)"
python finetune_rwkv.py \
  --repo RWKV/RWKV7-Goose-World3-1.5B-HF \
  --epochs 2 \
  --batch-size 4 --grad-accum 4 \
  --max-length 256 \
  --freeze-layers 16 \
  --dtype bf16 \
  --out rwkv7-heartly

echo "==> [2/2] boundary head + say/sense measurement"
python measure_say_sense.py \
  --model rwkv7-heartly \
  --eval-limit 300 \
  --head-out probe_head_rwkv7.pkl \
  --report say_sense_report_rwkv7.json

echo "==> DONE. Grab: rwkv7-heartly/  say_sense_report_rwkv7.json  probe_head_rwkv7.pkl"