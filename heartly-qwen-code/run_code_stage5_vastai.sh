#!/bin/bash
# run_code_stage5_vastai.sh — Heartly Qwen-Code Stage 5 on vast.ai 3090.
#
# Upload these 3 files alongside this script:
#   1. finetune_qwen.py
#   2. sft_dataset_code_v3.jsonl
#   3. run_code_stage5_vastai.sh (this file)
#
# Usage (in tmux on the instance):
#   bash run_code_stage5_vastai.sh 2>&1 | tee code_stage5.log
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vast.ai base image: python env lives at /venv/main
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

echo "============================================"
echo "  Heartly Qwen-Code Stage 5 — Conversational SFT"
echo "============================================"
echo ""

echo "== [0/4] Install dependencies =="
pip install -q --upgrade pip
pip install -q torch transformers datasets scikit-learn sentencepiece accelerate

echo ""
echo "== [1/4] Sanity check =="
test -f sft_dataset_code_v3.jsonl || { echo "ERROR: sft_dataset_code_v3.jsonl missing"; exit 1; }
test -f finetune_qwen.py || { echo "ERROR: finetune_qwen.py missing"; exit 1; }
nvidia-smi
echo ""
echo "Dataset size:"
wc -l sft_dataset_code_v3.jsonl
echo ""

echo "== [2/4] Fine-tune Qwen2.5-Coder-1.5B on Heartly v3 SFT =="
echo "  Full fine-tune (not QLoRA — 24GB is enough)"
echo "  5,200 samples, 2 epochs, batch 4, grad-accum 4, max-length 512"
echo "  Expected: ~35 min on RTX 3090"
echo ""
python finetune_qwen.py \
  --repo Qwen/Qwen2.5-Coder-1.5B \
  --data sft_dataset_code_v3.jsonl \
  --epochs 2 \
  --batch-size 4 --grad-accum 4 \
  --max-length 512 \
  --freeze-layers 12 \
  --dtype fp16 \
  --out heartly-qwen-code-v3

echo ""
echo "== [3/4] Verify model saved =="
ls -la heartly-qwen-code-v3/
echo ""

echo "== [4/4] Done! =="
echo ""
echo "============================================"
echo "  TRAINING COMPLETE"
echo "============================================"
echo ""
echo "Pull these files home:"
echo "  scp -P <PORT> user@<HOST>:~/heartly-qwen-code-v3/* ./heartly-qwen-code-v3/"
echo "  scp -P <PORT> user@<HOST>:~/code_stage5.log ."
echo ""
echo "Then DESTROY this instance!"
echo ""
echo "Next steps (on your laptop):"
echo "  1. Add heartly_stop_token_id to config.json:"
echo "     python -c \"import json; p='heartly-qwen-code-v3/config.json'; d=json.load(open(p)); d['heartly_stop_token_id']=9495; json.dump(d,open(p,'w'),indent=2)\""
echo "  2. Convert to GGUF:"
echo "     cd llama.cpp && python convert_hf_to_gguf.py ../heartly-qwen-code/heartly-qwen-code-v3 --outfile ../heartly-qwen-code/heartly-qwen-code-v3.gguf --outtype f16"
echo "  3. Test in LM Studio or:"
echo "     python chat_clean.py --model heartly-qwen-code-v3.gguf --gpu 0"
echo ""