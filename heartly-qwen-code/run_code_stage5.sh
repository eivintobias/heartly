#!/bin/bash
# Stage 5 training script for heartly-qwen-code
# Run on vast.ai RTX 3090 (24GB)
# Usage: bash run_code_stage5.sh

set -e

echo "=== Heartly Qwen-Code Stage 5 Training ==="
echo "Dataset: sft_dataset_code_v2.jsonl (7,261 samples)"
echo "Fixes: natural phrasing, conversational, single refusal, persona"

# Install deps
pip install -q peft bitsandbytes sentencepiece protobuf

# Train with QLoRA (works on 11GB, e.g. RTX 2080 Ti or 3090)
python finetune_qwen.py \
    --data sft_dataset_code_v2.jsonl \
    --out heartly-qwen-code-v2 \
    --epochs 2 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-length 512 \
    --lr 5e-5 \
    --qlora \
    --dtype fp16

echo "=== Training complete ==="
echo "Model saved to heartly-qwen-code-v2/"
echo ""
echo "Next steps:"
echo "1. Merge LoRA: python merge_and_convert.py"
echo "2. Convert to GGUF: python convert_hf_to_gguf.py heartly-qwen-code-v2-merged/ --outtype f16"
echo "3. Test in LM Studio with stop string: <stop>"
