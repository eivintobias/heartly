#!/usr/bin/env bash
# run_stage4.sh — Stage 4 gist save/load. Runs ON the vast.ai instance.
# Stack expected (same as Stage 3/3.5): transformers 4.56.2 + fla 0.5.1 +
# triton, torch bf16-capable GPU. New base image: python env at /venv/main.
set -e
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
cd /workspace/heartly-rnn
mkdir -p stage4_results
python -u stage4_gist.py 2>&1 | tee stage4_results/stage4.log