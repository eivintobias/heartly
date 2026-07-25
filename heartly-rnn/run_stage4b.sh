#!/bin/bash
# run_stage4b.sh — Stage 4b on a fresh vast.ai 3090 instance.
# Upload alongside: memory_store.py, stage4b_write_gate.py, stage4b_retrieval.py
# Usage (in tmux on the instance):
#   bash run_stage4b.sh 2>&1 | tee stage4b.log
set -e

# vast.ai base image: python env lives at /venv/main
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

echo "== pip install (pinned stack, same as Stage 3/3.5/4) =="
pip install -q "transformers==4.56.2" "flash-linear-attention==0.5.1" \
    sentence-transformers scikit-learn

mkdir -p stage4b_results

echo "== 0. store gate (should replicate the local 5/5) =="
python memory_store.py

echo "== 1. Part A: write-gate formats =="
python stage4b_write_gate.py 2>&1 | tee stage4b_results/write_gate.out

echo "== 2. Part B: retrieval + injection =="
python stage4b_retrieval.py 2>&1 | tee stage4b_results/retrieval.out

echo "== done — pull stage4b_results/ + stage4b.log home, then DESTROY the instance =="
ls -la stage4b_results/
