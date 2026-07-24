#!/usr/bin/env bash
# one-off diagnostics for the stage3 run (quoting-safe: runs ON the instance)
echo '=== SECTIONS 5-6 ==='
sed -n '/== 5. attention/,/SMOKE TEST DONE/p' /workspace/stage3/run.log | grep -v '^$' | head -30
echo '=== REGEX TEST ==='
python3 - <<'EOF'
import re
pat = re.compile(r"(?:blocks|layers)\.(\d+)\.")
print("match on model.layers.0.pre_norm.weight:", bool(pat.search("model.layers.0.pre_norm.weight")))
EOF
echo '=== FILE ON DISK (line 46) ==='
sed -n '46p' /workspace/stage3/finetune_rwkv.py
echo '=== TRAIN TAIL ==='
tail -5 /workspace/stage3/run.log
echo '=== GPU ==='
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader