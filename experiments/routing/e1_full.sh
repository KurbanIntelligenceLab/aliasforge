#!/usr/bin/env bash
# E1: measure Delta_i(a) over the powered stream. One shard per task.
#   $TASK_ID  shard index; N_SHARDS below must match the job's --count
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 8 20
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
N_SHARDS=8
python experiments/routing/e1_full.py \
  --model "$MODEL" \
  --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" \
  --out "$OUT" --seed "$TASK_ID" --shard "$TASK_ID" --n-shards "$N_SHARDS" --n-items 1280
