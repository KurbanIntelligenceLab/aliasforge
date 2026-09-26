#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 8 20
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
N_SHARDS=6
python experiments/routing/e19_placebo.py \
  --model "$MODEL" \
  --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" \
  --out "$OUT" --task "$TASK_ID" --shards $N_SHARDS --limit 384 --reps 2
