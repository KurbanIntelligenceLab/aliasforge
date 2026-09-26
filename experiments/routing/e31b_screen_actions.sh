#!/usr/bin/env bash
# E31: measure the dominated-selection column on certified pairs with the real verifier.
# One GPU task per seed; the model is staged node-local.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 2 20
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
N_SHARDS=2
python experiments/routing/e31b_screen_actions.py \
  --model "$MODEL" --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" --out "$OUT" --seed "$TASK_ID" --n-pairs 40
