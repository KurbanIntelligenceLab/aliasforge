#!/usr/bin/env bash
# E2: probe quality, with the certified set as a falsifiable chance-anchor.
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 8 20
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
python experiments/routing/e2_probe.py \
  --model "$MODEL" \
  --out "$OUT" --seed "$TASK_ID" --n-base 60
