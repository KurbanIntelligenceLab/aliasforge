#!/usr/bin/env bash
# E3: both IAS routes, run in parallel. $ITEM is the route letter.
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 8 20
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
python experiments/routing/e3_ias.py --route "$ITEM" \
  --model "$MODEL" \
  --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" \
  --gains "${GAINS_GLOB:-results/routing/e21-pathmatched/sh_*.csv}" --out "$OUT" --seed 0
