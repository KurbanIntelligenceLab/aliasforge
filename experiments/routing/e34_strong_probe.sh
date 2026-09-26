#!/usr/bin/env bash
# E34: strong Phi-measurable probe (vision-tower features -> cross-fitted router). GPU, model staged node-locally.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
python experiments/routing/e34_strong_probe.py \
  --model "$MODEL" --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" --out "$OUT"
