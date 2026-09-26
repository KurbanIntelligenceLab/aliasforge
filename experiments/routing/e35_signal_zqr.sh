#!/usr/bin/env bash
# E35: decision-position hidden state, a signal of the full interface state (z, Q, R). GPU, model staged node-locally.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PWD}/experiments/routing:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
python experiments/routing/e35_signal_zqr.py \
  --model "$MODEL" --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" --out "$OUT" ${E35_ARGS:-}
