#!/usr/bin/env bash
# E32: fill Table 2's empty cells. Processor only, CPU, no model weights.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
MODEL="$(stage_processor "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
python experiments/routing/e32_fill_table2.py \
  --model "$MODEL" --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" --out "$OUT"
