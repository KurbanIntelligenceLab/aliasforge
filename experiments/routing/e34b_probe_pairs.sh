#!/usr/bin/env bash
# E34b: strong router on the certified pairs, native and non-stop actions. GPU for the vision tower only.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
python experiments/routing/e34b_probe_pairs.py --model "$MODEL" --out "$OUT"
