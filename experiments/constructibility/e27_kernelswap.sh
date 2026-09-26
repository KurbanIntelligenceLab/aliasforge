#!/usr/bin/env bash
# E27: cost of the Lanczos defence on the frozen verifier + collision check. Needs the verifier.
# One model per task (staged node-locally), two processors (bicubic / Lanczos).
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 4 20
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
N_SHARDS=4
python experiments/constructibility/e27_kernelswap.py \
  --model "$MODEL" \
  --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" \
  --out "$OUT" --task "$TASK_ID" --shards $N_SHARDS --limit 1280 ${E27_BACKEND:+--backend "$E27_BACKEND"}
