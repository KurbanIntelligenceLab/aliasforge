#!/usr/bin/env bash
# E16: certified construction on natural images. Preprocessing only -- no GPU, no weights.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
N_SHARDS=4
python experiments/natural_images/e16_natural.py \
  --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" \
  --out "$OUT" --task "$TASK_ID" --shards $N_SHARDS --limit 200 --amps 20,10,5,2
