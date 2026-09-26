#!/usr/bin/env bash
# E10b: the constructibility screen recomputed on the true integer kernel lattice. CPU only,
# no weights, no dataset, nothing to stage. Unique sizes sharded round-robin (--count 4).
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 4 5
python experiments/constructibility/e10b_screen.py \
  --configs configs/resize_configs.txt \
  --out "$OUT" --task "$TASK_ID" --shards 4
