#!/usr/bin/env bash
# E20b: certified lattice bounds for all four Pillow kernels + search-window sweep. CPU only,
# no weights, no dataset, nothing to stage. One shard per kernel (--count 4).
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 4 5
python experiments/constructibility/e20b_bounds.py --out "$OUT" --task "$TASK_ID" --shards 4
