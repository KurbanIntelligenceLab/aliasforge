#!/usr/bin/env bash
# E28: certified alias on VisualPRM-8B's deployed (dynamic_preprocess + thumbnail) path, on the
# sound integer-kernel lattice. CPU only, no weights, no dataset, nothing to stage. Sides are
# sharded over --count 4.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 4 5
python experiments/certification/e28_visualprm.py --out "$OUT" --task "$TASK_ID" --shards 4 \
  --workers "${SLURM_CPUS_PER_TASK:-2}"
