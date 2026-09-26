#!/usr/bin/env bash
# E28c: certified pairs on VisualPRM-8B's DEPLOYED tiling path, CPU only (the transcribed
# preprocessing, no weights): the pair-construction half of e28b with 10 trials per seed.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 2 5
python experiments/certification/e28b_visualprm_logit.py --no-model --trials 10 --task "$TASK_ID" --shards 2 \
  --workers "${SLURM_CPUS_PER_TASK:-4}" --out "$OUT"
