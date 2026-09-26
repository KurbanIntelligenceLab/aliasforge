#!/usr/bin/env bash
# E28b: logit-gap arm for E28's certified alias on VisualPRM-8B's deployed path. Loads the
# 8B model once per task (bf16, one GPU); the model is staged node-locally. Seeds are sharded
# over --count 2 (seed = TASK_ID, 5 trials each). Resources: --gpus 1 --mem 32G --cpus 4 --time 1:30:00.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source experiments/setup/_stage.sh
stagger 2 20
MODEL="$(stage_model "$DATA_DIR/models/OpenGVLab__VisualPRM-8B")"
echo "model staged at: $MODEL"
N_SHARDS=2
python experiments/certification/e28b_visualprm_logit.py \
  --model "$MODEL" \
  --out "$OUT" --task "$TASK_ID" --shards $N_SHARDS --trials 5 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
