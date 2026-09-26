#!/usr/bin/env bash
# E26: processor-level certification across deployed pipelines. CPU only, no weights:
# the processor's small files are staged node-locally and nothing else is read.
#   $ITEM  HuggingFace repo id (a line of configs/processors.txt); a line that
#          starts with # is the mapping comment and is skipped with a placeholder row
#   $OUT   CSV shard, one row per (variant, trial, arm)
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 8 5

case "$ITEM" in
  \#*|"")
    mkdir -p "$(dirname "$OUT")"
    python experiments/constructibility/e26_breadth.py --skip-comment --name "$ITEM" --out "$OUT"
    exit 0;;
esac

NAME="${ITEM//\//__}"
SRC="$DATA_DIR/processors/$NAME"
[ -d "$SRC" ] || SRC="$DATA_DIR/models/$NAME"     # the paper's own targets already live here
PROC="$(stage_processor "$SRC")"
echo "processor staged at: $PROC"
python experiments/constructibility/e26_breadth.py \
  --processor "$PROC" --name "$ITEM" --out "$OUT" --seed "$TASK_ID" --trials 3
