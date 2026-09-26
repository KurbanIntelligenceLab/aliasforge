#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python experiments/natural_images/e25_padded.py --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" \
  --out "$OUT" --task "$TASK_ID" --shards 4 --limit 200 --amp 20
