#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python experiments/natural_images/e24_native.py --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" --out "$OUT" --limit 200
