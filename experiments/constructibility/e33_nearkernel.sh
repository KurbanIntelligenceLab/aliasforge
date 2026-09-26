#!/usr/bin/env bash
# E33: collision portability of exact-kernel versus near-kernel perturbations. CPU only.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python experiments/constructibility/e33_nearkernel.py \
  --data "$DATA_DIR/datasets/OpenGVLab__VisualProcessBench" --out "$OUT"
