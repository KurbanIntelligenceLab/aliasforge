#!/usr/bin/env bash
# E1 prerequisite: Monte Carlo power calculation fixing E1's pre-registered sample size.
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python experiments/routing/power_e1.py --out "$OUT" --seed "$TASK_ID" --reps 200
