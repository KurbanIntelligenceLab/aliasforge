#!/usr/bin/env bash
# E26: per-field interface schema + certificate re-check under the SCORED prompt.
# Processor only (configs/tokenizer), CPU, no weights. TASK_ID 0 = Qwen2.5-VL, 1 = LLaVA-1.5.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 2 5
case "$TASK_ID" in
  0) TARGET=qwen;  SRC="$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct" ;;
  1) TARGET=llava; SRC="$DATA_DIR/models/llava-hf__llava-1.5-7b-hf" ;;
  *) echo "e26_fields: unexpected TASK_ID=$TASK_ID (expects 0 or 1)"; exit 1 ;;
esac
PROC="$(stage_processor "$SRC")"
echo "processor staged at: $PROC"
python experiments/certification/e26_fields.py --model-dir "$PROC" --target "$TARGET" --out "$OUT" --trials 5
