#!/usr/bin/env bash
# E0b+E0c: certificate tested on the frozen verifier, with self and non-kernel controls.
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 8 20
MODEL="$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")"
echo "model staged at: $MODEL"
python - <<'PY'
import torch
print(f"gpu_check cuda_available={torch.cuda.is_available()} count={torch.cuda.device_count()} torch={torch.__version__}")
if torch.cuda.is_available(): print(f"gpu_name={torch.cuda.get_device_name(0)}")
PY
python experiments/certification/e0c_exact.py --model "$MODEL" --out "$OUT" --seed "$TASK_ID" --trials 4
