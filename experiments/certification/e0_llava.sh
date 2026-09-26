#!/usr/bin/env bash
# Second verifier: LLaVA-1.5 (CLIP-L/336 fixed tower). Feeding 672 gives an exact 2x
# downscale, which is dyadic and therefore constructible.
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source experiments/setup/_stage.sh
stagger 8 20
MODEL="$(stage_model "$DATA_DIR/models/llava-hf__llava-1.5-7b-hf")"
echo "model staged at: $MODEL"
python - <<'PY'
import torch; print(f"gpu cuda={torch.cuda.is_available()} name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
PY
python experiments/certification/e0_llava.py --model "$MODEL" --out "$OUT" --seed "$TASK_ID"
