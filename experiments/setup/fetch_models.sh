#!/usr/bin/env bash
# Fetch a target verifier into the shared data dir.  One task per model repo.
#
#   $ITEM     HuggingFace repo id, e.g. Qwen/Qwen2.5-VL-7B-Instruct
#   $OUT      CSV shard: repo, dest, n_files, bytes, elapsed_s
#
# Everything lands under $DATA_DIR.
set -euo pipefail

DEST="$DATA_DIR/models/${ITEM//\//__}"   # org/name -> org__name, matching the dir's convention
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0   # plain requests: fewer moving parts on a login node

echo "exp=fetch_models item=$ITEM dest=$DEST"

python - "$ITEM" "$DEST" "$OUT" <<'PY'
import os, sys, time, csv, pathlib
from huggingface_hub import snapshot_download

repo, dest, out = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()

# Weights only; skip the duplicate formats some repos ship.
path = snapshot_download(
    repo_id=repo,
    local_dir=dest,
    ignore_patterns=["*.pth", "*.msgpack", "*.h5", "*.onnx", "original/*"],
    max_workers=4,
)

files = [p for p in pathlib.Path(path).rglob("*") if p.is_file()]
nbytes = sum(p.stat().st_size for p in files)
elapsed = time.time() - t0

pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["repo", "dest", "n_files", "bytes", "elapsed_s"])
    w.writerow([repo, path, len(files), nbytes, round(elapsed, 1)])

print(f"RESULT repo={repo} n_files={len(files)} gib={nbytes/2**30:.2f} "
      f"elapsed_s={elapsed:.1f} dest={path}")
PY
