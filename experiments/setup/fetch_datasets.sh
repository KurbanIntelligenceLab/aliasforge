#!/usr/bin/env bash
# Fetch a benchmark into the shared data dir. One task per repo.
#   $ITEM  HuggingFace dataset repo id
#   $OUT   CSV shard: repo, dest, n_files, bytes, elapsed_s
set -euo pipefail
DEST="$DATA_DIR/datasets/${ITEM//\//__}"
export HF_HOME="$DATA_DIR/hf_cache"
echo "exp=fetch_datasets item=$ITEM dest=$DEST"
python - "$ITEM" "$DEST" "$OUT" <<'PY'
import sys, time, csv, pathlib
from huggingface_hub import snapshot_download
repo, dest, out = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, max_workers=4)
files = [p for p in pathlib.Path(path).rglob("*") if p.is_file()]
nbytes = sum(p.stat().st_size for p in files)
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["repo","dest","n_files","bytes","elapsed_s"])
    w.writerow([repo, path, len(files), nbytes, round(time.time()-t0,1)])
print(f"RESULT repo={repo} n_files={len(files)} gib={nbytes/2**30:.2f} elapsed_s={time.time()-t0:.1f}")
PY
