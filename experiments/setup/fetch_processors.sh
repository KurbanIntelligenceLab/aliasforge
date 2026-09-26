#!/usr/bin/env bash
# Fetch ONLY a model's small files (configs, tokenizer, remote code) into the shared data
# dir, so E26 can certify its preprocessing without ever downloading weights.
#
#   $ITEM  HuggingFace repo id (a line of configs/processors.txt; a line that
#          starts with # is the mapping comment and is skipped with a placeholder row)
#   $OUT   CSV shard: repo, dest, ok, n_files, bytes, error
#
# Network job: NOT offline; it loads no weights, so check_io has nothing to guard here.
# A gated or missing repo never fails the task; the error is recorded in the shard.
set -euo pipefail
export HF_HOME="$DATA_DIR/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0   # a network job: never inherit an offline flag

case "$ITEM" in
  \#*|"")
    mkdir -p "$(dirname "$OUT")"
    printf 'repo,dest,ok,n_files,bytes,error\n"%s",,0,0,0,skipped: comment line\n' "$ITEM" > "$OUT"
    echo "RESULT repo=comment skipped=1"; exit 0;;
esac

DEST="$DATA_DIR/processors/${ITEM//\//__}"   # org/name -> org__name
echo "exp=fetch_processors item=$ITEM dest=$DEST"

python - "$ITEM" "$DEST" "$OUT" <<'PY'
import csv, pathlib, sys, time
repo, dest, out = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
ok, err, path, n, nbytes = 0, "", dest, 0, 0
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id=repo, local_dir=dest, max_workers=2,
        allow_patterns=["*.json", "*.txt", "*.model", "*.py", "tokenizer*", "merges*",
                        "vocab*", "*.tiktoken"],
        ignore_patterns=["*.safetensors", "*.bin", "*.pth", "*.pt", "*.gguf", "*.h5",
                         "*.msgpack", "*.onnx"])
    files = [p for p in pathlib.Path(path).rglob("*") if p.is_file() and ".cache" not in p.parts]
    big = [p for p in files if p.suffix in (".safetensors", ".bin", ".pth", ".pt", ".gguf")]
    n, nbytes = len(files), sum(p.stat().st_size for p in files)
    ok = int(n > 0 and not big)
    if big:
        err = f"weights slipped through the filter: {[p.name for p in big][:3]}"
    elif n == 0:
        err = "no files matched"
except Exception as exc:  # noqa: BLE001 - gated/missing/network: record, never fail the task
    err = f"{type(exc).__name__}: {str(exc)[:300]}".replace("\n", " ")
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["repo", "dest", "ok", "n_files", "bytes", "error"])
    w.writerow([repo, path, ok, n, nbytes, err])
print(f"RESULT repo={repo} ok={ok} n_files={n} kib={nbytes/1024:.0f} "
      f"elapsed_s={time.time()-t0:.1f} error=\"{err[:120]}\"")
PY
