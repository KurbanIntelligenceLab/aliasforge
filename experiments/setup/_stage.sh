#!/usr/bin/env bash
# Stage a model directory onto NODE-LOCAL storage before loading it.
#
# Why: the project filesystem is shared, and an array of N tasks each opening the same
# multi-GB safetensors shards puts N concurrent readers (and their many internal threads)
# on one back-end file. Staging turns N reads of the
# shared file into at most one read per NODE, after which every task on that node reads
# from local disk.
#
# Guarantees:
#   - one copy per node, serialised by flock, so co-located tasks do not duplicate it;
#   - a .complete sentinel, so a task that arrives mid-copy waits rather than reading a
#     half-written directory;
#   - falls back through node-local disk before /dev/shm, because a 7B model in RAM costs
#     ~15 GB of the node's memory.
#
# Usage:  MODEL=$(stage_model "$DATA_DIR/models/Qwen__Qwen2.5-VL-7B-Instruct")

stage_model() {
  local src="$1" name base dst lock
  name="$(basename "$src")"
  base=""
  for cand in "${SLURM_TMPDIR:-}" "${TMPDIR:-}" /lscratch /tmp /dev/shm; do
    if [ -n "$cand" ] && [ -d "$cand" ] && [ -w "$cand" ]; then base="$cand"; break; fi
  done
  if [ -z "$base" ]; then echo "$src"; return 0; fi   # no local store: use the original

  dst="$base/aliasforge-stage/$name"
  mkdir -p "$(dirname "$dst")" 2>/dev/null || { echo "$src"; return 0; }
  lock="$(dirname "$dst")/.${name}.lock"

  if ! command -v flock >/dev/null 2>&1; then
    # No flock on this node: fall back to the shared copy rather than risk two tasks
    # writing the same staging directory at once.
    echo "$src"; return 0
  fi
  exec 9>"$lock" || { echo "$src"; return 0; }
  flock 9
  if [ ! -f "$dst/.stage_complete" ]; then
    rm -rf "$dst"
    mkdir -p "$dst"
    # single sequential copy; no parallel fan-out against the shared filesystem
    if cp -r "$src"/. "$dst"/ 2>/dev/null; then
      touch "$dst/.stage_complete"
    else
      rm -rf "$dst"; flock -u 9; echo "$src"; return 0    # fall back, never fail the job
    fi
  fi
  flock -u 9
  echo "$dst"
}

# Stage only a processor's SMALL files (configs, tokenizer, remote code) for CPU-only
# certification jobs that never load weights. Same lock + sentinel discipline as
# stage_model, but the copy is kilobytes, so an 8-shard array costs the shared
# filesystem nothing. Symlinks (hf_cache snapshots) are dereferenced.
#
# Usage:  PROC=$(stage_processor "$DATA_DIR/models/llava-hf__llava-1.5-7b-hf")
stage_processor() {
  local src="$1" name base dst lock
  name="$(basename "$src")"
  base=""
  for cand in "${SLURM_TMPDIR:-}" "${TMPDIR:-}" /lscratch /tmp /dev/shm; do
    if [ -n "$cand" ] && [ -d "$cand" ] && [ -w "$cand" ]; then base="$cand"; break; fi
  done
  if [ -z "$base" ]; then echo "$src"; return 0; fi
  dst="$base/aliasforge-stage/proc-$name"
  mkdir -p "$(dirname "$dst")" 2>/dev/null || { echo "$src"; return 0; }
  lock="$(dirname "$dst")/.proc-${name}.lock"
  if ! command -v flock >/dev/null 2>&1; then echo "$src"; return 0; fi
  exec 8>"$lock" || { echo "$src"; return 0; }
  flock 8
  if [ ! -f "$dst/.stage_complete" ]; then
    rm -rf "$dst"; mkdir -p "$dst"
    if ( cd "$src" && find . -maxdepth 2 \( -type f -o -type l \) \( -name '*.json' -o -name '*.txt' -o -name '*.model' \
           -o -name '*.py' -o -name 'tokenizer*' -o -name '*.tiktoken' -o -name 'merges*' -o -name 'vocab*' \) \
           -print0 | while IFS= read -r -d '' f; do
             mkdir -p "$dst/$(dirname "$f")" && cp -L "$f" "$dst/$f"; done ); then
      touch "$dst/.stage_complete"
    else
      rm -rf "$dst"; flock -u 8; echo "$src"; return 0
    fi
  fi
  flock -u 8
  echo "$dst"
}

# Spread the start of an array so tasks do not open the same shards in the same instant.
# Cheap insurance even with staging, since the first task on each node still reads it.
stagger() {
  local n="${1:-8}" step="${2:-20}"
  sleep $(( (${TASK_ID:-0} % n) * step ))
}
