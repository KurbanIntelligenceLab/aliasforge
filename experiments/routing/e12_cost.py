#!/usr/bin/env python3
"""E12: measure what each action actually COSTS, per action.

The paper's objective is g_i(a) = Delta_i(a) - lambda*c(a).  Every reported quantity is
currently at lambda=0, because c has never been measured: E1 logged wall-clock and decode
tokens SUMMED over the four actions, which cannot be split back apart.  So the cost term in
the framework is declared and never used.

This measures c directly.  Cost is a property of the ACTION, not of the item, so a few
hundred items give a tight mean; the expensive part of E1 (two sampled replicates over 1280
items to estimate Delta) is not needed here and is not repeated.  Emits one row per
(item, replicate, action) so the cost vector can be recomputed at any normalisation, and so
the lambda-Pareto frontier can be traced from records rather than asserted.

Output: item-level rows -> $OUT.  Also prints the mean cost vector, which is the number the
paper reports.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.actions import ActionRunner, gains  # noqa: E402
from aliasforge.items import ImageStore, load_items  # noqa: E402
from aliasforge.verifier import Verifier  # noqa: E402

VERSION = "e12.1"
ACTIONS = ["stop", "att", "rea", "enc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    a = ap.parse_args()

    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0))
             if i % a.shards == a.task]
    store = ImageStore(a.data)
    v = Verifier(a.model, max_pixels=a.max_pixels)
    # identical configuration to E1, so the costs describe the runs the paper reports
    runner = ActionRunner(v, rea_passes=3, sampled=True, temperature=0.8, seed=a.task)

    print(f"exp=e12_cost version={VERSION} task={a.task}/{a.shards} items={len(items)} "
          f"reps={a.reps} actions={','.join(ACTIONS)} model={os.path.basename(a.model)}")

    rows = [("key", "record", "rep", "action", "prefill_tokens", "decode_tokens", "wall_s")]
    acc = {k: {"pre": [], "dec": [], "wall": []} for k in ACTIONS}
    t0 = time.time()
    for n, it in enumerate(items):
        try:
            img = np.asarray(store.load(it.image_path))
        except Exception:
            continue
        H, W = img.shape[:2]
        box = (H // 4, W // 4, H // 2, W // 2)
        for rep in range(a.reps):
            res = runner.run_all(img, it.conditioned_question, it.step, box)
            for k in ACTIONS:
                r = res[k]
                rows.append((it.key, it.key.split(":")[0], rep, k,
                             r.prefill_tokens, r.decode_tokens, round(r.wall_s, 4)))
                acc[k]["pre"].append(r.prefill_tokens)
                acc[k]["dec"].append(r.decode_tokens)
                acc[k]["wall"].append(r.wall_s)
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            dec = {k: np.mean(acc[k]["dec"]) for k in ACTIONS}
            print(f"  progress {n+1}/{len(items)} elapsed_s={el:.0f} "
                  f"per_item_s={el/(n+1):.1f} mean_decode=" +
                  " ".join(f"{k}={dec[k]:.0f}" for k in ACTIONS))

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

    # the cost vector the paper reports: decode tokens, normalised to one critique (att)
    dec = {k: float(np.mean(acc[k]["dec"])) if acc[k]["dec"] else 0.0 for k in ACTIONS}
    wall = {k: float(np.mean(acc[k]["wall"])) if acc[k]["wall"] else 0.0 for k in ACTIONS}
    unit = dec["att"] if dec["att"] else 1.0
    cvec = {k: dec[k] / unit for k in ACTIONS}
    print("RESULT task=%d n_items=%d " % (a.task, len(items)) +
          " ".join(f"decode_{k}={dec[k]:.1f}" for k in ACTIONS) + " " +
          " ".join(f"wall_{k}={wall[k]:.3f}" for k in ACTIONS) + " " +
          " ".join(f"c_{k}={cvec[k]:.3f}" for k in ACTIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
