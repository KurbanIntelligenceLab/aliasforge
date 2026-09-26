#!/usr/bin/env python3
"""E14: the crop-oracle bound, corrected for the winner's curse.

E11 ran a grid of crops per item and took the per-item MAXIMUM gain, reporting +0.055
against a blind-centre-crop baseline of -0.031.  That maximum is selected on the same
measurement it is scored on, so it inherits the optimizer's curse exactly as the plug-in
routing value did (0.090 -> 0.057 there).  The paper therefore has to call +0.055 "an upper
bound of unknown tightness" and draw no conclusion from it, which wastes the result.

E11 also ran the verifier deterministically (sampled=False), so its two passes would have
been identical and no split-sample correction was possible from its records.

This re-runs the grid with two INDEPENDENT sampled replicates and dumps the gain of EVERY
crop on EVERY replicate.  That makes the honest estimator computable: choose the crop on
replicate A, score it on replicate B.  The difference between that and the plug-in maximum
is the curse, measured rather than assumed.

Why it matters beyond E11: an oracle over the crop family upper-bounds what ANY crop
selection policy could achieve, question-conditioned or attention-based, so the corrected
number also bounds the strongest objection to the enc arm - that a better crop policy would
have rescued it.

Output: one row per (item, replicate, crop) -> $OUT.
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

VERSION = "e14.1"


def crop_grid(H, W, k):
    """Identical geometry to E11 so the two runs are comparable: index 0 is the
    deployable half-frame centre crop, then a k x k grid of half-frame windows."""
    boxes = [(H // 4, W // 4, H // 2, W // 2)]
    h, w = H // 2, W // 2
    for i in range(k):
        for j in range(k):
            top = 0 if k == 1 else round(i * (H - h) / (k - 1))
            left = 0 if k == 1 else round(j * (W - w) / (k - 1))
            boxes.append((top, left, h, w))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--limit", type=int, default=1280)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    a = ap.parse_args()

    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0))
             if i % a.shards == a.task]
    store = ImageStore(a.data)
    v = Verifier(a.model, max_pixels=a.max_pixels)
    # sampled=True is the whole point: E11's deterministic runner made the two
    # replicates identical, which is why its records cannot support a correction.
    runner = ActionRunner(v, rea_passes=3, sampled=True, temperature=0.8, seed=a.task)

    n_crops = a.grid * a.grid + 1
    print(f"exp=e14_crop2 version={VERSION} task={a.task}/{a.shards} items={len(items)} "
          f"crops={n_crops} reps={a.reps} sampled=True model={os.path.basename(a.model)}")

    rows = [("key", "record", "label", "rep", "crop_idx", "g_enc")]
    plug, split, blind = [], [], []
    t0 = time.time()
    for n, it in enumerate(items):
        try:
            img = np.asarray(store.load(it.image_path))
        except Exception:
            continue
        H, W = img.shape[:2]
        boxes = crop_grid(H, W, a.grid)
        G = np.zeros((a.reps, len(boxes)))
        for rep in range(a.reps):
            for ci, box in enumerate(boxes):
                res = {"stop": runner.stop(img, it.conditioned_question, it.step),
                       "enc": runner.enc(img, it.conditioned_question, it.step, box)}
                g = gains(res, it.label)["enc"]
                G[rep, ci] = g
                rows.append((it.key, it.key.split(":")[0], it.label, rep, ci, round(g, 6)))
        # plug-in: choose and score on the same replicate.  split-sample: choose on A,
        # score on B -- the honest one.
        plug.append(G[1].max())
        split.append(G[1][int(G[0].argmax())])
        blind.append(G[1][0])
        if (n + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  progress {n+1}/{len(items)} elapsed_s={el:.0f} "
                  f"per_item_s={el/(n+1):.1f} eta_s={el/(n+1)*(len(items)-n-1):.0f} "
                  f"blind={np.mean(blind):+.4f} plug={np.mean(plug):+.4f} "
                  f"split={np.mean(split):+.4f}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

    mb, mp, ms = float(np.mean(blind)), float(np.mean(plug)), float(np.mean(split))
    print(f"RESULT task={a.task} items={len(blind)} crops={n_crops} "
          f"mean_blind={mb:+.4f} mean_plugin_oracle={mp:+.4f} mean_splitsample={ms:+.4f} "
          f"curse={mp-ms:+.4f} uplift_corrected={ms-mb:+.4f} still_negative={ms < 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
