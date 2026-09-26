#!/usr/bin/env python3
"""E18: push certified aliases on UNMODIFIED photographs as far as the algebra allows.

E17 reached 41% of natural images at amplitude 5 without touching the photograph, by
searching 103 carrier offsets and keeping only the rows with intensity headroom.  Two
restrictions survived from the original code and neither is required by the mathematics:

  1. THE MARKS ARE PINNED.  Their vertical positions are still the literal constants
     0.13, 0.38, 0.63 from the synthetic demo.  Nothing needs them there.  The atomic fact
     is "how many marks", so the heights are free, and a mark can slide to wherever the
     image has room.  This is ~800 positions per carrier instead of one.

  2. ONE CARRIER PER MARK.  Every row of a mark uses the same kernel vector.  But each row
     is perturbed independently, and a sum of kernel vectors is a kernel vector, so
     different row-groups within a mark may use different carriers at no cost to the
     certificate -- only to the mark's visual tidiness.

Three arms, in increasing order of how much visual coherence they trade for yield:

  aligned  one carrier for all three marks, heights free.  The marks form a single column
           of ticks, which is the tidiest thing to show in a figure.
  free     each mark picks its own carrier and height independently.
  perrow   each ROW picks whichever carrier fits it.  Maximum yield, least tidy; reported
           to bound what is achievable, not as the recommended construction.

Scoring is vectorised: for each (carrier, sign) the per-row feasibility is computed once for
the whole image, then a sliding window gives every candidate height at once.

Preprocessing only: no GPU, no model weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import integer_kernel_vector, pil_coeffs  # noqa: E402
from aliasforge.items import ImageStore, load_items  # noqa: E402

VERSION = "e18.1"


def carriers(side, target, width=64, step=4, max_abs=8):
    K = pil_coeffs(side, target)
    out = []
    for c0 in range(0, max(1, side - width), step):
        v = integer_kernel_vector(K, c0, width)
        if v is not None and 0 < int(np.abs(v).max()) <= max_abs:
            out.append(v)
    return out


def row_feasible(base, v, amp, sign):
    """Boolean per row: can this row carry sign*amp*v without clipping?"""
    nz = np.flatnonzero(v)
    d = (sign * amp * v)[nz]
    out = base[:, nz].astype(np.int64) + d[None, :, None]
    return ((out >= 0) & (out <= 255)).all(axis=(1, 2))


def windows(feas, T):
    """Fraction of feasible rows for a mark of height T starting at each row."""
    c = np.concatenate([[0], np.cumsum(feas.astype(np.int64))])
    return (c[T:] - c[:-T]) / float(T)


def pick_marks(feas_all, T, k, need, min_gap):
    """Greedily choose k non-overlapping (carrier_index, start_row) with the best coverage."""
    scores = np.stack([windows(f, T) for f in feas_all])      # (n_cand, n_starts)
    chosen, banned = [], np.zeros(scores.shape[1], bool)
    for _ in range(k):
        s = scores.copy()
        s[:, banned] = -1.0
        ci, yi = np.unravel_index(int(np.argmax(s)), s.shape)
        if s[ci, yi] < need:
            return None
        chosen.append((ci, int(yi), float(s[ci, yi])))
        lo = max(0, yi - T - min_gap)
        hi = min(scores.shape[1], yi + T + min_gap)
        banned[lo:hi] = True
    return chosen


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build(base, plan, target):
    """plan: list of (rows_index_array, add_vector). First 3 -> xa, first 2 -> xb."""
    from PIL import Image
    xa = base.astype(np.int64).copy()
    xb = base.astype(np.int64).copy()
    for i, (rows, add) in enumerate(plan):
        if i < 3:
            xa[rows] += add
        if i < 2:
            xb[rows] += add
    if min(xa.min(), xb.min()) < 0 or max(xa.max(), xb.max()) > 255:
        return None
    xa, xb = xa.astype(np.uint8), xb.astype(np.uint8)
    da = digest(np.asarray(Image.fromarray(xa).resize((target, target), Image.BICUBIC)))
    db = digest(np.asarray(Image.fromarray(xb).resize((target, target), Image.BICUBIC)))
    return int(da == db), int((xa != xb).any(axis=2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--side", type=int, default=896)
    ap.add_argument("--target", type=int, default=448)
    ap.add_argument("--amps", default="20,10,5")
    ap.add_argument("--need", type=float, default=0.90)
    ap.add_argument("--thick", type=float, default=0.12)
    a = ap.parse_args()

    amps = [int(x) for x in a.amps.split(",")]
    cands = carriers(a.side, a.target)
    if not cands:
        print("RESULT no kernel vectors"); return 1
    signed = [(v, s) for v in cands for s in (1, -1)]
    T = int(a.thick * a.side)

    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0))
             if i % a.shards == a.task]
    store = ImageStore(a.data)
    print(f"exp=e18_freeplace version={VERSION} task={a.task}/{a.shards} items={len(items)} "
          f"carriers={len(cands)} signed={len(signed)} mark_height={T} need={a.need} amps={amps}")

    from PIL import Image
    ARMS = ("aligned", "free", "perrow")
    rows_out = [("key", "arm", "amplitude", "certified", "pix_diff", "rows_kept", "n_carriers_used")]
    tally = {(m, x): [0, 0] for m in ARMS for x in amps}
    kept = {(m, x): [] for m in ARMS for x in amps}
    t0 = time.time()
    for n, it in enumerate(items):
        try:
            base = np.asarray(Image.fromarray(np.asarray(store.load(it.image_path))
                                              ).convert("RGB").resize((a.side, a.side), Image.BICUBIC))
        except Exception:
            continue
        for amp in amps:
            feas = np.stack([row_feasible(base, v, amp, s) for v, s in signed])
            results = {}

            # --- aligned: one (carrier,sign) for all three marks, heights free ----
            best = None
            for ci in range(len(signed)):
                sel = pick_marks(feas[ci:ci + 1], T, 3, a.need, T // 2)
                if sel is None:
                    continue
                cov = float(np.mean([c[2] for c in sel]))
                if best is None or cov > best[0]:
                    best = (cov, ci, sel)
            if best:
                cov, ci, sel = best
                v, s = signed[ci]
                add = (s * amp * v)[:, None]
                plan = [(np.flatnonzero(feas[ci][y:y + T]) + y, add) for _, y, _ in sel]
                results["aligned"] = (build(base, plan, a.target), cov, 1)

            # --- free: each mark its own carrier and height -----------------------
            sel = pick_marks(feas, T, 3, a.need, T // 2)
            if sel:
                plan = []
                for ci, y, _ in sel:
                    v, s = signed[ci]
                    plan.append((np.flatnonzero(feas[ci][y:y + T]) + y, (s * amp * v)[:, None]))
                results["free"] = (build(base, plan, a.target),
                                   float(np.mean([c[2] for c in sel])), len({c[0] for c in sel}))

            # --- perrow: every row takes whichever carrier fits it ----------------
            any_fit = feas.any(axis=0)
            starts = windows(any_fit, T)
            picks, banned = [], np.zeros(len(starts), bool)
            for _ in range(3):
                sc = np.where(banned, -1.0, starts)
                y = int(np.argmax(sc))
                if sc[y] < a.need:
                    picks = None; break
                picks.append((y, float(sc[y])))
                banned[max(0, y - T - T // 2):min(len(starts), y + T + T // 2)] = True
            if picks:
                first = feas.argmax(axis=0)
                plan, used = [], set()
                for y, _ in picks:
                    for ci in np.unique(first[y:y + T][any_fit[y:y + T]]):
                        rr = np.flatnonzero((first[y:y + T] == ci) & any_fit[y:y + T]) + y
                        v, s = signed[int(ci)]
                        plan.append((rr, (s * amp * v)[:, None])); used.add(int(ci))
                # rebuild honouring the 3-vs-2 split by mark, not by row-group
                xa = base.astype(np.int64).copy(); xb = base.astype(np.int64).copy()
                for mi, (y, _) in enumerate(picks):
                    for ci in np.unique(first[y:y + T][any_fit[y:y + T]]):
                        rr = np.flatnonzero((first[y:y + T] == ci) & any_fit[y:y + T]) + y
                        v, s = signed[int(ci)]
                        add = (s * amp * v)[:, None]
                        xa[rr] += add
                        if mi < 2:
                            xb[rr] += add
                if min(xa.min(), xb.min()) >= 0 and max(xa.max(), xb.max()) <= 255:
                    xa, xb = xa.astype(np.uint8), xb.astype(np.uint8)
                    da = digest(np.asarray(Image.fromarray(xa).resize((a.target, a.target), Image.BICUBIC)))
                    db = digest(np.asarray(Image.fromarray(xb).resize((a.target, a.target), Image.BICUBIC)))
                    results["perrow"] = ((int(da == db), int((xa != xb).any(axis=2).sum())),
                                         float(np.mean([p[1] for p in picks])), len(used))

            for arm in ARMS:
                r = results.get(arm)
                cert, pix, cov, nc = 0, 0, 0.0, 0
                if r and r[0]:
                    (cert, pix), cov, nc = r[0], r[1], r[2]
                rows_out.append((it.key, arm, amp, cert, pix, round(cov, 4), nc))
                tally[(arm, amp)][0] += cert
                tally[(arm, amp)][1] += 1
                if cert:
                    kept[(arm, amp)].append(cov)
        if (n + 1) % 10 == 0:
            s = " ".join(f"{m}@{amps[-1]}={tally[(m,amps[-1])][0]}/{tally[(m,amps[-1])][1]}" for m in ARMS)
            print(f"  progress {n+1}/{len(items)} elapsed_s={time.time()-t0:.0f} {s}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows_out)
    parts = []
    for arm in ARMS:
        for amp in amps:
            c, t = tally[(arm, amp)]
            parts.append(f"{arm}_a{amp}={c}/{t}")
            k = kept[(arm, amp)]
            parts.append(f"{arm}_kept_a{amp}=" + (f"{np.mean(k):.3f}" if k else "na"))
    print("RESULT task=%d " % a.task + " ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
