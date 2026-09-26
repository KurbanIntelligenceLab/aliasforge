#!/usr/bin/env python3
"""E27 summary: the accuracy cost of the Lanczos kernel, with a record-clustered bootstrap CI.

Reads the E27 shards (a glob), splits item rows from the `__pair_<i>__` collision rows, and
reports per kernel the mean soft correctness and the hard accuracy; the paired per-item
difference (lanczos - bicubic) with a 95% percentile bootstrap interval clustered by RECORD
(items from one benchmark record share an image and a prefix, so they are not independent);
the number of items whose hard verdict flips; and the pair-collision counts.

Usage:  python analysis/e27_summary.py 'results/constructibility/e27-kernelswap/sh_*.csv'
"""
from __future__ import annotations

import argparse
import csv
import glob

import numpy as np

VERSION = "e27s.1"


def read_rows(pattern: str):
    items, pairs = [], []
    for f in sorted(glob.glob(pattern)):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                (pairs if r["key"].startswith("__pair_") else items).append(r)
    return items, pairs


def cluster_bootstrap(d: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05):
    """Percentile CI of mean(d) resampling whole clusters with replacement.

    Each replicate draws C clusters with replacement and takes the mean over every
    item they contain, so a cluster's items always move together.
    """
    uniq, inv = np.unique(clusters, return_inverse=True)
    C = len(uniq)
    sums = np.bincount(inv, weights=d, minlength=C)
    cnts = np.bincount(inv, minlength=C).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, C, size=(n_boot, C))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob", help="shard glob, e.g. 'results/constructibility/e27-kernelswap/sh_*.csv'")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items, pairs = read_rows(a.glob)
    n_files = len(glob.glob(a.glob))
    print(f"exp=e27_summary version={VERSION} glob={a.glob} files={n_files} items={len(items)} "
          f"pair_rows={len(pairs)} n_boot={a.n_boot} seed={a.seed}")
    if not items:
        print("RESULT items=0")
        return 2

    rec = np.array([r["record"] for r in items])
    lab = np.array([int(r["label"]) for r in items])
    c = {k: np.array([float(r[f"correct_{k}"]) for r in items]) for k in ("bicubic", "lanczos")}
    h = {k: np.array([int(r[f"hard_{k}"]) for r in items]) for k in ("bicubic", "lanczos")}
    p = {k: np.array([float(r[f"p_stop_{k}"]) for r in items]) for k in ("bicubic", "lanczos")}

    d_soft = c["lanczos"] - c["bicubic"]
    d_hard = (h["lanczos"] - h["bicubic"]).astype(float)
    lo_s, hi_s, C = cluster_bootstrap(d_soft, rec, a.n_boot, a.seed)
    lo_h, hi_h, _ = cluster_bootstrap(d_hard, rec, a.n_boot, a.seed)
    flips = int((h["bicubic"] != h["lanczos"]).sum())
    to_wrong = int(((h["bicubic"] == 1) & (h["lanczos"] == 0)).sum())
    to_right = int(((h["bicubic"] == 0) & (h["lanczos"] == 1)).sum())

    print(f"items={len(items)} records={C} label_pos={int(lab.sum())} label_neg={int((1-lab).sum())}")
    for k in ("bicubic", "lanczos"):
        print(f"kernel={k} mean_correct={c[k].mean():.4f} hard_acc={h[k].mean():.4f} "
              f"mean_p_stop={p[k].mean():.4f}")
    print(f"delta_soft_lanczos_minus_bicubic={d_soft.mean():+.4f} ci95=[{lo_s:+.4f},{hi_s:+.4f}] "
          f"excludes_zero={int(lo_s > 0 or hi_s < 0)}")
    print(f"delta_hard_lanczos_minus_bicubic={d_hard.mean():+.4f} ci95=[{lo_h:+.4f},{hi_h:+.4f}] "
          f"excludes_zero={int(lo_h > 0 or hi_h < 0)}")
    print(f"mean_abs_p_shift={np.abs(p['lanczos']-p['bicubic']).mean():.4f} "
          f"max_abs_p_shift={np.abs(p['lanczos']-p['bicubic']).max():.4f}")
    print(f"flips={flips} flips_right_to_wrong={to_wrong} flips_wrong_to_right={to_right} "
          f"flip_rate={flips/len(items):.4f}")

    if pairs:
        cb = sum(int(r["collide_bicubic"]) for r in pairs)
        cl = sum(int(r["collide_lanczos"]) for r in pairs)
        gb = [float(r["gap_bicubic"]) for r in pairs if r.get("gap_bicubic", "") != ""]
        gl = [float(r["gap_lanczos"]) for r in pairs if r.get("gap_lanczos", "") != ""]
        print(f"pairs={len(pairs)} collide_bicubic={cb}/{len(pairs)} collide_lanczos={cl}/{len(pairs)} "
              f"max_gap_bicubic={max(gb) if gb else float('nan'):.6f} "
              f"max_gap_lanczos={max(gl) if gl else float('nan'):.6f} "
              f"defence_breaks_all_pairs={int(cb == len(pairs) and cl == 0)}")
    else:
        print("pairs=0 (no task-0 shard in the glob)")
    print(f"RESULT items={len(items)} mean_correct_bicubic={c['bicubic'].mean():.4f} "
          f"mean_correct_lanczos={c['lanczos'].mean():.4f} delta={d_soft.mean():+.4f} "
          f"ci95=[{lo_s:+.4f},{hi_s:+.4f}] hard_bicubic={h['bicubic'].mean():.4f} "
          f"hard_lanczos={h['lanczos'].mean():.4f} flips={flips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
