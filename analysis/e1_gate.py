#!/usr/bin/env python3
"""E1's pre-registered gate, computed once over the merged shards.

The gate, fixed before the data existed:

    if the 95% lower bound on the SPLIT-SAMPLE routing value fails to exceed 0
    at the powered sample size, stop and write the negative result.

Three details are load-bearing and all three were established before this ran.

1.  The estimator is split-sample, not plug-in.  The plug-in V averages a
    per-item maximum over noisy gains and inherits the winner's curse; under a
    null with no item-specific signal it reads ~0.125 and does NOT shrink with n,
    so a bootstrap interval on it excludes zero essentially always and cannot
    fail.  Split-sample chooses each item's action on one replicate and scores it
    on the other; measured size 0.00, power 1.00 at V = 0.055 (checks C9/C9b).

2.  The bootstrap resamples RECORDS, not items.  Several steps share one image
    and question, so their gains are correlated; item-level resampling would be
    anti-conservative and would narrow the interval that decides the gate.

3.  A negative result is a real outcome here, not a failure to be explained away.
    If the bound covers zero at the powered n, no per-item router can beat the
    best fixed action by more than noise, and the paper says so.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aliasforge.metrics import routing_value, routing_value_crossfit  # noqa: E402

ACTIONS = ["stop", "att", "rea", "enc"]


def load(pattern: str):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def assemble(rows):
    """(A, B, clusters): replicate-0 gains, replicate-1 gains, record ids."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["key"]][int(r["rep"])] = r
    keys = sorted(k for k, v in by.items() if 0 in v and 1 in v)
    A = np.array([[float(by[k][0][f"g_{a}"]) for a in ACTIONS] for k in keys])
    B = np.array([[float(by[k][1][f"g_{a}"]) for a in ACTIONS] for k in keys])
    clusters = np.array([by[k][0].get("record", k.split(":")[0]) for k in keys])
    return A, B, clusters, keys


def clustered_bootstrap(A, B, clusters, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in uniq]
    stat = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        stat[b] = routing_value_crossfit(A[idx], B[idx])
    return float(np.quantile(stat, alpha / 2)), float(np.quantile(stat, 1 - alpha / 2)), stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/routing/e1-full/sh_*.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load(a.glob)
    if not rows:
        print(f"RESULT no rows matched {a.glob}")
        return 1
    A, B, clusters, keys = assemble(rows)
    n_items, n_rec = len(keys), len(np.unique(clusters))

    v_plug = routing_value(A)
    v_split = routing_value_crossfit(A, B)
    lo, hi, stat = clustered_bootstrap(A, B, clusters, a.n_boot, seed=a.seed)

    mean_g = A.mean(axis=0)
    best = int(np.argmax(mean_g))
    # the path-matched run (E21) carries no same-interface control column
    ctrl = (np.array([float(r["g_ctrl_same"]) for r in rows])
            if "g_ctrl_same" in rows[0] else None)
    ident = int(np.array_equal(A, B))

    print(f"exp=e1_gate items={n_items} records={n_rec} n_boot={a.n_boot} seed={a.seed}")
    print("mean gain per action: " + " ".join(f"{k}={mean_g[i]:+.4f}" for i, k in enumerate(ACTIONS)))
    print(f"best_fixed_action={ACTIONS[best]} value={mean_g[best]:+.4f}")
    print(f"replicates_identical={ident} (must be 0)")
    print("ctrl_same_interface max|gain|=" + (f"{np.abs(ctrl).max():.3e} (Thm 1 requires 0)" if ctrl is not None else "n/a (no control column in these shards)"))
    print(f"V_plugin={v_plug:.4f}  V_splitsample={v_split:.4f}  bias_gap={v_plug-v_split:+.4f}")
    print(f"V_splitsample 95% CI (clustered by record) = [{lo:.4f}, {hi:.4f}]")

    passed = lo > 0.0
    verdict = ("PASS - routing has headroom; proceed to E2/E3"
               if passed else
               "FAIL - interval covers 0; per-item routing has no demonstrable headroom, "
               "publish the negative result")
    print(f"GATE {verdict}")
    print(f"RESULT gate_passed={int(passed)} V_splitsample={v_split:.4f} lo={lo:.4f} hi={hi:.4f} "
          f"items={n_items} records={n_rec}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
