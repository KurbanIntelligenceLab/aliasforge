#!/usr/bin/env python3
"""Record-clustered 95% intervals on E1's per-action mean gains.

The gate reports an interval on V only, because V is the pre-registered
endpoint.  The three mean gains it prints alongside it (att, rea, enc) are bare
point estimates, so Figure 1 could show a confidence interval on the oracle and
nothing on the actions it is being compared against.  This supplies the missing
intervals using the SAME estimator the gate uses, so the two are comparable:

  * the same items -- assemble() from e1_gate, so an item enters here exactly
    when it entered the gate (both replicates present);
  * the same statistic base -- replicate 0, matching `mean_g = A.mean(axis=0)`;
  * the same resampling -- records with replacement, not items, because several
    steps share one image and question and item-level resampling would be
    anti-conservative;
  * the same 2000 draws and seed, so the numbers are reproducible.

stop is omitted: gain is measured against stopping, so its 0 is definitional
and has no sampling distribution.

Run:  python analysis/e1_action_ci.py
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from e1_gate import ACTIONS, assemble, load  # noqa: E402

VERSION = "e1_action_ci/1"


def clustered_ci(x, clusters, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap on the mean of x, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in uniq]
    stat = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        stat[b] = x[idx].mean()
    return (float(np.quantile(stat, alpha / 2)),
            float(np.quantile(stat, 1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/routing/e1-full/sh_*.csv")
    ap.add_argument("--out", default="analysis/output/action_ci.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load(a.glob)
    if not rows:
        print(f"RESULT no rows matched {a.glob}")
        return 1
    A, _B, clusters, keys = assemble(rows)
    n_items, n_rec = len(keys), len(np.unique(clusters))

    print(f"exp=e1_action_ci version={VERSION} glob={a.glob} "
          f"items={n_items} records={n_rec} n_boot={a.n_boot} seed={a.seed}")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["action", "mean", "lo95", "hi95", "n_items", "n_records",
                    "n_boot", "seed", "version"])
        for i, act in enumerate(ACTIONS):
            if act == "stop":          # the reference: 0 by construction
                continue
            mean = float(A[:, i].mean())
            lo, hi = clustered_ci(A[:, i], clusters, a.n_boot, seed=a.seed)
            w.writerow([act, f"{mean:.4f}", f"{lo:.4f}", f"{hi:.4f}", n_items,
                        n_rec, a.n_boot, a.seed, VERSION])
            print(f"RESULT action={act} mean={mean:+.4f} "
                  f"ci95=[{lo:+.4f},{hi:+.4f}] excludes_zero={int(hi < 0)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
