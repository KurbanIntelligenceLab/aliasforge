#!/usr/bin/env python3
"""E1 power calculation: how many verifier decisions are needed to detect routing value V.

E1 is the paper's go/no-go gate and it is pre-registered, so its sample size must
be fixed BEFORE it runs.  This computes it by Monte Carlo rather than by a
formula, for two reasons that a closed form gets wrong:

  1. V = E_i[max_a g_i(a)] - max_a E_i[g_i(a)] is a MAX-type statistic with a
     boundary null.  V >= 0 always, and V = 0 sits on the boundary of the
     parameter space, so its null sampling distribution is not normal and a
     t-test sample size would be wrong in an unknown direction.
  2. Items nest inside render templates.  The manuscript already commits to
     clustering the bootstrap at the template and source image because item-level
     resampling is anti-conservative there; the effective sample size is
     therefore driven by the number of TEMPLATES, not the number of items, and
     how strongly depends on the intra-cluster correlation.

Data-generating process, stated explicitly because the answer depends on it:
each item has a latent best action; with probability `rho` it is inherited from
the item's template (the clustering), otherwise drawn independently.  Gains are
a per-action base plus a bonus `delta` on the best action plus noise.  `delta` is
calibrated by bisection so the population V equals the target.  This is a
deliberately simple model of heterogeneity -- it fixes a sample size, it does not
claim to describe real verifiers.

Reports, per (target V, rho): the smallest number of templates reaching 80%
power, and the implied item count.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from aliasforge.metrics import (  # noqa: E402
    routing_value, routing_value_crossfit,
)

N_ACTIONS = 4
ITEMS_PER_TEMPLATE = 8
NOISE = 0.15


def simulate(n_templates: int, delta: float, rho: float, rng: np.random.Generator):
    """Return (paired replicates a, b of the SAME items, cluster ids).

    Two independent measurements per item, sharing the latent best action and
    differing only in noise -- which is what E1 obtains by running every action
    twice under stochastic decoding, and what the split-sample estimator needs.
    """
    k = ITEMS_PER_TEMPLATE
    n = n_templates * k
    clusters = np.repeat(np.arange(n_templates), k)
    tmpl_best = rng.integers(0, N_ACTIONS, size=n_templates)
    inherit = rng.random(n) < rho
    own = rng.integers(0, N_ACTIONS, size=n)
    best = np.where(inherit, tmpl_best[clusters], own)

    base = np.array([0.0, -0.02, -0.04, -0.06])  # stop is cheapest by default

    def draw():
        g = base[None, :] + rng.normal(scale=NOISE, size=(n, N_ACTIONS))
        g[np.arange(n), best] += delta
        return g

    return draw(), draw(), clusters


def calibrate_delta(target_V: float, rho: float, rng: np.random.Generator) -> float:
    """Bisect delta so the large-sample routing value matches the target."""
    lo, hi = 0.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        a, b, _ = simulate(4000 // ITEMS_PER_TEMPLATE, mid, rho, rng)
        if routing_value_crossfit(a, b) < target_V:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def power_at(n_templates: int, delta: float, rho: float, reps: int, rng: np.random.Generator,
             n_perm: int = 300, alpha: float = 0.05) -> float:
    """Fraction of replicates whose clustered lower bound on SPLIT-SAMPLE V exceeds 0.

    Two earlier versions of this were wrong and are recorded so they are not
    retried.  (1) Bootstrapping the PLUG-IN routing value returned power 1.00 in
    every cell including the smallest effect at the smallest sample -- the tell
    that it was measuring the estimator's upward bias, not signal.  (2) A
    within-item permutation test had power 0.00 everywhere, because permuting a
    row leaves its maximum, and hence E_i[max_a g_i(a)], exactly invariant.
    The split-sample estimator is the one with correct size AND power; see C9/C9b.
    """
    hits = 0
    for _ in range(reps):
        a, b, clusters = simulate(n_templates, delta, rho, rng)
        n_items = a.shape[0]
        uniq = np.unique(clusters)
        members = [np.flatnonzero(clusters == c) for c in uniq]
        stat = []
        for _ in range(n_perm):
            pick = rng.integers(0, len(uniq), size=len(uniq))   # cluster bootstrap
            idx = np.concatenate([members[q] for q in pick])
            stat.append(routing_value_crossfit(a[idx], b[idx]))
        hits += int(float(np.quantile(stat, alpha / 2)) > 0.0)
    return hits / reps


def size_at(n_templates: int, rho: float, reps: int, rng: np.random.Generator,
            n_perm: int = 300, alpha: float = 0.05) -> float:
    """Empirical false-positive rate: same test with delta = 0 (no signal).

    Reported alongside power in every row.  A power curve without a size column is
    uninterpretable, which is exactly how the first version of this calculation
    went wrong.
    """
    return power_at(n_templates, 0.0, rho, reps, rng, n_perm=n_perm, alpha=alpha)


def run(out: str, seed: int, reps: int):
    rng = np.random.default_rng(seed)
    grid_V = [0.01, 0.02, 0.03, 0.05]          # routing value in accuracy points
    grid_rho = [0.0, 0.5, 0.9]                  # 0 = no clustering, 0.9 = strong
    grid_T = [10, 20, 40, 80, 160, 320]         # templates

    print(f"exp=power_e1 seed={seed} reps={reps} actions={N_ACTIONS} "
          f"items_per_template={ITEMS_PER_TEMPLATE} noise={NOISE}")
    rows = []
    for V in grid_V:
        for rho in grid_rho:
            delta = calibrate_delta(V, rho, np.random.default_rng(seed + 7))
            chosen = None
            for T in grid_T:
                p = power_at(T, delta, rho, reps, rng)
                sz = size_at(T, rho, max(40, reps // 4), rng)
                rows.append(dict(target_V=V, rho=rho, delta=round(delta, 4), templates=T,
                                 items=T * ITEMS_PER_TEMPLATE, power=round(p, 3),
                                 size=round(sz, 3)))
                print(f"V={V:.3f} rho={rho:.1f} delta={delta:.3f} templates={T:4d} "
                      f"items={T*ITEMS_PER_TEMPLATE:5d} power={p:.3f} size={sz:.3f}")
                if chosen is None and p >= 0.80:
                    chosen = T
            print(f"MIN80 V={V:.3f} rho={rho:.1f} templates={chosen} "
                  f"items={chosen*ITEMS_PER_TEMPLATE if chosen else -1}")

    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"RESULT rows={len(rows)} grid_V={grid_V} grid_rho={grid_rho}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    run(a.out, a.seed, a.reps)
