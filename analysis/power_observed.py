#!/usr/bin/env python3
"""Power of the routing gate at the clustering the analysis actually used.

The pre-registered calculation (Table 5) simulated 160 templates of 8 items.  The natural
stream that was analyzed has 1018 records over 1280 items, mean cluster 1.26, and its
intervals are clustered by record.  This recomputes power with the OBSERVED cluster sizes,
same generative model and same split-sample estimator as the pre-registration, at the
pre-registered rho grid, and reports the record-level intra-cluster correlation of the
per-item headroom alongside.  The effect size is calibrated exactly as the pre-registration
calibrated it, by bisection of delta against the routing value the simulation measures.
"""
import collections, csv, glob, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from aliasforge.metrics import routing_value_crossfit  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "routing"))
from power_e1 import calibrate_delta  # noqa: E402  the pre-registration's own effect-size calibration

ACT = ["stop", "att", "rea", "enc"]; NOISE = 0.15; BASE = np.array([0.0, -0.02, -0.04, -0.06])

ROOT = pathlib.Path(__file__).resolve().parents[1]
rows = [r for f in sorted(glob.glob(str(ROOT / "results/routing/e21-pathmatched/sh_*.csv"))) for r in csv.DictReader(open(f))]
by = collections.defaultdict(dict)
for r in rows: by[r["key"]][r["rep"]] = r
keys = sorted(k for k, v in by.items() if "0" in v and "1" in v)
rec = np.array([by[k]["0"]["record"] for k in keys])
G1 = np.array([[float(by[k]["1"][f"g_{a}"]) for a in ACT] for k in keys])
sizes = np.array(sorted(collections.Counter(rec).values()))
n = sizes.sum()

# record-level ICC(1) of the per-item headroom over stop
y = G1.max(1) - G1[:, 0]; grp = collections.defaultdict(list)
for i, r in enumerate(rec): grp[r].append(y[i])
k = len(grp); gm = y.mean()
ssb = sum(len(v) * (np.mean(v) - gm) ** 2 for v in grp.values()); ssw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in grp.values())
msb, msw = ssb / (k - 1), ssw / (n - k); n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
icc = (msb - msw) / (msb + (n0 - 1) * msw)
print(f"observed: items={n} records={k} mean_cluster={sizes.mean():.2f} max={sizes.max()} singletons={(sizes==1).sum()} ICC={icc:.3f}")

def simulate(delta, rho, rng):
    clusters = np.repeat(np.arange(len(sizes)), sizes)
    cl_best = rng.integers(0, 4, size=len(sizes)); inherit = rng.random(n) < rho
    best = np.where(inherit, cl_best[clusters], rng.integers(0, 4, size=n))
    def draw():
        g = BASE[None, :] + rng.normal(scale=NOISE, size=(n, 4)); g[np.arange(n), best] += delta; return g
    return draw(), draw(), clusters

def power(delta, rho, reps=200, n_boot=300, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed); hits = 0
    for _ in range(reps):
        a, b, cl = simulate(delta, rho, rng); uniq = np.unique(cl); members = [np.flatnonzero(cl == c) for c in uniq]
        stat = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq)); idx = np.concatenate([members[q] for q in pick])
            stat.append(routing_value_crossfit(a[idx], b[idx]))
        hits += int(float(np.quantile(stat, alpha / 2)) > 0.0)
    return hits / reps

out = []
crng = np.random.default_rng(1)
for rho in (0.0, 0.5, 0.9):
    for V in (0.0, 0.02, 0.03):
        delta = 0.0 if V == 0.0 else calibrate_delta(V, rho, crng)
        pw = power(delta, rho); out.append(dict(rho=rho, V=V, delta=round(delta, 4), power=pw))
        print(f"RESULT rho={rho} V={V} delta={delta:.4f} power={pw:.3f}", flush=True)
with open(ROOT / "analysis" / "output" / "power_observed.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
