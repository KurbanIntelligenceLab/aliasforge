#!/usr/bin/env python3
"""Recompute Table 2's natural-stream columns on ONE population, with intervals.

Every row is the same 1280 items, the same choosing replicate and the same held-out scoring
replicate, and every interval is a bootstrap clustered by record, the unit items are
correlated within.  Two decision rules are reported because they disagree and the
disagreement is the finding: the argmax rule the protocol specifies, and the median-threshold
rule.  A rule that forces action on half the items is not a property of the score.

The gain-over-stop column carries the paired comparison the text makes, so a reader does not
subtract rounded regrets.  The random-score row averages over 200 draws of the score, and its
interval redraws the score inside each bootstrap replicate, so it is comparable with the
others.  The raw-distance and learned-router rows join the per-item file of the fill run
(e32) on item key, so they use exactly the gains and bootstrap of every other row.
"""
from __future__ import annotations
import argparse, csv, glob
import numpy as np

ACTIONS = ["stop", "att", "rea", "enc"]
COST = np.array([0.0, 1.00, 2.75, 1.01])   # measured decode-token cost, Table 4
N_DRAWS = 200                               # random-score draws


def load(pat, route="B"):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += [r for r in csv.DictReader(open(f)) if r["route"] == route]
    return rows


def regret(pi, B):
    return float(B.max(axis=1).mean() - B[np.arange(len(B)), pi].mean())


def order_acc(pi, B):
    return float((pi == B.argmax(axis=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/routing")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ias-dir", default="e3-pathmatched")
    ap.add_argument("--fill", default="e32-fill-table2/sh_0000.csv", help="per-item file of the fill run")
    ap.add_argument("--lam", type=float, default=0.0, help="compute price in the specified rule")
    ap.add_argument("--out", default="analysis/output/e3_table_pathmatched.csv")
    a = ap.parse_args()
    rows = load(f"{a.results_dir}/{a.ias_dir}/sh_*_peritem.csv")
    A = np.array([[float(r[f"gA_{x}"]) for x in ACTIONS] for r in rows])
    B = np.array([[float(r[f"gB_{x}"]) for x in ACTIONS] for r in rows])
    ias = np.array([float(r["ias"]) for r in rows])
    conf = np.array([float(r["conf"]) for r in rows])
    dev = np.array([r["split"] for r in rows]) == "dev"
    rec = np.array([int(r["record"]) for r in rows])
    keys = [r["key"] for r in rows]
    n = len(rows)
    print(f"exp=e3_table source={a.ias_dir} population=route-B n_items={n} n_records={len(np.unique(rec))} "
          f"n_boot={a.n_boot} lambda={a.lam} cost={list(COST)}")

    fill = {r["key"]: r for r in csv.DictReader(open(f"{a.results_dir}/{a.fill}"))}
    missing = [k for k in keys if k not in fill]
    if missing:
        raise SystemExit(f"fill run lacks {len(missing)} items, e.g. {missing[:3]}")
    dist = np.array([float(fill[k]["dist"]) for k in keys])
    router = np.array([ACTIONS.index(fill[k]["router_action"]) for k in keys])

    def spec(sc):                       # protocol's rule: argmax of fitted gain minus lam*cost
        gh = np.zeros_like(A)
        for j in range(len(ACTIONS)):
            gh[:, j] = np.polyval(np.polyfit(sc[dev], A[dev, j], 1), sc)
        return (gh - a.lam * COST).argmax(axis=1)

    def median_rule(sc):                # act below the median; choose the best non-stop action
        return np.where(sc <= np.median(sc), A[:, 1:].argmax(axis=1) + 1, 0)

    pi_spec = spec(ias)
    print(f"specified rule on IAS: reason on {int((pi_spec == 2).sum())}/{n} items, "
          + ", ".join(f"{ACTIONS[j]} on {int((pi_spec == j).sum())}" for j in (0, 1, 3) if (pi_spec == j).any()))

    rng = np.random.default_rng(0)
    pol = [("always-stop", np.zeros(n, int)), ("always-re-attend", np.full(n, 1)),
           ("always-reason", np.full(n, 2)), ("always-re-encode", np.full(n, 3)),
           ("IAS (specified argmax rule)", pi_spec),
           ("confidence (specified rule)", spec(conf)),
           ("IAS (median-threshold rule)", median_rule(ias)),
           ("confidence (median-threshold)", median_rule(conf)),
           ("raw interface distance (median-threshold)", median_rule(dist)),
           ("random action", rng.integers(0, 4, n)),
           ("random score (median-threshold)", None),          # averaged over draws below
           ("learned router (oracle-supervised)", router),
           ("per-item oracle (split-sample)", A.argmax(axis=1)),
           ("per-item oracle (plug-in)", B.argmax(axis=1))]

    uniq = np.unique(rec)
    idx_by = {r: np.flatnonzero(rec == r) for r in uniq}
    stop = np.zeros(n, int)
    draw_rng = np.random.default_rng(7)
    draws = [median_rule(draw_rng.random(n)) for _ in range(N_DRAWS)]

    def stats(pi, ii):
        return regret(pi[ii], B[ii]), order_acc(pi[ii], B[ii]), regret(stop[ii], B[ii]) - regret(pi[ii], B[ii])

    print(f"\n{'policy':44} {'regret':>7} {'95% CI':>17} {'ord.acc':>8} {'gain/stop':>10} {'95% CI':>17}")
    out = []
    for name, pi in pol:
        brng = np.random.default_rng(0)
        full = np.arange(n)
        if pi is None:
            pts = np.array([stats(d, full) for d in draws]).mean(axis=0)
        else:
            pts = np.array(stats(pi, full))
        samp = []
        for b in range(a.n_boot):
            pick = brng.choice(uniq, size=len(uniq), replace=True)
            ii = np.concatenate([idx_by[r] for r in pick])
            p = draws[b % N_DRAWS] if pi is None else pi
            samp.append(stats(p, ii))
        samp = np.array(samp)
        lo, hi = np.percentile(samp, 2.5, axis=0), np.percentile(samp, 97.5, axis=0)
        r, o, d = pts
        print(f"{name:44} {r:7.4f} [{lo[0]:6.4f},{hi[0]:6.4f}] {o:8.4f} {d:+10.4f} [{lo[2]:+.4f},{hi[2]:+.4f}]")
        out.append(dict(policy=name, regret=round(r, 4), regret_lo=round(lo[0], 4), regret_hi=round(hi[0], 4),
                        order_acc=round(o, 4), order_acc_lo=round(lo[1], 4), order_acc_hi=round(hi[1], 4),
                        gain_over_stop=round(d, 4), gain_lo=round(lo[2], 4), gain_hi=round(hi[2], 4), n=n))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
