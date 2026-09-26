#!/usr/bin/env python3
"""A1 -- natural verifier ERRORS under each language-side action (path-matched, local analysis).

Reads results/routing/e21-pathmatched/sh_*.csv, one row per item x replicate.  A row is an error
when the stop verdict (p_stop > 0.5 reads as label 1) disagrees with the label.  On the
error set (and its complement) it reports: the inert fraction (the three sampled critiques give one verdict,
rea_spread == 0), the fraction each action repairs (its verdict is correct; the placebo critique is the
control, being a fresh draw with no new instruction), and the mean
path-matched gain per action, each with a percentile bootstrap clustered by source record.
Writes analysis/output/a1_errors.csv and prints one RESULT line per subset.
"""
import csv, glob, sys
import numpy as np

VERSION = "a1.3"
ACTIONS = ["att", "rea", "enc"]
B, SEED = 2000, 0


def load(results_dir):
    rows = [r for f in sorted(glob.glob(f"{results_dir}/e21-pathmatched/sh_*.csv")) for r in csv.DictReader(open(f))]
    return rows


def correct_p(r, a):
    p = float(r[f"p_{a}"]); return p if r["label"] == "1" else 1.0 - p


def stats(rows):
    inert = np.array([float(r["rea_spread"]) == 0.0 for r in rows], float)
    rep = {a: np.array([correct_p(r, a) > 0.5 for r in rows], float) for a in ACTIONS + ["placebo"]}
    rep["any"] = np.maximum.reduce([rep[a] for a in ACTIONS])
    gain = {a: np.array([float(r[f"g_{a}"]) for r in rows]) for a in ACTIONS}
    return inert, rep, gain


def cluster_boot(rows, fn, rng):
    recs = sorted({r["record"] for r in rows}); by = {k: [] for k in recs}
    for i, r in enumerate(rows): by[r["record"]].append(i)
    idx = {k: np.array(v) for k, v in by.items()}
    vals = []
    for _ in range(B):
        pick = rng.choice(recs, size=len(recs), replace=True)
        sel = np.concatenate([idx[k] for k in pick]); vals.append(fn(sel))
    return np.percentile(vals, [2.5, 97.5])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", default="results/routing")
    ap.add_argument("--out", default="analysis/output/a1_errors.csv"); a = ap.parse_args()
    rows = load(a.results_dir); rng = np.random.default_rng(SEED)
    errors = [r for r in rows if correct_p(r, "stop") <= 0.5]
    right = [r for r in rows if correct_p(r, "stop") > 0.5]
    print(f"exp=a1_errors version={VERSION} input={a.results_dir}/e21-pathmatched rows={len(rows)} "
          f"errors={len(errors)} records={len({r['record'] for r in rows})}")
    out = []
    for name, sub in (("all", rows), ("errors", errors), ("correct", right)):
        inert, rep, gain = stats(sub)
        rec = {"subset": name, "n_rows": len(sub), "n_records": len({r["record"] for r in sub})}
        for key, arr in [("inert", inert)] + [(f"repaired_{a}", rep[a]) for a in ACTIONS + ["placebo", "any"]] + [(f"gain_{a}", gain[a]) for a in ACTIONS]:
            lo, hi = cluster_boot(sub, lambda sel, arr=arr: arr[sel].mean(), rng)
            rec[key], rec[f"{key}_lo"], rec[f"{key}_hi"] = round(float(arr.mean()), 4), round(float(lo), 4), round(float(hi), 4)
        out.append(rec)
        print(f"RESULT subset={name} n={len(sub)} inert={rec['inert']:.3f}[{rec['inert_lo']:.3f},{rec['inert_hi']:.3f}] "
              + " ".join(f"repaired_{a}={rec[f'repaired_{a}']:.3f}" for a in ACTIONS + ["placebo", "any"]) + " "
              + " ".join(f"gain_{a}={rec[f'gain_{a}']:+.4f}[{rec[f'gain_{a}_lo']:+.4f},{rec[f'gain_{a}_hi']:+.4f}]" for a in ACTIONS))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
