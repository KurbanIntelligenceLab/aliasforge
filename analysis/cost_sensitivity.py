#!/usr/bin/env python3
"""The lambda sweep of Section 6 under three measurements of action cost.

Cost per action is measured on the same records three ways: decode tokens (the vector the
paper uses), prefill tokens, and wall-clock seconds, each net of stopping and normalized to one
critique. For each vector the specified rule is formed as in Table 2 and scored on the held-out
replicate over the lambda grid. Reports the largest regret gap to always-stop and the smallest
lambda from which every rule stops on every item. Writes generated/cost_sensitivity.csv.
"""
import csv, glob, pathlib
import numpy as np

ACTIONS = ["stop", "att", "rea", "enc"]
LAMS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
R = pathlib.Path("results/routing")

cost_rows = [r for f in sorted(glob.glob(str(R / "e12-cost/sh_*.csv"))) for r in csv.DictReader(open(f))]
vecs = {}
for metric in ("decode_tokens", "prefill_tokens", "wall_s"):
    m = {a: np.mean([float(r[metric]) for r in cost_rows if r["action"] == a]) for a in ACTIONS}
    net = {a: m[a] - m["stop"] for a in ACTIONS}
    vecs[metric] = np.array([net[a] / net["att"] for a in ACTIONS])

rows = [r for f in sorted(glob.glob(str(R / "e3-pathmatched/sh_*_peritem.csv"))) for r in csv.DictReader(open(f)) if r["route"] == "B"]
A = np.array([[float(r[f"gA_{x}"]) for x in ACTIONS] for r in rows])
B = np.array([[float(r[f"gB_{x}"]) for x in ACTIONS] for r in rows])
dev = np.array([r["split"] for r in rows]) == "dev"
scores = {"IAS": np.array([float(r["ias"]) for r in rows]), "confidence": np.array([float(r["conf"]) for r in rows])}


def regret(pi, G):
    return float(G.max(1).mean() - G[np.arange(len(G)), pi].mean())


out = []
for metric, c in vecs.items():
    gap, collapse = 0.0, None
    for lam in LAMS:
        all_stop = True
        for name, sc in scores.items():
            gh = np.column_stack([np.polyval(np.polyfit(sc[dev], A[dev, j], 1), sc) for j in range(4)])
            pi = (gh - lam * c).argmax(1)
            gap = max(gap, abs(regret(pi, B) - regret(np.zeros(len(pi), int), B)))
            all_stop &= bool((pi == 0).all())
        if all_stop and collapse is None:
            collapse = lam
    out.append(dict(metric=metric, c_att=round(c[1], 3), c_rea=round(c[2], 3), c_enc=round(c[3], 3),
                    max_regret_gap_to_stop=round(gap, 4), all_stop_from_lambda=collapse))
    print(f"metric={metric} cost=({', '.join(f'{x:.3f}' for x in c)}) max_gap_to_stop={gap:.4f} all_stop_from={collapse}")
with open("analysis/output/cost_sensitivity.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
