#!/usr/bin/env python3
"""Record-clustered intervals for every strong-probe router, from E34's saved features.

E34 prints point estimates.  Table 2's other rows carry a bootstrap clustered by record, the
unit items are correlated within, so the strong-probe rows must carry the same thing or they
are not comparable.  This refits each (features, model) router with the same cross-fitting E34
used, then bootstraps the fitted out-of-fold policy over records, exactly as e3_table.py does:
the policy is held fixed and the resampling is over clusters, so the interval describes the
estimate's sampling variability rather than the fit's.

Reads results/routing/e34-strong-probe/sh_0000_features.npz.  Writes generated/e34_intervals.csv.
"""
from __future__ import annotations
import argparse, csv, pathlib
import numpy as np

ACTIONS = ["stop", "att", "rea", "enc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/routing/e34-strong-probe/sh_0000_features.npz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/output/e34_intervals.csv")
    a = ap.parse_args()
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor

    root = pathlib.Path(__file__).resolve().parent
    z = np.load(root.parent / a.features, allow_pickle=True)
    XP, XE, A, B, rec = z["proj"].astype(np.float32), z["enc"].astype(np.float32), z["A"], z["B"], z["rec"]
    n = len(A)
    feats = {"proj": XP, "enc": XE, "proj+enc": np.concatenate([XP, XE], 1)}
    # HistGradientBoosting rather than the exact-split GradientBoosting the cluster run used:
    # on 7168 features the exact splitter is hours and the histogram one is seconds, and it is
    # the stronger learner, so a null from it is the stronger statement.
    makers = {"ridge": lambda: RidgeCV(alphas=(1e-1, 1, 10, 100, 1e3, 1e4)),
              "gbt": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                           early_stopping=True, random_state=0)}
    print(f"items={n} records={len(np.unique(rec))} dims={ {k: v.shape[1] for k, v in feats.items()} }", flush=True)

    uniq = np.unique(rec); rng = np.random.default_rng(a.seed)
    fold = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
    folds = np.array([fold[g] for g in rec])
    idx_by = {r: np.flatnonzero(rec == r) for r in uniq}
    stop = np.zeros(n, int)

    def stats(pi, ii):
        return (float(B[ii].max(1).mean() - B[ii, pi[ii]].mean()),
                float((B[ii].argmax(1) == pi[ii]).mean()),
                float(B[ii, pi[ii]].mean() - B[ii, 0].mean()))

    out, picks = [], {}
    for fname, X in feats.items():
        mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
        for mname, mk in makers.items():
            gh = np.zeros_like(A)
            for f in range(5):
                tr, te = folds != f, folds == f
                for j in range(len(ACTIONS)):
                    gh[te, j] = mk().fit(Xs[tr], A[tr, j]).predict(Xs[te])
            pi = gh.argmax(1)
            picks[f"{fname}:{mname}"] = pi
            pt = stats(pi, np.arange(n))
            brng = np.random.default_rng(a.seed); samp = []
            for _ in range(a.n_boot):
                ii = np.concatenate([idx_by[r] for r in brng.choice(uniq, len(uniq))])
                samp.append(stats(pi, ii))
            samp = np.array(samp)
            lo, hi = np.percentile(samp, 2.5, axis=0), np.percentile(samp, 97.5, axis=0)
            acts = {ACTIONS[j]: int((pi == j).sum()) for j in range(4)}
            print(f"{mname:6s} {fname:9s} regret {pt[0]:.4f} [{lo[0]:.4f},{hi[0]:.4f}]  "
                  f"ord {pt[1]:.4f}  gain/stop {pt[2]:+.4f} [{lo[2]:+.4f},{hi[2]:+.4f}]  acts {acts}", flush=True)
            out.append(dict(router=mname, features=fname, dim=X.shape[1],
                            regret=round(pt[0], 4), regret_lo=round(lo[0], 4), regret_hi=round(hi[0], 4),
                            order_acc=round(pt[1], 4), order_acc_lo=round(lo[1], 4), order_acc_hi=round(hi[1], 4),
                            gain_over_stop=round(pt[2], 4), gain_lo=round(lo[2], 4), gain_hi=round(hi[2], 4),
                            n=n, **{f"acts_{k}": v for k, v in acts.items()}))
    p = root.parent / a.out; p.parent.mkdir(exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {p}")
    # per-item cross-fitted choices, so other analyses can evaluate these routers on subsets
    q = p.with_name("e34_actions.csv")
    with open(q, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key"] + list(picks))
        for i, k in enumerate(z["keys"]):
            w.writerow([k] + [ACTIONS[picks[c][i]] for c in picks])
    print(f"wrote {q}")


if __name__ == "__main__":
    main()
