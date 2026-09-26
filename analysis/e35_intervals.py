#!/usr/bin/env python3
"""Routers whose signal reads the full interface state (z, Q, R), from E35's saved features.

E35 saved the language model's hidden state at the decision position for every natural item and
for the 80 certified pairs.  That vector is a function of (z, Q, R), so Proposition 4 applies
to it with S = (z, Q, R), a richer signal than the vision-tower features of E34, which read z
alone.  This fits the same routers as e34_intervals.py with the same folds, learners and
record-clustered bootstrap, so the rows are comparable with Tables 2 and 10.  For the
certified-pair screen each router is refit on the full natural stream, as E34b does.

Reads results/routing/e35-signal-zqr/sh_0000_features.npz.
Writes generated/e35_intervals.csv and generated/e35_actions.csv.
"""
from __future__ import annotations
import argparse, csv, pathlib
import numpy as np

ACTIONS = ["stop", "att", "rea", "enc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/routing/e35-signal-zqr/sh_0000_features.npz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="analysis/output/e35_intervals.csv")
    a = ap.parse_args()
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor

    root = pathlib.Path(__file__).resolve().parent
    z = np.load(root.parent / a.features, allow_pickle=True)
    A, B, rec = z["A"], z["B"], z["rec"]
    n = len(A)
    feats = {"last": z["last"].astype(np.float32), "mid": z["mid"].astype(np.float32)}
    feats["last+mid"] = np.concatenate([feats["last"], feats["mid"]], 1)
    pair = {"last": z["pair_last"].astype(np.float32), "mid": z["pair_mid"].astype(np.float32)}
    pair["last+mid"] = np.concatenate([pair["last"], pair["mid"]], 1)
    makers = {"ridge": lambda: RidgeCV(alphas=(1e-1, 1, 10, 100, 1e3, 1e4)),
              "gbt": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                           early_stopping=True, random_state=0)}
    print(f"exp=e35_intervals items={n} records={len(np.unique(rec))} pairs={len(pair['last'])} "
          f"pair_max_abs_signal_diff={float(z['pair_max_abs_signal_diff'])} pair_same_digest={int(z['pair_same_digest'])} "
          f"dims={ {k: v.shape[1] for k, v in feats.items()} }", flush=True)

    uniq = np.unique(rec); rng = np.random.default_rng(a.seed)
    fold = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
    folds = np.array([fold[g] for g in rec])
    idx_by = {r: np.flatnonzero(rec == r) for r in uniq}

    def stats(pi, ii):
        return (float(B[ii].max(1).mean() - B[ii, pi[ii]].mean()),
                float((B[ii].argmax(1) == pi[ii]).mean()),
                float(B[ii, pi[ii]].mean() - B[ii, 0].mean()))

    out, picks = [], {}
    for fname, X in feats.items():
        mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd; Ps = (pair[fname] - mu) / sd
        for mname, mk in makers.items():
            gh = np.zeros_like(A)
            for f in range(5):
                tr, te = folds != f, folds == f
                for j in range(len(ACTIONS)):
                    gh[te, j] = mk().fit(Xs[tr], A[tr, j]).predict(Xs[te])
            pi = gh.argmax(1); picks[f"{fname}:{mname}"] = pi
            pt = stats(pi, np.arange(n))
            brng = np.random.default_rng(a.seed)
            samp = np.array([stats(pi, np.concatenate([idx_by[r] for r in brng.choice(uniq, len(uniq))]))
                             for _ in range(a.n_boot)])
            lo, hi = np.percentile(samp, 2.5, axis=0), np.percentile(samp, 97.5, axis=0)
            # certified pairs: refit on the full natural stream, take the native action
            full = [mk().fit(Xs, A[:, j]) for j in range(len(ACTIONS))]
            nat_rate = float((np.array([m.predict(Xs) for m in full]).T.argmax(1) != 0).mean())
            pp = np.array([m.predict(Ps) for m in full]).T.argmax(1)
            acts = {ACTIONS[j]: int((pp == j).sum()) for j in range(4)}
            any_action = int((pp != 0).sum()); dominated = int(np.isin(pp, (1, 2)).sum())
            print(f"RESULT {mname:6s} {fname:9s} regret {pt[0]:.4f} [{lo[0]:.4f},{hi[0]:.4f}] ord {pt[1]:.4f} "
                  f"gain/stop {pt[2]:+.4f} [{lo[2]:+.4f},{hi[2]:+.4f}] natural_act_rate={nat_rate:.3f} "
                  f"pairs any_action={any_action}/{len(pp)} dominated={dominated}/{len(pp)} pair_acts={acts}", flush=True)
            out.append(dict(router=mname, features=fname, dim=X.shape[1],
                            regret=round(pt[0], 4), regret_lo=round(lo[0], 4), regret_hi=round(hi[0], 4),
                            order_acc=round(pt[1], 4), gain_over_stop=round(pt[2], 4),
                            gain_lo=round(lo[2], 4), gain_hi=round(hi[2], 4), n=n,
                            natural_act_rate_insample=round(nat_rate, 4), pairs=len(pp),
                            pairs_any_action=any_action, pairs_dominated=dominated,
                            **{f"pair_{k}": v for k, v in acts.items()}))
    p = root.parent / a.out; p.parent.mkdir(exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {p}")
    q = p.with_name("e35_actions.csv")
    with open(q, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key"] + list(picks))
        for i, k in enumerate(z["keys"]):
            w.writerow([k] + [ACTIONS[picks[c][i]] for c in picks])
    print(f"wrote {q}")


if __name__ == "__main__":
    main()
