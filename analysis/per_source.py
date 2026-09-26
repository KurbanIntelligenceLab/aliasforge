#!/usr/bin/env python3
"""Routing results within each source benchmark of the natural stream, against its best fixed action.

The 1280-item stream draws its records from five benchmarks. Each policy is formed exactly as
in Table 2 on the full stream (choosing replicate A, scoring replicate B) and then evaluated
inside each source, so a row answers whether the full-stream conclusion holds there. Intervals
are bootstraps clustered by record within the source.

Also reports the specified rule with its per-action gain map cross-fitted across the two record
halves (fit on one half, applied to the other), next to the rule as Table 2 runs it (fit on the
development half, applied to all items).

Writes generated/per_source.csv and generated/tab_sources_body.tex.
"""
from __future__ import annotations
import argparse, csv, glob, pathlib
import numpy as np

ACTIONS = ["stop", "att", "rea", "enc"]
N_DRAWS = 200
SHORT = {"MathVerse_MINI_Vision_Only": "MathVerse", "MathVision_MINI": "MathVision",
         "DynaMath": "DynaMath", "WeMath": "WeMath", "MMMU_DEV_VAL": "MMMU"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/routing")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="analysis/output/per_source.csv")
    ap.add_argument("--tex", default="analysis/output/tab_sources_body.tex")
    ap.add_argument("--router-actions", default="analysis/output/e34_actions.csv")
    a = ap.parse_args()
    R = pathlib.Path(a.results_dir)

    rows = [r for f in sorted(glob.glob(str(R / "e3-pathmatched/sh_*_peritem.csv")))
            for r in csv.DictReader(open(f)) if r["route"] == "B"]
    src = {r["key"]: r["source"] for f in sorted(glob.glob(str(R / "e21-pathmatched/sh_*.csv")))
           for r in csv.DictReader(open(f))}
    # Table 2's learned router: boosting on patch features, cross-fitted by record (e34_intervals.py)
    picks = {r["key"]: r["enc:gbt"] for r in csv.DictReader(open(a.router_actions))}
    keys = [r["key"] for r in rows]
    A = np.array([[float(r[f"gA_{x}"]) for x in ACTIONS] for r in rows])
    B = np.array([[float(r[f"gB_{x}"]) for x in ACTIONS] for r in rows])
    ias = np.array([float(r["ias"]) for r in rows])
    conf = np.array([float(r["conf"]) for r in rows])
    dev = np.array([r["split"] for r in rows]) == "dev"
    rec = np.array([r["record"] for r in rows])
    source = np.array([SHORT[src[k]] for k in keys])
    router = np.array([ACTIONS.index(picks[k]) for k in keys])
    n = len(rows)

    def spec(sc, crossfit=False):
        gh = np.zeros_like(A)
        if not crossfit:
            for j in range(4):
                gh[:, j] = np.polyval(np.polyfit(sc[dev], A[dev, j], 1), sc)
        else:
            for fit, app in ((dev, ~dev), (~dev, dev)):
                for j in range(4):
                    gh[app, j] = np.polyval(np.polyfit(sc[fit], A[fit, j], 1), sc[app])
        return gh.argmax(axis=1)

    def median_rule(sc):
        return np.where(sc <= np.median(sc), A[:, 1:].argmax(axis=1) + 1, 0)

    rng = np.random.default_rng(7)
    draws = [median_rule(rng.random(n)) for _ in range(N_DRAWS)]
    pol = {"oracle": A.argmax(axis=1),
           "ias_spec": spec(ias), "ias_spec_xfit": spec(ias, True),
           "conf_spec": spec(conf), "conf_spec_xfit": spec(conf, True),
           "ias_median": median_rule(ias), "random_median": None, "learned_router": router}

    def gain(pi, ii):                      # gain over the subset's best fixed action, chosen on A, scored on B
        best = int(A[sel].mean(0).argmax())
        return float(B[ii, :][np.arange(len(ii)), pi[ii]].mean() - B[ii, best].mean())

    def vsplit(ii):
        return float(B[ii][np.arange(len(ii)), A[ii].argmax(1)].mean() - B[ii].mean(0).max())

    out = []
    for name in ["all", "MathVerse", "MathVision", "DynaMath", "WeMath", "MMMU"]:
        sel = np.arange(n) if name == "all" else np.flatnonzero(source == name)
        recs = np.unique(rec[sel])
        members = {r: sel[rec[sel] == r] for r in recs}
        brng = np.random.default_rng(0)
        boots = [np.concatenate([members[r] for r in brng.choice(recs, len(recs))]) for _ in range(a.n_boot)]

        def stat(fn):
            pt = fn(sel)
            s = np.array([fn(ii) for ii in boots])
            return pt, float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
        row = dict(source=name, items=len(sel), records=len(recs), best_fixed=ACTIONS[int(A[sel].mean(0).argmax())],
                   best_fixed_over_stop=round(float(B[sel, int(A[sel].mean(0).argmax())].mean() - B[sel, 0].mean()), 6))
        for k, (pt, lo, hi) in (("V", stat(vsplit)),):
            row.update({k: round(pt, 6), f"{k}_lo": round(lo, 6), f"{k}_hi": round(hi, 6)})
        for pname, pi in pol.items():
            if pi is None:
                pt = float(np.mean([gain(d, sel) for d in draws]))
                s = np.array([gain(draws[b % N_DRAWS], ii) for b, ii in enumerate(boots)])
                lo, hi = float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
            else:
                pt, lo, hi = stat(lambda ii, pi=pi: gain(pi, ii))
            row.update({pname: round(pt, 6), f"{pname}_lo": round(lo, 6), f"{pname}_hi": round(hi, 6)})
        out.append(row)
        print(f"{name:10s} best={row['best_fixed']}({row['best_fixed_over_stop']:+.3f}) n={len(sel):4d} rec={len(recs):4d} V={row['V']:+.3f} [{row['V_lo']:+.3f},{row['V_hi']:+.3f}] "
              f"oracle={row['oracle']:+.3f} ias_spec={row['ias_spec']:+.3f} [{row['ias_spec_lo']:+.3f},{row['ias_spec_hi']:+.3f}] "
              f"xfit={row['ias_spec_xfit']:+.3f} conf_spec={row['conf_spec']:+.3f} xfit={row['conf_spec_xfit']:+.3f} "
              f"ias_med={row['ias_median']:+.3f} rand_med={row['random_median']:+.3f} pix={row['learned_router']:+.3f} "
              f"[{row['learned_router_lo']:+.3f},{row['learned_router_hi']:+.3f}]", flush=True)

    for p in ("ias_spec", "ias_spec_xfit", "conf_spec", "conf_spec_xfit"):
        pi = pol[p]
        print(f"actions {p}: " + " ".join(f"{ACTIONS[j]}={int((pi == j).sum())}" for j in range(4)))
    pathlib.Path(a.out).parent.mkdir(exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {a.out}")

    def f3(x):
        return f"{0.0 if abs(x) < 5e-4 else x:+.3f}"

    def ci(r, k):
        return f"${f3(r[k])}$ {{\\scriptsize$[{f3(r[k+'_lo'])},{f3(r[k+'_hi'])}]$}}"
    L = [r"\begin{tabular}{@{}l l c c c c@{}}", r"\toprule",
         r"Source & Best & $\hat V$ & \begin{tabular}{@{}c@{}}\ROUTE{},\\ specified rule\end{tabular} & \begin{tabular}{@{}c@{}}Median rule,\\ \ROUTE{} / random\end{tabular} & \begin{tabular}{@{}c@{}}Learned\\ router\end{tabular} \\",
         r"\midrule"]
    for r in out[1:] + out[:1]:
        if r["source"] == "all":
            L.append(r"\midrule")
        name = "all five" if r["source"] == "all" else r["source"]
        L.append(f"{name} (${r['items']}$) & \\textsf{{{r['best_fixed']}}} & {ci(r, 'V')} & {ci(r, 'ias_spec')} & "
                 f"${f3(r['ias_median'])}$ / ${f3(r['random_median'])}$ & {ci(r, 'learned_router')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    pathlib.Path(a.tex).write_text("\n".join(L) + "\n")
    print(f"wrote {a.tex}")


if __name__ == "__main__":
    main()
