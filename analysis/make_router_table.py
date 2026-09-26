#!/usr/bin/env python3
"""Emit the strong-probe router table: six oracle-supervised routers on the vision tower's
outputs, their held-out gains with record-clustered intervals (generated/e34_intervals.csv),
and their behavior on the 80 certified pairs (results/routing/e34b-probe-pairs/sh_0000_summary.csv).
Writes generated/tab_routers_body.tex."""
import csv, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
gains = {(r["router"], r["features"]): r for r in csv.DictReader(open(root / "analysis/output/e34_intervals.csv"))}
pairs = {(r["router"], r["features"]): r for r in csv.DictReader(open(root / "results/routing/e34b-probe-pairs/sh_0000_summary.csv"))}
rows = [r for r in csv.DictReader(open(root / "results/routing/e34b-probe-pairs/sh_0000.csv"))]
FEAT = {"proj": "projected tokens", "enc": "patch features", "proj+enc": "both"}
LEARN = {"ridge": "ridge", "gbt": "boosting"}
out = [r"\begingroup\scriptsize", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrr@{}}", r"\toprule",
       r"Features & Learner & Gain over \textsf{stop}, $95\%$ CI & Order acc. & Acts, natural & Acts, certified (dominated) \\",
       r"\midrule"]
for f in ("proj", "enc", "proj+enc"):
    for m in ("ridge", "gbt"):
        g, p = gains[(m, f)], pairs[(m, f)]
        acted = [r for r in rows if r["router"] == m and r["features"] == f and r["native_action"] != "stop"]
        dom = sum(r["native_action"] in ("att", "rea") for r in acted)
        cert = f"${len(acted)}$ of $80$ (${dom}$)" if acted else "$0$ of $80$"
        out.append(f"{FEAT[f]} & {LEARN[m]} & ${float(g['gain_over_stop']):+.3f}\\ [{float(g['gain_lo']):+.3f},\\,{float(g['gain_hi']):+.3f}]$ & "
                   f"${float(g['order_acc']):.3f}$ & ${float(p['natural_act_rate_insample']):.2f}$ & {cert} \\\\")
out += [r"\bottomrule", r"\end{tabular*}", r"\endgroup"]
(root / "analysis/output/tab_routers_body.tex").write_text("\n".join(out) + "\n")
print("wrote analysis/output/tab_routers_body.tex")
