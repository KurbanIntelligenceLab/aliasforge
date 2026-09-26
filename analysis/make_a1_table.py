#!/usr/bin/env python3
"""Render Table tab:a1 from a1_errors.csv.

Transposed: quantities down the side, subsets across. The row-per-subset layout needed ten
columns, three of them carrying a value and a bracketed interval, and came out at 588pt
against a 397pt text width, so \\resizebox shrank it to about 6pt. This shape fits unscaled.
"""
import argparse, csv, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="analysis/output/a1_errors.csv")
ap.add_argument("--out", default="analysis/output/tab_a1_body.tex")
a = ap.parse_args()
rows = {r["subset"]: r for r in csv.DictReader(open(a.csv))}
ORDER = ["all", "errors", "correct"]
HEAD = {"all": "all", "errors": "stop wrong", "correct": "stop correct"}

import math
def wilson(p, n, z=1.96):
    """Wilson 95% interval for a proportion, the interval pre-registered for verifier stochasticity."""
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def val(r, k):    return f"${float(r[k]):.3f}$"
def prop(r, k):
    lo, hi = wilson(float(r[k]), int(r["n_rows"]))
    return f"${float(r[k]):.3f}$ {{\\scriptsize$[{lo:.3f},\\,{hi:.3f}]$}}"
def gain(r, k):   return (f"${float(r[k]):+.3f}$ {{\\scriptsize$[{float(r[k+'_lo']):+.3f},"
                          f"\\,{float(r[k+'_hi']):+.3f}]$}}")

L = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l ccc@{}}", r"\toprule",
     "Quantity & " + " & ".join(HEAD[s] for s in ORDER) + r" \\",
     "measurements $n$ & " + " & ".join(rows[s]["n_rows"] for s in ORDER) + r" \\",
     r"\midrule",
     r"\multicolumn{4}{@{}l}{\emph{inert under three critiques}}\\",
     "\\quad rate & " + " & ".join(prop(rows[s], "inert") for s in ORDER) + r" \\",
     r"\midrule",
     r"\multicolumn{4}{@{}l}{\emph{verdict correct after the action}}\\"]
for lbl, key in (("placebo", "repaired_placebo"), (r"\textsf{att}", "repaired_att"),
                 (r"\textsf{rea}", "repaired_rea"), (r"\textsf{enc}", "repaired_enc")):
    L.append(f"\\quad {lbl} & " + " & ".join(prop(rows[s], key) for s in ORDER) + r" \\")
L += [r"\midrule", r"\multicolumn{4}{@{}l}{\emph{mean gain against the placebo}}\\"]
for lbl, key in ((r"\textsf{att}", "gain_att"), (r"\textsf{rea}", "gain_rea"),
                 (r"\textsf{enc}", "gain_enc")):
    L.append(f"\\quad {lbl} & " + " & ".join(gain(rows[s], key) for s in ORDER) + r" \\")
L += [r"\bottomrule", r"\end{tabular*}"]
pathlib.Path(a.out).write_text("\n".join(L) + "\n")
print(f"wrote {a.out} ({len(ORDER)} subsets, transposed)")
