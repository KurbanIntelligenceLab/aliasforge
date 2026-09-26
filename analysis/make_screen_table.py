#!/usr/bin/env python3
"""Emit the body of the constructibility-screen table from the E10b records (sound lattice).

The resize operator at a fixed ratio is shift-invariant, so every declared configuration at
one ratio is the same computation: the sound Gram-Schmidt bound, the sublattice number the
old tool printed, the shortest LLL witness and the verdict are identical across them, which
this script asserts.  One row per distinct ratio, with the configurations it covers listed
beside it.  Configurations whose processor resolves to a different geometry at its default
(Table 9) carry a dagger.  Writes generated/tab_screen_body.tex from
results/constructibility/e10b-screen/sh_*.csv.
"""
import csv, glob, math, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
rows = [r for f in sorted(glob.glob(str(root / "results/constructibility/e10b-screen/sh_*.csv")))
        for r in csv.DictReader(open(f)) if r.get("name")]
if not rows: sys.exit("no e10b records")
# declared geometries that the processor's default configuration does not use (Table 9)
DAGGER = {"phi3v-336-2x", "smolvlm-384-2x", "idefics2-378-1.5x", "fuyu-300-1.5x"}

def sci(x):
    try: x = float(x)
    except (ValueError, TypeError): return "{--}"
    if x == 0: return "$0$"
    e = int(math.floor(math.log10(abs(x)))); m = x / 10**e
    return f"${x:.2f}$" if -1 <= e <= 2 else f"${m:.1f}{{\\times}}10^{{{e}}}$"

def verdict(r):
    if r["gate_ok"] != "1": return "n/a"
    if r["lll_verdict"] == "certified": return "constructible"
    if r["lll_verdict"] == "dagger": return "unplaceable"
    try: b = float(r["gs_bound_min"])
    except ValueError: b = None
    return "excluded" if (b is not None and b > 255) else "undetermined"

groups = {}
for r in rows:
    groups.setdefault(round(float(r["ratio"]), 4), []).append(r)
# scriptsize and a wide names column: eleven configurations share the 2x row
# One line per ratio: the configuration names drop the ratio suffix (the row states it),
# so the names column no longer wraps into lines with empty numeric cells.
out = ["\\begingroup\\scriptsize",
       "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}rrrrl>{\\raggedright\\arraybackslash}m{0.46\\textwidth}@{}}",
       "\\toprule",
       "Ratio & Sound bound & Rational bound & Witness & Verdict & Configurations at this ratio \\\\",
       "\\midrule"]
import re as _re
def short(name):
    return _re.sub(r"-\d+(?:\.\d+)?x$", "", name)
counts, n_cfg = {}, 0
for ratio in sorted(groups):
    g = groups[ratio]
    keyf = lambda r: (r["gs_bound_min"], r["gs_bound_sublattice"], r["lll_maxabs"], verdict(r))
    assert len({keyf(r) for r in g}) == 1, f"ratio {ratio}: rows differ, {[keyf(r) for r in g]}"
    r = g[0]; v = verdict(r); counts[v] = counts.get(v, 0) + len(g); n_cfg += len(g)
    w = r["lll_maxabs"] if (v in ("constructible", "unplaceable") and r["lll_maxabs"] not in ("", None)) else None
    wcell = f"${w}$" if w is not None else "{--}"
    names = ", ".join(f"\\texttt{{{short(x['name'])}}}" + ("$^\\dagger$" if x["name"] in DAGGER else "")
                      for x in sorted(g, key=lambda x: short(x["name"])))
    if len(out) > 5:
        out.append("\\specialrule{0.2pt}{1.4pt}{1.2pt}")   # hairline between ratios, same height as the old 1.5pt gap
    out.append(f"${ratio:.2f}\\times$ & {sci(r['gs_bound_min'])} & {sci(r['gs_bound_sublattice'])} & "
               f"{wcell} & {v} ({len(g)}) & {names} \\\\")
out += ["\\bottomrule", "\\end{tabular*}", "\\endgroup"]
(root / "analysis" / "output").mkdir(exist_ok=True)
(root / "analysis" / "output" / "tab_screen_body.tex").write_text("\n".join(out) + "\n")
print(f"wrote analysis/output/tab_screen_body.tex: {len(groups)} ratios covering {n_cfg} configurations", counts)
