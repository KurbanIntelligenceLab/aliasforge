#!/usr/bin/env python3
"""Emit the body of the kernel-susceptibility table from the E20b records (sound lattice).

Cell states, per (kernel, ratio):
    v            an LLL witness on the TRUE integer kernel with sup-norm v whose constructed
                 pair was verified bit-identical after resizing (Pillow)
    v^dag        a witness with sup-norm v <= 255 whose level span exceeds 255, so no 8-bit
                 pedestal fits it and no pair can be built from it (no certified pair found)
    >= b         no realizable witness, and the sound Gram-Schmidt bound b over 32-wide
                 windows exceeds 255: a certified (window-local) negative
    --           no realizable witness and the bound is at most 255: undetermined
    n/a          the reconstructed operator failed the bit-exact gate against the library

Two size pairs share the 3x column; a column takes the strongest positive state, else the
strongest negative. Writes generated/tab_kernels_body.tex (whole tabular), which main.tex inputs.
Records: results/constructibility/e20b-bounds/sh_*.csv (width-32 rows carry the LLL columns).
"""
import csv, glob, math, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(root / "results/constructibility/e20b-bounds/sh_*.csv")))
rows = [r for f in files for r in csv.DictReader(open(f)) if r.get("kernel")]
if not rows: sys.exit("no e20b records")
rows = [r for r in rows if r.get("width") in ("32", "")]          # one record per (kernel, size)
ks = ["bicubic", "bilinear", "box", "lanczos"]
ratios = sorted({round(float(r["ratio"]), 2) for r in rows})
def state(r):
    if r["gate_ok"] != "1": return ("na", None)
    v = r.get("lll_shortest_maxabs", "")
    if r.get("lll_verdict") == "certified": return ("cert", int(v))
    if r.get("lll_verdict") == "dagger": return ("dag", int(v))
    b = r.get("gs_bound_min", "")
    try: b = float(b)
    except ValueError: b = None
    if b is not None and b > 255: return ("neg", b)
    return ("und", None)
rank = {"cert": 0, "dag": 1, "neg": 2, "und": 3, "na": 4}
def render(s):
    k, v = s
    if k == "cert": return f"${v}$"
    if k == "dag": return f"${v}^\\dagger$"
    if k == "neg":
        e = int(math.floor(math.log10(v))); m = v / 10**e
        return f"$\\ge {m:.1f}{{\\times}}10^{{{e}}}$" if e >= 3 else f"$\\ge {v:.0f}$"
    if k == "und": return "{--}"
    return "{n/a}"
out = ["\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l *{%d}{c}@{}}" % len(ratios), "\\toprule",
       "Kernel & " + " & ".join(f"${x:g}\\times$" for x in ratios) + " \\\\", "\\midrule"]
summary = {}
for k in ks:
    cells = []
    for x in ratios:
        m = [state(r) for r in rows if r["kernel"] == k and abs(round(float(r["ratio"]), 2) - x) < 1e-9]
        best = min(m, key=lambda s: (rank[s[0]], s[1] if s[0] in ("cert", "dag") else 0)) if m else ("und", None)
        cells.append(render(best)); summary[(k, x)] = best[0]
    out.append(f"\\textsf{{{k}}} & " + " & ".join(cells) + " \\\\")
out += ["\\bottomrule", "\\end{tabular*}"]
(root / "analysis" / "output").mkdir(exist_ok=True)
(root / "analysis" / "output" / "tab_kernels_body.tex").write_text("\n".join(out) + "\n")
print("wrote analysis/output/tab_kernels_body.tex")
for k in ks: print(k, {x: summary[(k, x)] for x in ratios})
