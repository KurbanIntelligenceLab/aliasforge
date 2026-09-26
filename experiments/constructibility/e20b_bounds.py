#!/usr/bin/env python3
"""E20b: certified lattice bounds for all four Pillow kernels, and a search-window sweep.

E20 reported the kernel table (Table tab:kernels) from a SEARCH: for each (kernel, ratio)
it looked for a short integer kernel vector among the free-variable back-substitutions of
64-wide windows and printed a dash when none came in under 255.  A search negative is not
the paper's stated methodology.  Section 5 argues from an a-priori CERTIFIED Gram-Schmidt
lower bound on the shortest lattice vector, and that bound existed only for bicubic (the
tool `verify/lattice_bounds.py` was hard-wired to the bicubic coefficients).  Section 5
also states that widening the search window from 32 to 256 leaves every verdict unchanged,
and no record backed that: every E20 shard ran width 64.

This experiment records both, per kernel and per ratio, and it does one more thing that
turned out to matter.  The basis `lattice_bounds.py` bounds is `kernel_basis`, whose vectors
are rational free-variable back-substitutions with denominators cleared.  Their Z-span is a
SUBLATTICE of the integer kernel {v in Z^W : K v = 0}, and at non-dyadic ratios its index is
astronomical (index^2 ~ 1e100 at bicubic 4x).  A Gram-Schmidt bound on a sublattice does not
bound the lattice.  So the certified column here is computed on a Z-basis of the FULL integer
kernel (Hermite-style unimodular reduction, then float-guided LLL kept exact, then the exact
Gram-Schmidt bound), and the sublattice number the old tool would have printed is recorded
beside it for reconciliation.  The reduced basis also yields a witness vector, which is put
through the same Pillow bit-exact pair certification E20 uses, so a case the free-variable
search misses is not reported as a dash.

Per (kernel, n_in -> n_out):
  gate      the reconstructed fixed-point operator must reproduce PIL.Image.resize bit for
            bit on a random image (E20's gate); n/a everywhere when it fails.
  bound     min over 32-wide windows at stride 16 of the certified lower bound on the
            sup-norm of any nonzero integer kernel vector supported on the window.
  sweep     E20's own search (integer_kernel_vector over windows at stride 16, max_abs 255)
            at widths 32, 64, 128, 256: shortest sup-norm, its level span, and the verdict
            (certified / dagger / dash), then whether the verdict is width-invariant.

CPU only; no model weights, no dataset.  One shard per kernel.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "verify"))
sys.path.insert(0, HERE)

from aliasforge.exact import KERNELS, integer_kernel_vector  # noqa: E402
from aliasforge.lattice import kernel_basis  # noqa: E402
from e20_resamplers import bit_exact, certify_pair, reconstruct  # noqa: E402
from lattice_bounds import certified_window_bound, gs_bound_exact  # noqa: E402

VERSION = "e20b.1"
COLUMNS = ("kernel", "n_in", "n_out", "ratio", "gate_ok", "gs_bound_min", "gs_windows",
           "width", "shortest_maxabs", "shortest_span", "verdict", "width_invariant",
           "elapsed_s",
           # reconciliation columns, appended after the spec'd thirteen
           "gs_bound_sublattice", "lll_shortest_maxabs", "lll_shortest_span", "lll_verdict")


def windows(n_in, width, stride):
    return list(range(0, max(1, n_in - width), stride))


def verdict_for(name, n_in, n_out, v, rng):
    """Three-way verdict for a vector with ||v||_inf <= 255, plus its level span."""
    v = np.asarray([int(x) for x in v], dtype=np.int64)
    span = int(v.max() - v.min())
    if span > 255:
        return "dagger", span                   # under the bound, but no 8-bit pedestal fits
    cert, _amp, _pix = certify_pair(name, n_in, n_out, v, rng)
    return ("certified" if cert else "uncertified"), span


def certified_bound(K, name, n_in, n_out, rng, width=32, stride=16):
    """Sound certified bound over 32-wide windows, with the sublattice number beside it."""
    lbs, sub, wit = [], [], None
    for c0 in windows(n_in, width, stride):
        r = certified_window_bound(K, c0, width)
        if r is None:
            continue
        lbs.append(r["bound"])
        B = kernel_basis(K, c0, width)
        b = gs_bound_exact(B) if B else None
        if b is not None:
            sub.append(b)
        if wit is None or r["witness_maxabs"] < wit[0]:
            full = np.zeros(n_in, dtype=object)
            full[c0:c0 + width] = r["witness"]
            wit = (r["witness_maxabs"], full)
    if not lbs:
        return "trivial kernel", 0, "trivial kernel", "", "", "-"
    lll_max, lll_span, lll_verdict = wit[0], "", "-"
    if lll_max <= 255:
        lll_verdict, lll_span = verdict_for(name, n_in, n_out, wit[1], rng)
    return (f"{min(lbs):.6g}", len(lbs), f"{min(sub):.6g}" if sub else "",
            lll_max, lll_span, lll_verdict)


def search(K, name, n_in, n_out, width, max_abs, rng, stride=16):
    """E20's search at one width: shortest sup-norm under max_abs, its span, the verdict."""
    best = None
    ws = windows(n_in, width, stride)
    for c0 in ws:
        v = integer_kernel_vector(K, c0, width)
        if v is not None and 0 < int(np.abs(v).max()) <= max_abs:
            if best is None or int(np.abs(v).max()) < int(np.abs(best).max()):
                best = v
    if best is None:
        return "", "", "-", len(ws)
    verdict, span = verdict_for(name, n_in, n_out, best, rng)
    return int(np.abs(best).max()), span, verdict, len(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--kernels", default="bicubic,bilinear,box,lanczos")
    ap.add_argument("--sizes", default="896:448,896:224,672:224,1344:448,896:112,1344:224,672:448,896:298")
    ap.add_argument("--widths", default="32,64,128,256")
    ap.add_argument("--max-abs", type=int, default=255)
    ap.add_argument("--bound-width", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    kernels = [k for i, k in enumerate(a.kernels.split(",")) if i % a.shards == a.task]
    pairs = [tuple(int(x) for x in s.split(":")) for s in a.sizes.split(",")]
    widths = [int(w) for w in a.widths.split(",")]
    print(f"exp=e20b_bounds version={VERSION} task={a.task}/{a.shards} kernels={kernels} "
          f"sizes={a.sizes} widths={widths} max_abs={a.max_abs} bound_width={a.bound_width} "
          f"stride={a.stride}", flush=True)

    rows = [COLUMNS]
    n_cases = n_gate = n_inv = 0
    for name in kernels:
        for n_in, n_out in pairs:
            t0 = time.time()
            ratio = round(n_in / n_out, 4)
            K = reconstruct(name, n_in, n_out)
            ok = bit_exact(name, n_in, n_out, K, rng)
            n_cases += 1
            if not ok:
                for w in widths:
                    rows.append((name, n_in, n_out, ratio, 0, "n/a", 0, w, "", "", "n/a", "n/a",
                                 round(time.time() - t0, 1), "n/a", "", "", "n/a"))
                print(f"  {name:8} {n_in:>5}->{n_out:<4} gate=FAIL  (excluded)", flush=True)
                continue
            n_gate += 1
            gs_min, gs_n, gs_sub, lll_max, lll_span, lll_v = certified_bound(
                K, name, n_in, n_out, rng, a.bound_width, a.stride)
            t_bound = time.time() - t0
            per_width = []
            for w in widths:
                t1 = time.time()
                short, span, verdict, nw = search(K, name, n_in, n_out, w, a.max_abs, rng, a.stride)
                per_width.append([w, short, span, verdict, nw, time.time() - t1])
            verdicts = {p[3] for p in per_width}
            inv = int(len(verdicts) == 1)
            n_inv += inv
            for w, short, span, verdict, nw, dt in per_width:
                rows.append((name, n_in, n_out, ratio, 1, gs_min, gs_n, w, short, span, verdict,
                             inv, round(dt, 1), gs_sub, lll_max, lll_span, lll_v))
            sweep = " ".join(f"w{w}:{short if short != '' else '-'}/{verdict}" for w, short, _, verdict, _, _ in per_width)
            print(f"  {name:8} {n_in:>5}->{n_out:<4} ratio={ratio:<7} gate=ok bound={gs_min} "
                  f"(sublattice={gs_sub}, windows={gs_n}, {t_bound:.0f}s) lll_witness={lll_max}/{lll_v} "
                  f"sweep[{sweep}] width_invariant={inv} search_windows={per_width[-1][4]}"
                  f"..{per_width[0][4]} total={time.time() - t0:.0f}s", flush=True)

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    body = rows[1:]
    cert = sum(1 for r in body if r[10] == "certified")
    lll_cert = sum(1 for r in body if r[7] == widths[0] and r[16] == "certified")
    print(f"RESULT task={a.task} kernels={','.join(kernels)} cases={n_cases} gate_ok={n_gate} "
          f"width_invariant={n_inv} rows={len(body)} search_certified_rows={cert} "
          f"lll_certified_cases={lll_cert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
