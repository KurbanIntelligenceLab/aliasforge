#!/usr/bin/env python3
"""E10b: the constructibility screen (E10) recomputed on the TRUE integer kernel lattice.

E10 screened the deployed resizes in `resize_configs.txt` with a Gram-Schmidt bound on
`aliasforge.lattice.kernel_basis`.  That basis spans a proper SUBLATTICE of the integer kernel
{v in Z^W : K v = 0} (index ~1e500 at non-dyadic ratios), so a bound on it says nothing about
the lattice and every E10 "not-eligible" verdict was unsound.  On the full lattice, bicubic
1344->448 has a witness of sup-norm 13 and 896->224 one of sup-norm 5, both verified to
collide under Pillow.

This re-runs the screen with E20b's sound instrument, per config (all Pillow bicubic):
  gate       the reconstructed fixed-point operator must reproduce PIL.Image.resize bit for
             bit (E20's gate); "n/a" everywhere when it fails.
  bound      min over 32-wide windows at stride 16 of the exact Gram-Schmidt lower bound on
             the FULL integer kernel (Hermite-style Z-basis, float-guided LLL kept exact),
             with the old sublattice number recorded beside it for reconciliation.
  witness    the shortest LLL-reduced vector: its sup-norm, level span and Pillow-verified
             verdict (certified / dagger / uncertified / dash).
  search64   E20's free-variable search at width 64 (the weak instrument), for comparison.

Several config names share one size; each unique size is computed once and one row is
emitted per NAME.  Unique sizes are sharded round-robin over --shards by --task.
CPU only; no model weights, no dataset.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "verify"))
sys.path.insert(0, HERE)

from e20_resamplers import bit_exact, reconstruct  # noqa: E402
from e20b_bounds import certified_bound, search  # noqa: E402

VERSION = "e10b.1"
KERNEL = "bicubic"
COLUMNS = ("name", "n_in", "n_out", "ratio", "dyadic", "integral", "gate_ok",
           "gs_bound_min", "gs_windows", "gs_bound_sublattice",
           "lll_maxabs", "lll_span", "lll_verdict",
           "search64_maxabs", "search64_verdict", "elapsed_s")


def read_configs(path):
    """(name, n_in, n_out) per non-comment line, in file order."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, n_in, n_out = (x.strip() for x in line.split(","))
            out.append((name, int(n_in), int(n_out)))
    return out


def is_dyadic(n_in, n_out):
    r = n_in / n_out
    return abs(math.log2(r) - round(math.log2(r))) < 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=os.path.join(HERE, "resize_configs.txt"))
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--names", default="",
                    help="comma list restricting the screen to these config names (self-test)")
    ap.add_argument("--bound-width", type=int, default=32)
    ap.add_argument("--search-width", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-abs", type=int, default=255)
    a = ap.parse_args()

    configs = read_configs(a.configs)
    if a.names:
        keep = set(a.names.split(","))
        configs = [c for c in configs if c[0] in keep]
    sizes = []                                   # unique (n_in, n_out), first-appearance order
    for _, n_in, n_out in configs:
        if (n_in, n_out) not in sizes:
            sizes.append((n_in, n_out))
    mine = [s for i, s in enumerate(sizes) if i % a.shards == a.task]
    rng = np.random.default_rng(0)
    print(f"exp=e10b_screen version={VERSION} task={a.task}/{a.shards} kernel={KERNEL} "
          f"configs={len(configs)} unique_sizes={len(sizes)} my_sizes={len(mine)} "
          f"bound_width={a.bound_width} search_width={a.search_width} stride={a.stride} "
          f"max_abs={a.max_abs} configs_file={os.path.relpath(a.configs)}", flush=True)

    rows = [COLUMNS]
    n_sizes = n_gate = n_lll_cert = n_s64_cert = n_bound_le255 = 0
    for n_in, n_out in mine:
        t0 = time.time()
        names = [c[0] for c in configs if (c[1], c[2]) == (n_in, n_out)]
        ratio = f"{n_in / n_out:.4f}"
        dy, integral = int(is_dyadic(n_in, n_out)), int(n_in % n_out == 0)
        K = reconstruct(KERNEL, n_in, n_out)
        ok = bit_exact(KERNEL, n_in, n_out, K, rng)
        n_sizes += 1
        if not ok:
            dt = round(time.time() - t0, 1)
            for name in names:
                rows.append((name, n_in, n_out, ratio, dy, integral, 0, "n/a", 0, "n/a",
                             "", "", "n/a", "", "n/a", dt))
            print(f"  {n_in:>5}->{n_out:<4} ratio={ratio:<7} gate=FAIL (excluded) "
                  f"names={','.join(names)}", flush=True)
            continue
        n_gate += 1
        gs_min, gs_n, gs_sub, lll_max, lll_span, lll_v = certified_bound(
            K, KERNEL, n_in, n_out, rng, a.bound_width, a.stride)
        t_bound = time.time() - t0
        s64_max, _s64_span, s64_v, s64_nw = search(
            K, KERNEL, n_in, n_out, a.search_width, a.max_abs, rng, a.stride)
        dt = round(time.time() - t0, 1)
        n_lll_cert += lll_v == "certified"
        n_s64_cert += s64_v == "certified"
        try:
            n_bound_le255 += float(gs_min) <= a.max_abs
        except ValueError:
            pass
        for name in names:
            rows.append((name, n_in, n_out, ratio, dy, integral, 1, gs_min, gs_n, gs_sub,
                         lll_max, lll_span, lll_v, s64_max, s64_v, dt))
        print(f"  {n_in:>5}->{n_out:<4} ratio={ratio:<7} dyadic={dy} integral={integral} gate=ok "
              f"bound={gs_min} (sublattice={gs_sub}, windows={gs_n}, {t_bound:.0f}s) "
              f"lll_witness={lll_max}/span={lll_span}/{lll_v} "
              f"search64={s64_max if s64_max != '' else '-'}/{s64_v} (windows={s64_nw}) "
              f"names={','.join(names)} total={dt}s", flush=True)

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    body = rows[1:]
    print(f"RESULT task={a.task} sizes={n_sizes} gate_ok={n_gate} rows={len(body)} "
          f"bound_le_{a.max_abs}={n_bound_le255} lll_certified_sizes={n_lll_cert} "
          f"search64_certified_sizes={n_s64_cert} "
          f"lll_certified_rows={sum(1 for r in body if r[12] == 'certified')} "
          f"search64_certified_rows={sum(1 for r in body if r[14] == 'certified')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
