#!/usr/bin/env python3
"""E28: a certified alias on VisualPRM-8B's DEPLOYED preprocessing path, on the sound lattice.

The paper excluded VisualPRM-8B "by the bound".  That bound was computed on
`aliasforge.lattice.kernel_basis`, whose Z-span is a proper sublattice of the integer
kernel (index ~1e500), so the exclusion certified nothing.  An earlier dead-region
attempt on this target was retracted because it went through AutoProcessor's CLIP
path, which is not what the model is fed.  This experiment redoes the target on the
path that is actually deployed (`aliasforge.internvl`, transcribed from the OpenGVLab
sources: `dynamic_preprocess` with max_num=12, image_size=448, use_thumbnail=True)
and on the FULL integer kernel (`verify/lattice_bounds.py`).

Construction.  For a square input of side S the deployed rule picks the 3x3 grid, so
the model sees (a) the whole image resized S -> 1344 and cut into nine identity tiles
and (b) a thumbnail of the whole image resized S -> 448.  Both resizes are Pillow's
default resample (BICUBIC, recorded at run time) applied to the SAME source columns,
and Pillow runs the horizontal pass first in fixed-point integers.  A row
perturbation v that is an exact integer kernel vector of BOTH horizontal operators
leaves both accumulators bit-identical, so the whole interface state collides.  The
two coefficient matrices are stacked into one integer matrix K_stacked; the true
integer kernel of a window is taken with `integer_kernel_basis`, LLL-reduced, and its
shortest vector is the witness.  If the witness fits an 8-bit image the pair is built
with `make_exact_pair` (row masks as in E0a, scaled to S), both members are run
through `internvl.interface_state`, and the digests are compared.  The exact
Gram-Schmidt bound (min over windows) is recorded alongside, so a negative is a
window-local certificate on the sound lattice.

CPU only; no model weights, no dataset.  Sides are sharded over --shards by --task.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "verify"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "constructibility"))

from aliasforge import internvl  # noqa: E402
from aliasforge.exact import PRECISION_BITS, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.interface import per_field_report, state_digest  # noqa: E402
from e20_resamplers import bit_exact  # noqa: E402
from lattice_bounds import gs_bound_exact, integer_kernel_basis, lll_reduce  # noqa: E402

VERSION = "e28.1"
COLUMNS = ("side", "grid", "n_tiles", "primary_n_out", "primary_ratio", "thumb_ratio", "width",
           "gate_primary", "gate_thumb", "gs_bound_min", "gs_windows", "witness_maxabs",
           "witness_span", "pair_built", "digests_collide", "fields_differ", "pix_diff",
           "amplitude", "elapsed_s")
SIDES = (1792, 2016, 2240, 2688, 3136, 3584, 4032, 4480)
TILE = internvl.IMAGE_SIZE
BIG_BITS = 500          # above this the float copy inside lll_reduce would overflow


# --------------------------------------------------------------------------- path facts
def default_resample_name():
    """Which resampler `Image.resize` uses when the deployed code passes none."""
    from PIL import Image
    rng = np.random.default_rng(1)
    im = Image.fromarray(rng.integers(0, 256, (64, 900, 3), dtype=np.uint8))
    ref = np.asarray(im.resize((300, 64)))
    hits = [n for n in ("NEAREST", "BOX", "BILINEAR", "HAMMING", "BICUBIC", "LANCZOS")
            if np.array_equal(ref, np.asarray(im.resize((300, 64), getattr(Image, n))))]
    return "+".join(hits) if hits else "none"


def gate_default(K, n_in, n_out, rng):
    """The transcribed operator must reproduce the DEPLOYED call (no resample argument)."""
    from PIL import Image
    img = rng.integers(0, 256, (n_in, 4, 3), dtype=np.uint8)
    ref = np.asarray(Image.fromarray(img).resize((4, n_out)))
    mine = np.empty_like(ref)
    half = 1 << (PRECISION_BITS - 1)
    for c in range(4):
        for ch in range(3):
            ss = K @ img[:, c, ch].astype(object) + half
            mine[:, c, ch] = np.clip([int(x) >> PRECISION_BITS for x in ss], 0, 255)
    return bool(np.array_equal(ref, mine))


def deployed_grid(side):
    """The grid `dynamic_preprocess` realizes for a side x side image, and its tile count."""
    from PIL import Image
    ratios = sorted({(i, j) for n in range(internvl.MIN_NUM, internvl.MAX_NUM + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if internvl.MIN_NUM <= i * j <= internvl.MAX_NUM}, key=lambda x: x[0] * x[1])
    g = internvl.find_closest_aspect_ratio(1.0, ratios, side, side, TILE)
    tiles = internvl.dynamic_preprocess(Image.new("RGB", (side, side)), use_thumbnail=True)
    sizes_ok = all(t.size == (TILE, TILE) for t in tiles)
    return g, len(tiles), sizes_ok


def row_masks(H, n_a=3, n_b=2):
    """E0a's atomic fact -- how many horizontal marks are present -- scaled to side H."""
    a = np.zeros(H, bool)
    b = np.zeros(H, bool)
    for i, y0 in enumerate((int(.13 * H), int(.38 * H), int(.63 * H))):
        if i < n_a:
            a[y0:y0 + int(.12 * H)] = True
        if i < n_b:
            b[y0:y0 + int(.12 * H)] = True
    return a, b


# --------------------------------------------------------------------------- lattice
def _prereduce_scaled(basis, delta=0.99, max_iter=20000):
    """lll_reduce's float-guided sweep with a uniformly scaled float copy.

    Only used when the Hermite basis has entries too wide for float64; the scaling
    changes neither mu nor the Lovasz test, and every integer step is exact, so the
    output is a basis of the same lattice.  Stops once entries fit the canonical tool.
    """
    B = [list(map(int, b)) for b in basis]
    n = len(B)
    if n < 2:
        return B

    def fl():
        bits = max(abs(x).bit_length() for b in B for x in b)
        s = 1 << max(0, bits - 200)
        return np.array([[x / s for x in b] for b in B], dtype=float)

    def gs():
        _, R = np.linalg.qr(fl().T)
        d = np.diag(R)
        return d, (R / np.where(d == 0, 1.0, d)[:, None]).T

    k, it = 1, 0
    while k < n and it < max_iter:
        it += 1
        d, mu = gs()
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k, j]))
            if q:
                B[k] = [a - q * b for a, b in zip(B[k], B[j])]
                mu[k, :j + 1] -= q * mu[j, :j + 1]
        d, mu = gs()
        if d[k] ** 2 >= (delta - mu[k, k - 1] ** 2) * d[k - 1] ** 2:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            k = max(k - 1, 1)
        if it % 50 == 0 and max(abs(x).bit_length() for b in B for x in b) < BIG_BITS:
            break
    return B


_K = None   # the stacked matrix, inherited by forked workers


def window_bound(args):
    """Sound bound + shortest reduced vector for one window of the stacked operator."""
    c0, width = args
    K = _K
    S = integer_kernel_basis(K, c0, width)
    if not S:
        return c0, None
    if max(abs(x).bit_length() for b in S for x in b) >= BIG_BITS:
        S = _prereduce_scaled(S)
    S, meta = lll_reduce(S)
    Kw = K[:, c0:c0 + width]
    for v in S:
        if any(int(x) != 0 for x in (Kw.astype(object) @ np.array(v, dtype=object))):
            raise AssertionError(f"reduced basis vector not in the stacked kernel at c0={c0}")
    wit = min((b for b in S if any(b)), key=lambda b: max(abs(x) for x in b))
    return c0, {"bound": gs_bound_exact(S), "dim": len(S), "witness": wit,
                "maxabs": max(abs(x) for x in wit), "converged": meta["converged"]}


def sweep(K, side, width, stride, workers):
    global _K
    _K = K
    c0s = list(range(0, max(1, side - width), stride))
    jobs = [(c0, width) for c0 in c0s]
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            out = pool.map(window_bound, jobs, chunksize=1)
    else:
        out = []
        for i, j in enumerate(jobs):
            out.append(window_bound(j))
            if (i + 1) % 50 == 0:
                print(f"    progress side={side} width={width} windows={i + 1}/{len(jobs)}",
                      flush=True)
    lbs, best, n_unconv = [], None, 0
    for c0, r in out:
        if r is None:
            continue
        lbs.append(r["bound"])
        n_unconv += not r["converged"]
        if best is None or r["maxabs"] < best[1]:
            best = (c0, r["maxabs"], r["witness"], r["dim"])
    return len(c0s), lbs, best, n_unconv


# --------------------------------------------------------------------------- the pair
def build_and_digest(side, vec, rng, noise=10):
    """Pair around a saturating-safe base; both members through the deployed path."""
    v = np.asarray([int(x) for x in vec], dtype=np.int64)
    span = int(v.max() - v.min())
    for nz in (noise, 0):
        if span > 255 - 2 * nz:
            continue
        amp = max(1, min(60, (255 - 2 * nz) // span))
        w = amp * v
        lo, hi = int(w.min()), int(w.max())
        ped = (-lo + (255 - hi)) // 2
        base = (ped + rng.integers(-nz, nz + 1, (side, side, 3))).astype(np.int64)
        ma, mb = row_masks(side)
        xa, xb, meta = make_exact_pair(base, v, amp, ma, mb)
        if xa is None:
            continue
        sa, sb = internvl.interface_state(xa), internvl.interface_state(xb)
        rep = per_field_report(sa, sb)
        collide = int(state_digest(sa) == state_digest(sb) and rep["collides"])
        # control: the digest is not vacuous -- one pixel off the base must change it
        xc = xa.copy()
        xc[side // 2, side // 2, 0] ^= 1
        ctrl = int(state_digest(internvl.interface_state(xc)) != state_digest(sa))
        return {"pair_built": 1, "amplitude": amp, "pix_diff": meta["pix_diff"],
                "collide": collide, "differ": ";".join(rep["differ"] + rep["missing"]),
                "n_tiles": sa["n_tiles"], "control_ok": ctrl}
    return {"pair_built": 0, "amplitude": 0, "pix_diff": 0, "collide": 0, "differ": "",
            "n_tiles": "", "control_ok": ""}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--sides", default=",".join(map(str, SIDES)))
    ap.add_argument("--widths", default="32,64")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    sides = [s for i, s in enumerate(int(x) for x in a.sides.split(",")) if i % a.shards == a.task]
    widths = [int(w) for w in a.widths.split(",")]
    import PIL
    resample = default_resample_name()
    print(f"exp=e28_visualprm version={VERSION} task={a.task}/{a.shards} sides={sides} "
          f"widths={widths} stride={a.stride} workers={a.workers} pillow={PIL.__version__} "
          f"deployed_resample={resample} tile={TILE} min_num={internvl.MIN_NUM} "
          f"max_num={internvl.MAX_NUM} thumbnail=1", flush=True)
    val = internvl.validate()
    n_bad = sum(1 for _, ok, _ in val if not ok)
    for name, ok, detail in val:
        print(f"  transcription_check {'PASS' if ok else 'FAIL'} {name} [{detail}]")
    print(f"  transcription_failed={n_bad}", flush=True)

    rows = [COLUMNS]
    n_cases = n_pairs = n_collide = 0
    for side in sides:
        (gx, gy), n_tiles, sizes_ok = deployed_grid(side)
        grid = f"{gx}x{gy}"
        n_out = TILE * gx
        r_primary, r_thumb = round(side / n_out, 4), round(side / TILE, 4)
        K1, K2 = pil_coeffs(side, n_out), pil_coeffs(side, TILE)
        g1 = int(gate_default(K1, side, n_out, rng) and bit_exact("bicubic", side, n_out, K1, rng))
        g2 = int(gate_default(K2, side, TILE, rng) and bit_exact("bicubic", side, TILE, K2, rng))
        print(f"  side={side} grid={grid} n_tiles={n_tiles} tiles_448={int(sizes_ok)} "
              f"primary={side}->{n_out} ({r_primary}x) thumb={side}->{TILE} ({r_thumb}x) "
              f"gate_primary={g1} gate_thumb={g2}", flush=True)
        K = np.vstack([K1, K2])
        for width in widths:
            t0 = time.time()
            n_cases += 1
            if not (g1 and g2 and sizes_ok and n_tiles == gx * gy + 1):
                rows.append((side, grid, n_tiles, n_out, r_primary, r_thumb, width, g1, g2, "n/a",
                             0, "", "", 0, 0, "", 0, 0, round(time.time() - t0, 1)))
                print(f"    width={width} excluded (gate or grid check failed)", flush=True)
                continue
            n_win, lbs, best, n_unconv = sweep(K, side, width, a.stride, a.workers)
            if not lbs:
                rows.append((side, grid, n_tiles, n_out, r_primary, r_thumb, width, g1, g2, "inf",
                             0, "", "", 0, 0, "", 0, 0, round(time.time() - t0, 1)))
                print(f"    width={width} windows={n_win} stacked kernel trivial in every window "
                      f"(certified: no window-supported integer vector) {time.time() - t0:.0f}s",
                      flush=True)
                continue
            c0, maxabs, wit, dim = best
            full = np.zeros(side, dtype=object)
            full[c0:c0 + width] = wit
            span = int(max(wit) - min(wit))
            res = {"pair_built": 0, "amplitude": 0, "pix_diff": 0, "collide": 0, "differ": "",
                   "control_ok": ""}
            if maxabs <= 255 and span <= 255:
                res = build_and_digest(side, full, rng)
                n_pairs += res["pair_built"]
                n_collide += res["collide"]
            rows.append((side, grid, n_tiles, n_out, r_primary, r_thumb, width, g1, g2,
                         f"{min(lbs):.6g}", len(lbs), maxabs, span, res["pair_built"],
                         res["collide"], res["differ"], res["pix_diff"], res["amplitude"],
                         round(time.time() - t0, 1)))
            print(f"    width={width} windows={n_win} nontrivial={len(lbs)} bound_min={min(lbs):.6g} "
                  f"lll_unconverged={n_unconv} witness_maxabs={maxabs} span={span} c0={c0} dim={dim} "
                  f"pair_built={res['pair_built']} amp={res['amplitude']} pix_diff={res['pix_diff']} "
                  f"digests_collide={res['collide']} fields_differ={res['differ'] or '-'} "
                  f"control_ok={res['control_ok']} {time.time() - t0:.0f}s", flush=True)

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    body = rows[1:]
    print(f"RESULT task={a.task} sides={','.join(map(str, sides))} transcription_failed={n_bad} "
          f"cases={len(body)} gated={sum(1 for r in body if r[7] and r[8])} pairs_built={n_pairs} "
          f"collisions={n_collide} rows={len(body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
