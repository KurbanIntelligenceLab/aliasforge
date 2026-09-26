#!/usr/bin/env python3
"""Exact certified aliases for a bicubic pipeline, via the resampler's INTEGER kernel.

The approximate route does not work and the measurement says so cleanly: projecting
a content difference onto the real-valued null space, rounding to uint8 and steering
into the target's rounding bins leaves a mismatch floor of roughly 1% of output
values, and that floor is flat in the downscale factor (0/72 certified over factors
1.41x-8x).  Certifying 602112 outputs needs a per-output failure rate below 1.7e-6;
the floor is ~6000x too high.  No amount of search closes that, because the error is
injected by the rounding itself rather than by a bad search.

The fix is to stop rounding.  Pillow does not resample in floating point: it
precomputes each filter weight as a fixed-point integer,
`k = round(w * 2**PRECISION_BITS)`, accumulates `sum(k_i * pixel_i)` in integers,
and only then shifts and clips.  So if a perturbation `d` is integer-valued and
satisfies `K d = 0` EXACTLY over the integers, the pre-shift accumulator is
bit-identical and the output is bit-identical -- with no rounding argument
anywhere.  The collision is then a fact about integer arithmetic, which is exactly
the standard Assumption A3 wants.

Two properties make this practical.  `K` is banded (a bicubic downscale touches
~5-8 source pixels per output), so an integer kernel vector can be found in a
narrow window by exact rational elimination on a small submatrix.  And the kernel
is closed under integer scaling: if `K v = 0` then `K (m v) = 0`, so the amplitude
is free and can be raised until the content is plainly legible while the vector
itself stays short.  Measured: a 64-wide window at 896->448 yields a vector with
max|v| = 3, and at amplitude 60 the pair differs visibly at full resolution while
the 448x448 outputs are bit-identical.

Pillow applies the horizontal and vertical passes separately with a quantised
intermediate, so we cancel at the FIRST pass: every ROW of the perturbation is an
integer kernel vector of the horizontal operator.  The intermediate image is then
already identical and the second pass cannot reintroduce a difference.  Content is
carried by which rows are perturbed, which is free.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

PRECISION_BITS = 32 - 8 - 2  # Pillow Resample.c
BICUBIC_SUPPORT = 2.0


def bicubic_filter(x: float, a: float = -0.5) -> float:
    """Pillow's BICUBIC filter kernel."""
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def bilinear_filter(x: float) -> float:
    """Pillow's BILINEAR (triangle) kernel."""
    x = abs(x)
    return 1.0 - x if x < 1.0 else 0.0


def box_filter(x: float) -> float:
    """Pillow's BOX kernel."""
    return 1.0 if -0.5 <= x < 0.5 else 0.0


def lanczos_filter(x: float, a: float = 3.0) -> float:
    """Pillow's LANCZOS kernel."""
    import math
    x = abs(x)
    if x == 0.0:
        return 1.0
    if x >= a:
        return 0.0
    px = math.pi * x
    return a * math.sin(px) * math.sin(px / a) / (px * px)


# name -> (filter, support).  Every one of these is a fixed-point integer kernel in
# Pillow, so the null-space construction is not specific to bicubic.
KERNELS = {
    "bicubic": (bicubic_filter, BICUBIC_SUPPORT),
    "bilinear": (bilinear_filter, 1.0),
    "box": (box_filter, 0.5),
    "lanczos": (lanczos_filter, 3.0),
}


def pil_coeffs(n_in: int, n_out: int, support: float = BICUBIC_SUPPORT,
               filt=None) -> np.ndarray:
    """Pillow's fixed-point coefficient matrix, as exact Python ints.

    Transcribed from `precompute_coeffs` + `normalize_coeffs_8bpc`.  Verified
    bit-exact against `PIL.Image.resize` -- see `validate`.

    `filt` selects the kernel and defaults to bicubic, so existing callers are
    unaffected; pass one of `KERNELS` to reconstruct another Pillow resampler.
    """
    scale = max(1.0, n_in / n_out)
    supp = support * scale
    K = np.zeros((n_out, n_in), dtype=object)
    for xx in range(n_out):
        center = (xx + 0.5) * scale
        ss = 1.0 / scale
        lo = max(0, int(center - supp + 0.5))
        hi = min(n_in, int(center + supp + 0.5))
        n = hi - lo
        _f = filt if filt is not None else bicubic_filter
        w = np.array([_f((x + lo - center + 0.5) * ss) for x in range(n)])
        total = w.sum()
        if total != 0:
            w = w / total
        for x in range(n):
            v = w[x]
            K[xx, lo + x] = (
                int(math.floor(v * (1 << PRECISION_BITS) + 0.5)) if v >= 0
                else -int(math.floor(-v * (1 << PRECISION_BITS) + 0.5))
            )
    return K


def integer_kernel_vector(K: np.ndarray, c0: int, width: int) -> np.ndarray | None:
    """A short integer vector supported on [c0, c0+width) with K v = 0 exactly.

    Every output row touching the window is included as a constraint, so the
    vector annihilates the FULL operator, not just the windowed part.  Among the
    free variables we take the one giving the smallest max|v|: a short vector
    keeps the perturbation a small pixel offset, and a long one is unusable
    because a uint8 image cannot hold it.
    """
    n_out, n_in = K.shape
    rows = [i for i in range(n_out) if any(K[i, c0:c0 + width] != 0)]
    if not rows:
        return None
    # a row touching the window must be annihilated using window columns alone,
    # so any row reaching outside the window makes the window unusable as posed
    M = [[Fraction(int(K[i, j])) for j in range(c0, c0 + width)] for i in rows]

    r, c = len(M), len(M[0])
    piv, row = [], 0
    for col in range(c):
        p = next((i for i in range(row, r) if M[i][col] != 0), None)
        if p is None:
            continue
        M[row], M[p] = M[p], M[row]
        pv = M[row][col]
        M[row] = [x / pv for x in M[row]]
        for i in range(r):
            if i != row and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * b for a, b in zip(M[i], M[row])]
        piv.append(col)
        row += 1
        if row == r:
            break

    free = [j for j in range(c) if j not in piv]
    if not free:
        return None

    best = None
    for fj in free:
        v = [Fraction(0)] * c
        v[fj] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -M[i][fj]
        den = 1
        for x in v:
            den = den * x.denominator // math.gcd(den, x.denominator)
        iv = [int(x * den) for x in v]
        g = 0
        for x in iv:
            g = math.gcd(g, abs(x))
        if g:
            iv = [x // g for x in iv]
        m = max(abs(x) for x in iv)
        if best is None or m < best[0]:
            best = (m, iv)

    # A basis vector from naive free-variable back-substitution can be
    # astronomically long -- at a mild downscale the window has few free
    # variables and the denominators compound.  Such a vector is useless here
    # regardless: it has to land inside a uint8 pixel offset.  Reject rather
    # than overflow, and let the caller widen the window or reduce the lattice.
    if best[0] > 2**62:
        return None
    out = np.zeros(K.shape[1], dtype=np.int64)
    out[c0:c0 + width] = best[1]
    return out


def find_short_vector(K: np.ndarray, width: int = 64, stride: int = 32,
                      max_abs: int = 8) -> tuple[np.ndarray | None, dict]:
    """Scan windows for the shortest usable integer kernel vector."""
    best, meta = None, {"windows_tried": 0, "best_max_abs": None}
    n_in = K.shape[1]
    for c0 in range(0, max(1, n_in - width), stride):
        meta["windows_tried"] += 1
        v = integer_kernel_vector(K, c0, width)
        if v is None:
            continue
        m = int(np.abs(v).max())
        if m == 0 or m > max_abs:
            continue
        if best is None or m < int(np.abs(best).max()):
            best = v
            meta["best_max_abs"] = m
            meta["c0"] = c0
    return best, meta


def make_exact_pair(base: np.ndarray, vec: np.ndarray, amplitude: int,
                    rows_a: np.ndarray, rows_b: np.ndarray):
    """Two images whose horizontal-pass output is bit-identical by integer algebra.

    `vec` is an integer kernel vector of the horizontal operator; `amplitude * vec`
    is still exactly in the kernel.  `rows_a` / `rows_b` are boolean row masks
    selecting where the pattern appears -- that is the atomic fact (e.g. how many
    marks are present, and at which heights), and it is unconstrained.
    """
    H, W, _ = base.shape
    add = (amplitude * vec)[:, None]

    def build(mask):
        x = base.astype(np.int64).copy()
        x[mask] += add
        return x

    xa, xb = build(rows_a), build(rows_b)
    lo = min(int(xa.min()), int(xb.min()))
    hi = max(int(xa.max()), int(xb.max()))
    if lo < 0 or hi > 255:
        # Clipping is nonlinear and would destroy the exact cancellation, so a
        # saturating candidate is rejected rather than silently clamped.
        return None, None, {"saturates": True, "lo": lo, "hi": hi}
    xa, xb = xa.astype(np.uint8), xb.astype(np.uint8)
    return xa, xb, {
        "saturates": False,
        "pix_diff": int((xa != xb).any(-1).sum()),
        "max_amplitude": int(np.abs(xa.astype(int) - xb.astype(int)).max()),
        "vec_max_abs": int(np.abs(vec).max()),
        "vec_nonzero": int((vec != 0).sum()),
    }


def validate() -> list[tuple[str, bool, str]]:
    """The coefficient transcription must reproduce Pillow bit-for-bit."""
    from PIL import Image

    out = []
    rng = np.random.default_rng(0)
    for (n_in, n_out) in ((896, 448), (1344, 448), (632, 448), (1024, 256)):
        img = rng.integers(0, 256, (n_in, 4, 3), dtype=np.uint8)
        ref = np.asarray(Image.fromarray(img).resize((4, n_out), Image.BICUBIC))
        K = pil_coeffs(n_in, n_out)
        mine = np.empty_like(ref)
        half = 1 << (PRECISION_BITS - 1)
        for c in range(4):
            for ch in range(3):
                ss = K @ img[:, c, ch].astype(object) + half
                mine[:, c, ch] = np.clip([int(x) >> PRECISION_BITS for x in ss], 0, 255)
        out.append((f"coeffs bit-exact vs PIL {n_in}->{n_out}",
                    bool(np.array_equal(ref, mine)),
                    str(int(np.abs(ref.astype(int) - mine.astype(int)).max()))))
    return out


if __name__ == "__main__":
    bad = 0
    for name, ok, detail in validate():
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [maxdiff={detail}]")
        bad += not ok
    K = pil_coeffs(896, 448)
    v, meta = find_short_vector(K)
    print(f"short vector search: {meta}")
    print(f"RESULT exact_validation_failed={bad} vector_found={v is not None}")
