#!/usr/bin/env python3
"""Drive a null-space construction toward an EXACT interface collision.

Status: this closes the gap for a box/area downscale (exactly, in integer
arithmetic) and gets to within 2 of 9408 output values for bicubic, bilinear and
Lanczos -- but it does NOT yet reach exactness for those three.  That distinction
is the whole point and is not smoothed over anywhere: a pair with two differing
output values is NOT certified, because Proposition 1(b) says that once the
states differ at all, the attainable accuracy is unconstrained and no tolerance
certifies anything.  Near-collisions belong in the GRADED set, never the
certified one.

The pipeline, in the order it is applied:

  1. PROJECT.  For a separable downscale D(X) = A X B^T, the set {X : D(X) = 0}
     is {X : P_A X P_B = 0} for the row-space projectors P_A = A^+A, P_B = B^+B,
     and the orthogonal projection of a desired difference d onto it is
     d - P_A d P_B.  This is exact in real arithmetic: measured residual 2e-13.
     A rank-1 carrier is a special case and a much weaker one -- it forces the
     content to be a single outer product, whereas the projection keeps most of
     an arbitrary content image.

  2. RETARGET TO BIN CENTRES.  Integer input is the real obstacle: rounding the
     projected perturbation to uint8 injects ~0.5 of noise, which is exactly the
     uint8 quantum, so aiming at D(x0)'s exact float value straddles the rounding
     boundary.  Aiming instead at round(D(x0)) -- the centre of the bin the
     production pipeline will land in -- buys a full +/-0.5 of slack and cuts the
     residual by an order of magnitude (218 -> 9 differing values on bicubic).

  3. LOCAL SEARCH against the REAL operator.  The probed model A is a float
     surrogate; PIL's uint8 path uses fixed-point integer arithmetic, so a
     residual survives step 2 that no amount of work on the surrogate can remove.
     The search adjusts free source pixels -- never the content region, whose
     integrity is what makes the two members carry opposite labels -- and is
     scored by the true operator.  It reaches 2/9408 and stalls there.

Why it stalls is the open question: the last values sit where the support window
is clipped at the image border, and the moves that fix one break the other.
"""

from __future__ import annotations

import numpy as np


def project_to_nullspace(d: np.ndarray, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Orthogonal projection of a desired difference onto null(D), D(X)=A X B^T."""
    if B is None:
        B = A
    PA = np.linalg.pinv(A) @ A
    PB = np.linalg.pinv(B) @ B
    return d - PA @ d @ PB


def snap_to_bins(
    base: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    resize_u8,
    max_iters: int = 60,
    slack: float = 0.45,
) -> np.ndarray:
    """Round to uint8 while steering the downscale into x0's rounding bins."""
    target = resize_u8(base).astype(np.float64)
    Ap = np.linalg.pinv(A)
    x = base.astype(np.float64) + (delta[..., None] if delta.ndim == 2 else delta)
    for _ in range(max_iters):
        xi = np.clip(np.rint(x), 0, 255)
        r = np.stack([A @ xi[..., c] @ A.T for c in range(xi.shape[2])], -1) - target
        if np.abs(r).max() < slack:
            break
        x = x - np.stack([Ap @ r[..., c] @ Ap.T for c in range(r.shape[2])], -1)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def local_search(
    x0: np.ndarray,
    x1: np.ndarray,
    resize_u8,
    free_mask: np.ndarray,
    passes: int = 6,
    deltas=(1, -1, 2, -2),
) -> tuple[np.ndarray, dict]:
    """Greedy integer search against the true operator, from a near-collision.

    Scored by the real downscale, so it is immune to any mismatch between the
    probed surrogate and the production arithmetic.  Only free pixels move.
    """
    target = resize_u8(x0).astype(np.int64)
    H, W = x1.shape[:2]
    h, w = target.shape[:2]
    sy, sx = H / h, W / w
    x1 = x1.copy()
    cur = int((resize_u8(x1).astype(np.int64) != target).sum())
    start = cur

    for p in range(passes):
        if cur == 0:
            break
        for (oy, ox, oc) in np.argwhere(resize_u8(x1).astype(np.int64) != target):
            y0, y1_ = max(0, int(oy * sy) - 3), min(H, int((oy + 1) * sy) + 3)
            x0_, x1_ = max(0, int(ox * sx) - 3), min(W, int((ox + 1) * sx) + 3)
            best = None
            for yy in range(y0, y1_):
                for xx in range(x0_, x1_):
                    if not free_mask[yy, xx]:
                        continue
                    for dv in deltas:
                        v = int(x1[yy, xx, oc]) + dv
                        if not 0 <= v <= 255:
                            continue
                        old = x1[yy, xx, oc]
                        x1[yy, xx, oc] = v
                        n = int((resize_u8(x1).astype(np.int64) != target).sum())
                        x1[yy, xx, oc] = old
                        if best is None or n < best[0]:
                            best = (n, yy, xx, oc, v)
            if best and best[0] < cur:
                _, yy, xx, oc, v = best
                x1[yy, xx, oc] = v
                cur = best[0]

    return x1, {"start_mismatch": start, "end_mismatch": cur, "exact": cur == 0, "passes": p + 1}
