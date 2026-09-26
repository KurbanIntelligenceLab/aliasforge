#!/usr/bin/env python3
"""AliasForge: construct counterfactual image pairs that collide at the interface.

A pair is two full-resolution images carrying DIFFERENT legible content -- so a
shared question and candidate step get opposite correctness labels -- whose
preprocessed interface state is bit-identical.  Theorem 1 then pins every
language-side repair at exactly zero gain on that pair.

Stratum S1 (resize / quantization) is implemented here.  The mechanism is the
null space of the resize operator, used as an instrument rather than an attack:

    A mainstream downscale is linear before quantization, D(X) = A X B^T.  If
    every COLUMN of a perturbation delta lies in null(A), then A delta = 0 and
    D(base + delta) = D(base), whatever delta looks like at full resolution.

The legibility requirement is what makes the construction non-obvious: an
arbitrary null-space vector is high-frequency noise, not a readable digit.  We
get both by modulating a null-space CARRIER with a mask that is piecewise
constant on the downsampling cells.  For an integer-factor box downscale with
factor s, the carrier alternates sign within each cell, so each cell sums to
zero exactly -- in integer arithmetic, with no rounding argument needed -- while
at full resolution the masked region reads as a striped shape against flat
background.

Nothing here is trusted on the strength of that argument.  Every candidate is
verified by pushing BOTH members through the real processor and comparing
digests (Assumption A3); candidates that fail are rejected and counted.  The
rejection rate is a measured quantity, reported by E0a as yield.
"""

from __future__ import annotations

import numpy as np

from .interface import capture, per_field_report, state_digest

# ---------------------------------------------------------------------------
# null-space machinery
# ---------------------------------------------------------------------------


def probe_row_operator(resize_fn, H: int, W: int) -> np.ndarray:
    """Recover the row-resampling operator A (up to a global scale) by impulses.

    Feeding a profile that is constant along columns, X = u (x) 1_W, gives
    D(X)[y, x] = (A u)_y * (B 1)_x, so column 0 of the output recovers A u up to
    the fixed factor (B 1)_0.  A global scale does not move the null space, which
    is all we use A for.

    Cost note: this is H calls, so passing the FULL width makes it O(H^2 W) and
    it dominates everything else at large H.  The row operator depends only on
    (H, h) and the kernel, not on the width, so callers should probe through a
    narrow image -- see `probe_row_operator_narrow`, which is the same operator
    for a fraction of the work.
    """
    cols = []
    for i in range(H):
        X = np.zeros((H, W), dtype=np.float64)
        X[i, :] = 1.0
        cols.append(np.asarray(resize_fn(X), dtype=np.float64)[:, 0])
    return np.stack(cols, axis=1)  # (h, H)


def probe_row_operator_narrow(make_resize, H: int, h: int, width: int = 4) -> np.ndarray:
    """The same row operator, probed through a `width`-column image.

    A separable resize applies the row operator independently of the column one,
    so a narrow probe recovers A exactly while doing O(H * h * width) work instead
    of O(H^2 * W).  `make_resize(w_in, w_out)` must return a float resize taking
    (H, w_in) to (h, w_out); the column resize is made an identity by asking for
    the same width in and out, which keeps the recovered scale clean.
    """
    resize_fn = make_resize(width, width)
    cols = []
    for i in range(H):
        X = np.zeros((H, width), dtype=np.float64)
        X[i, :] = 1.0
        cols.append(np.asarray(resize_fn(X), dtype=np.float64)[:, 0])
    return np.stack(cols, axis=1)


def nullspace(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for null(A), as columns."""
    _u, s, vh = np.linalg.svd(A)
    tol = (s.max() if s.size else 0.0) * rtol
    rank = int((s > tol).sum())
    return vh[rank:].T.conj()


def is_linear(resize_fn, H: int, W: int, rng: np.random.Generator, trials: int = 4) -> float:
    """Max relative departure from linearity, measured rather than assumed.

    The whole S1 mechanism rests on D being linear before quantization.  If this
    returns something far above float noise, the operator is not linear (an
    antialias policy that switches on size, say) and S1 must not be claimed for
    it.
    """
    worst = 0.0
    for _ in range(trials):
        X = rng.normal(size=(H, W))
        Y = rng.normal(size=(H, W))
        a, b = rng.normal(), rng.normal()
        lhs = np.asarray(resize_fn(a * X + b * Y), dtype=np.float64)
        rhs = a * np.asarray(resize_fn(X), dtype=np.float64) + b * np.asarray(
            resize_fn(Y), dtype=np.float64
        )
        scale = max(np.abs(rhs).max(), 1e-12)
        worst = max(worst, float(np.abs(lhs - rhs).max() / scale))
    return worst


# ---------------------------------------------------------------------------
# content rendering
# ---------------------------------------------------------------------------

# 5x3 bitmap font: enough for a digit, which is the atomic fact we need.  Kept
# inline so the constructor depends on no font file or renderer.
_GLYPHS = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
}


def glyph_mask(char: str, cell_h: int, cell_w: int, scale: int = 1, top: int = 0, left: int = 0) -> np.ndarray:
    """Render a digit on the DOWNSAMPLED cell grid, as a 0/1 mask.

    Rendering on the cell grid (not the full-resolution grid) is what makes the
    mask piecewise constant on cells, which is exactly the condition under which
    the modulated carrier stays in the null space.
    """
    rows = _GLYPHS[char]
    m = np.zeros((cell_h, cell_w), dtype=np.int64)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch != "1":
                continue
            y0, x0 = top + r * scale, left + c * scale
            m[y0 : y0 + scale, x0 : x0 + scale] = 1
    return m


def upsample_nearest(mask: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """Cell grid -> full resolution, by exact replication."""
    return np.repeat(np.repeat(mask, sy, axis=0), sx, axis=1)


# ---------------------------------------------------------------------------
# S1: the exact integer construction for a box / area downscale
# ---------------------------------------------------------------------------


def box_carrier(s: int, length: int) -> np.ndarray:
    """A vertical carrier that sums to zero within every cell of size s.

    For even s this alternates +1/-1; for odd s one row is left at zero so the
    cell still sums to zero exactly.  Integer valued, so the cancellation is
    exact in integer arithmetic -- there is no rounding to argue about.
    """
    cell = np.zeros(s, dtype=np.int64)
    half = s // 2
    cell[:half] = 1
    cell[half : 2 * half] = -1
    return np.tile(cell, length // s)[:length]


def make_s1_pair(
    base: np.ndarray,
    char_a: str,
    char_b: str,
    cell: int,
    amplitude: int = 40,
    scale: int = 2,
    top: int = 1,
    left: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build an S1 candidate pair on a uint8 RGB base image.

    Returns (x0, x1, meta).  x0 renders `char_a`, x1 renders `char_b`, and both
    have identical cell sums in every channel -- so an exact box downscale by
    `cell` maps them to the same tensor.

    The amplitude is clipped against the base so no pixel saturates: saturation
    is a clip, clipping is nonlinear, and a nonlinearity would break the very
    cancellation the construction relies on.  Candidates that cannot be built
    without saturation are rejected here rather than failing a digest later.
    """
    H, W, C = base.shape
    ch, cw = H // cell, W // cell
    carrier = box_carrier(cell, H)[:, None, None]  # (H,1,1)

    out = []
    for char in (char_a, char_b):
        m_cell = glyph_mask(char, ch, cw, scale=scale, top=top, left=left)
        mask = upsample_nearest(m_cell, cell, W // cw)[:H, :W, None]  # (H,W,1)
        delta = amplitude * mask * carrier  # piecewise constant on cells x carrier
        out.append(base.astype(np.int64) + delta)

    x0, x1 = out
    lo = min(int(x0.min()), int(x1.min()))
    hi = max(int(x0.max()), int(x1.max()))
    saturates = lo < 0 or hi > 255

    meta = {
        "stratum": "S1",
        "cell": cell,
        "amplitude": amplitude,
        "char_a": char_a,
        "char_b": char_b,
        "saturates": bool(saturates),
        "pix_diff": int((x0 != x1).sum()),
        "pix_total": int(x0.size),
    }
    if saturates:
        return None, None, meta
    return x0.astype(np.uint8), x1.astype(np.uint8), meta


def cell_sums_match(x0: np.ndarray, x1: np.ndarray, cell: int) -> bool:
    """Exact integer check that both members share every cell sum.

    This is the mathematical precondition; it is NOT the certificate.  The
    certificate is the digest comparison against the real processor, because a
    real pipeline may reorder float accumulation even when the exact sums agree.
    """
    H, W, C = x0.shape
    ch, cw = H // cell, W // cell
    a = x0[: ch * cell, : cw * cell].astype(np.int64).reshape(ch, cell, cw, cell, C).sum((1, 3))
    b = x1[: ch * cell, : cw * cell].astype(np.int64).reshape(ch, cell, cw, cell, C).sum((1, 3))
    return bool(np.array_equal(a, b))


# ---------------------------------------------------------------------------
# certification
# ---------------------------------------------------------------------------


def certify(processor, x0: np.ndarray, x1: np.ndarray, text: str = "Describe the image.") -> dict:
    """Push both members through the real processor and compare digests.

    This is Assumption A3, executed.  A pair is certified only if EVERY emitted
    field is bit-identical; a single differing field rejects it, and the report
    names which one so E0b can tell a genuine near-miss from an enumeration gap.
    """
    from PIL import Image

    sa = capture(processor, Image.fromarray(x0), text)
    sb = capture(processor, Image.fromarray(x1), text)
    rep = per_field_report(sa, sb)
    return {
        "certified": rep["collides"],
        "digest_a": state_digest(sa),
        "digest_b": state_digest(sb),
        "fields_differ": ",".join(rep["differ"]),
        "fields_missing": ",".join(rep["missing"]),
        "n_fields": len(sa),
    }
