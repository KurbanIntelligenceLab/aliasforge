#!/usr/bin/env python3
"""Find the regions a verifier's interface throws away, by measuring rather than reading code.

Strata S2 (crop / pad), S3 (tiling / aspect bucketing) and S4 (pre-encoder
masking) all say the same thing in different words: some source pixels never
reach the language model, so label-flipping content can be written there for
free.  Which pixels those are depends on the implementation -- a center crop, a
letterbox band, an aspect bucket that rounds the extent down, a dropped patch.

Reimplementing each target's preprocessing to find them invites exactly the error
Appendix B warns about: a reimplementation that is subtly unfaithful produces
pairs that look certified and are not.  So we do not reimplement anything.  We
perturb a block of the real image, push it through the REAL processor, and ask
whether the digest moved.  A block whose digest never moves is provably ignored
BY THAT PROCESSOR -- no code reading, no architectural assumption, and it works
the same way for every target and every stratum.

One caution, respected in the API below.  Zero influence is measured one block at
a time, and blocks that are individually inert can in principle interact: a
pipeline could ignore either of two regions alone but not both together.  The map
is therefore a CANDIDATE GENERATOR, never a certificate.  The certificate is
always the digest of the finished pair, checked in `certify`.
"""

from __future__ import annotations

import numpy as np

from .interface import capture, state_digest


def influence_map(
    processor,
    image: np.ndarray,
    block: int = 16,
    text: str = "Describe the image.",
    mode: str = "invert",
) -> tuple[np.ndarray, dict]:
    """Per-block indicator: does perturbing this block change the interface digest?

    Parameters
    ----------
    image : uint8 (H, W, 3)
    block : int
        Probe granularity.  Smaller finds more usable area but costs
        (H/block)*(W/block) processor calls.
    mode : "invert" | "noise"
        How hard to hit each block.  Inversion is the largest possible change to
        a uint8 block, which minimises the chance of calling a block inert merely
        because the perturbation was too small to survive quantization.

    Returns
    -------
    (infl, stats) where infl[by, bx] is 1 if the block moved the digest.
    """
    from PIL import Image

    H, W = image.shape[:2]
    nby, nbx = (H + block - 1) // block, (W + block - 1) // block
    base = state_digest(capture(processor, Image.fromarray(image), text))

    infl = np.zeros((nby, nbx), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for by in range(nby):
        for bx in range(nbx):
            y0, y1 = by * block, min(H, (by + 1) * block)
            x0, x1 = bx * block, min(W, (bx + 1) * block)
            x = image.copy()
            if mode == "invert":
                x[y0:y1, x0:x1] = 255 - x[y0:y1, x0:x1]
            else:
                x[y0:y1, x0:x1] = rng.integers(0, 256, x[y0:y1, x0:x1].shape, dtype=np.uint8)
            infl[by, bx] = int(state_digest(capture(processor, Image.fromarray(x), text)) != base)

    n = infl.size
    dead = int((infl == 0).sum())
    return infl, {
        "blocks": n,
        "dead_blocks": dead,
        "dead_fraction": dead / n if n else 0.0,
        "block": block,
        "H": H,
        "W": W,
        "base_digest": base[:16],
    }


def dead_pixel_mask(infl: np.ndarray, block: int, H: int, W: int) -> np.ndarray:
    """Expand a block-level influence map to a full-resolution bool mask of ignored pixels."""
    mask = np.repeat(np.repeat(infl == 0, block, axis=0), block, axis=1)
    return mask[:H, :W]


def largest_dead_rectangle(dead: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned all-True rectangle in a boolean mask.

    Standard maximal-rectangle-in-histogram sweep: O(H*W), exact.  Returns
    (top, left, height, width).
    """
    H, W = dead.shape
    heights = np.zeros(W + 1, dtype=np.int64)  # sentinel column keeps the stack honest
    best = (0, 0, 0, 0)
    best_area = 0
    for y in range(H):
        heights[:W] = np.where(dead[y], heights[:W] + 1, 0)
        stack: list[int] = []
        for x in range(W + 1):
            while stack and heights[stack[-1]] >= heights[x]:
                h = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                w = x - left
                if h * w > best_area:
                    best_area = h * w
                    best = (y - h + 1, left, h, w)
            stack.append(x)
    return best


def make_dead_region_pair(
    image: np.ndarray,
    dead: np.ndarray,
    char_a: str,
    char_b: str,
    amplitude: int = 110,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[None, None, dict]:
    """Write two different legible glyphs into the ignored region.

    This is the S2/S3/S4 constructor.  Unlike S1 it needs no null space, no
    kernel model and no repair: the region is discarded outright, so the content
    can be arbitrary and maximally legible.  The glyph is rendered at whatever
    scale the dead region admits, and both members share every non-dead pixel
    exactly, so any digest difference would have to come from the dead region --
    which is what `certify` then checks.
    """
    from .aliasforge import glyph_mask

    if not dead.any():
        return None, None, {"stratum": "S2/S3/S4", "reason": "no dead region"}

    # The bounding box of all dead pixels is the wrong frame: for a center crop
    # the dead set is two disjoint side bands, whose bounding box spans the whole
    # image and is mostly LIVE.  A glyph drawn there gets masked away exactly
    # where the two glyphs differ.  Use the largest all-dead rectangle instead,
    # so the whole glyph -- and therefore the whole difference -- is inside dead
    # territory by construction.
    y0, x0, rh, rw = largest_dead_rectangle(dead)
    scale = min(rh // 6, rw // 4)
    if scale < 1:
        return None, None, {
            "stratum": "S2/S3/S4",
            "reason": f"largest all-dead rectangle too small ({rh}x{rw})",
        }
    y1, x1 = y0 + rh, x0 + rw

    out, painted = [], []
    for ch in (char_a, char_b):
        g = glyph_mask(ch, rh, rw, scale=scale, top=0, left=0)
        x = image.copy()
        # paint ONLY where the region is genuinely dead, so a ragged dead region
        # never leaks a single live pixel into the difference
        sub = x[y0:y1, x0:x1]
        d = dead[y0:y1, x0:x1]
        paint = (g > 0) & d
        sub[paint] = amplitude
        x[y0:y1, x0:x1] = sub
        out.append(x)
        painted.append(int(paint.sum()))

    a, b = out
    pix_diff = int((a != b).any(-1).sum())
    meta = {
        "stratum": "S2/S3/S4",
        "char_a": char_a,
        "char_b": char_b,
        "region": (y0, x0, rh, rw),
        "scale": scale,
        "pix_diff": pix_diff,
        "pix_total": int(a.shape[0] * a.shape[1]),
        "painted_a": painted[0],
        "painted_b": painted[1],
    }

    # A pair whose members are IDENTICAL certifies nothing: equal digests are
    # then a tautology, not evidence, and the pair carries no opposite labels.
    # This happens when the dead region misses the strokes on which the two
    # glyphs differ -- the dead region can be large while that intersection is
    # empty -- so it must be caught here rather than inferred from a digest.
    if pix_diff == 0:
        meta["reason"] = (
            f"degenerate: members identical (painted {painted[0]}/{painted[1]} px, "
            f"glyph difference does not intersect the dead region)"
        )
        return None, None, meta

    return a, b, meta
