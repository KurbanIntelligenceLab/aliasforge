#!/usr/bin/env python3
"""Qwen2.5-VL preprocessing: smart_resize, and the regime in which it has a null space.

Transcribed from `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`.  The
detail that matters is the pair of STRICT inequalities: rescaling happens only
when the rounded area is `> max_pixels` or `< min_pixels`, so an input whose
dimensions are already multiples of `factor` and whose area lies in the CLOSED
interval passes through untouched.  Both resize backends then short-circuit on
equal size -- Pillow's `Image.resize` returns `self.copy()`, torchvision's
`resize_image` returns the input tensor -- so the resize is bit-exact identity and
its null space is exactly {0}.

That is not a limitation of the method; it is a fact about the deployment.  It
explains the measured 0 dead blocks on Qwen at every probe geometry: the shipped
`preprocessor_config.json` sets `max_pixels = 12845056` (12.8 MP), so ordinary
images are never resized at all and there is nothing to alias.

A null space appears exactly when the deployment downscales, which is what a
resolution-adaptive policy does by construction -- the closest comparator in the
manuscript starts from a downsampled image and requests higher resolution only on
demand.  `max_pixels` is therefore a property of the routing configuration under
test, and the manuscript already records that the certificate is interface-
specific and that changing the resolution policy invalidates an alias set.  We
measure at a router-realistic budget and say which one.
"""

from __future__ import annotations

import math

import numpy as np

IMAGE_FACTOR = 28          # patch_size (14) * merge_size (2)
MIN_PIXELS = 4 * 28 * 28   # 3136
MAX_PIXELS_STOCK = 12845056        # Qwen2.5-VL-7B-Instruct shipped config
MAX_PIXELS_TRANSFORMERS = 28 * 28 * 1280   # 1003520, the library default
MAX_RATIO = 200


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS_STOCK):
    """Verbatim structure from the transformers implementation."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def is_identity(height: int, width: int, **kw) -> bool:
    """True when the resize stage is bit-exact identity, i.e. null space = {0}."""
    return smart_resize(height, width, **kw) == (height, width)


def null_space_dim(height: int, width: int, **kw) -> tuple[int, int]:
    """(vertical, horizontal) null-space dimensions of the resize, per channel."""
    h_bar, w_bar = smart_resize(height, width, **kw)
    return max(0, height - h_bar), max(0, width - w_bar)


def router_budget(fraction: float = 1.0 / 4) -> int:
    """A max_pixels a resolution-adaptive policy would plausibly run at.

    Expressed as a fraction of the library default rather than an arbitrary
    constant, and snapped to a whole number of 28x28 patches so smart_resize's
    own rounding is not doing something surprising underneath the measurement.
    """
    patches = max(4, int((MAX_PIXELS_TRANSFORMERS * fraction) // (IMAGE_FACTOR ** 2)))
    return patches * IMAGE_FACTOR ** 2


def describe(height: int, width: int, max_pixels: int) -> dict:
    h_bar, w_bar = smart_resize(height, width, max_pixels=max_pixels)
    vy, vx = null_space_dim(height, width, max_pixels=max_pixels)
    return {
        "in": (height, width),
        "out": (h_bar, w_bar),
        "identity": (h_bar, w_bar) == (height, width),
        "null_dim_y": vy,
        "null_dim_x": vx,
        "downscale": round(height / h_bar, 4) if h_bar else float("nan"),
        "max_pixels": max_pixels,
    }


if __name__ == "__main__":
    print(f"stock max_pixels={MAX_PIXELS_STOCK} ({MAX_PIXELS_STOCK/1e6:.1f} MP)")
    print(f"transformers default={MAX_PIXELS_TRANSFORMERS} "
          f"({MAX_PIXELS_TRANSFORMERS/1e6:.2f} MP)")
    print(f"router budget (1/4 of default)={router_budget()}\n")
    print(f"{'geometry':>12s} {'max_pixels':>11s} {'-> out':>12s} {'identity':>9s} "
          f"{'null_y':>7s} {'null_x':>7s}")
    for (h, w) in ((448, 448), (560, 560), (1008, 756), (1024, 768), (4032, 3024)):
        for mp in (MAX_PIXELS_STOCK, MAX_PIXELS_TRANSFORMERS, router_budget()):
            d = describe(h, w, mp)
            print(f"{str(h)+'x'+str(w):>12s} {mp:>11d} {str(d['out']):>12s} "
                  f"{str(d['identity']):>9s} {d['null_dim_y']:>7d} {d['null_dim_x']:>7d}")
