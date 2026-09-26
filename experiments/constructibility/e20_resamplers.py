#!/usr/bin/env python3
"""E20: is exact interface aliasing specific to bicubic, or a property of fixed-point resampling?

Every certified pair in this paper was built against one resampler: Pillow's bicubic.  The
obvious question is whether the mechanism is a quirk of
that kernel.  If it is, the contribution is about one code path; if it is not, it is about a
class of deployed implementations.

This adds no new mechanism and no new claim.  It applies the SAME instrument -- reconstruct
the fixed-point operator exactly, take the Gram-Schmidt lower bound on the shortest lattice
vector, search for a short integer kernel vector, then build a pair and verify the resized
outputs are bit-identical -- across Pillow's four resampling kernels and a range of ratios.

Every reconstructed operator is GATED on reproducing `PIL.Image.resize` bit-for-bit before
any conclusion is drawn from it, so a negative row is a property of the kernel and not of a
mis-transcribed filter.  That gate matters: an operator recovered by probing rather than by
transcription quantises the weights and reports false negatives.

Three things this can settle, all of which serve the existing narrative rather than opening
a new one:
  - generality: whether susceptibility survives a change of kernel;
  - the criterion: whether "dyadic" is a universal rule or bicubic's particular answer to a
    bound that is really computed from K;
  - defence: whether some deployed kernel is resistant, which the ethics section can name.

CPU only; no model weights, no dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import (KERNELS, PRECISION_BITS, integer_kernel_vector,  # noqa: E402
                             pil_coeffs)

VERSION = "e20.3"


def pil_resample(name):
    from PIL import Image
    return {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR,
            "box": Image.BOX, "lanczos": Image.LANCZOS}[name]


def reconstruct(name, n_in, n_out):
    f, supp = KERNELS[name]
    return pil_coeffs(n_in, n_out, support=supp, filt=f)


def bit_exact(name, n_in, n_out, K, rng):
    """Gate: the reconstructed operator must reproduce Pillow exactly."""
    from PIL import Image
    img = rng.integers(0, 256, (n_in, 4, 3), dtype=np.uint8)
    ref = np.asarray(Image.fromarray(img).resize((4, n_out), pil_resample(name)))
    mine = np.empty_like(ref)
    half = 1 << (PRECISION_BITS - 1)
    for c in range(4):
        for ch in range(3):
            ss = K @ img[:, c, ch].astype(object) + half
            mine[:, c, ch] = np.clip([int(x) >> PRECISION_BITS for x in ss], 0, 255)
    return bool(np.array_equal(ref, mine))


def certify_pair(name, n_in, n_out, v, rng, amps=(1, 2, 5, 10, 20, 40, 60)):
    """Build a real pair and check the resized outputs collide bit-for-bit."""
    from PIL import Image
    dg = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    for amp in amps:
        w = amp * v
        lo, hi = int(w.min()), int(w.max())
        if hi - lo > 255:                       # no pedestal fits this carrier in eight bits
            continue
        ped = (-lo + (255 - hi)) // 2           # midpoint of the feasible pedestal interval
        base = np.full((n_in, 8, 3), ped, dtype=np.int64)
        x1 = base + w[:, None, None]
        if x1.min() < 0 or x1.max() > 255:
            continue
        a = np.asarray(Image.fromarray(base.astype(np.uint8)).resize((8, n_out), pil_resample(name)))
        b = np.asarray(Image.fromarray(x1.astype(np.uint8)).resize((8, n_out), pil_resample(name)))
        if dg(a) == dg(b):
            return 1, amp, int((base != x1).any(axis=2).sum())
    return 0, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--kernels", default="bicubic,bilinear,box,lanczos")
    ap.add_argument("--sizes", default="896:448,896:224,672:224,1344:448,896:112,1344:224,672:448,896:298")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--max-abs", type=int, default=255,
                    help="realizability threshold on ||v||_inf; the paper's definition is 255")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    kernels = a.kernels.split(",")
    pairs = [tuple(int(x) for x in s.split(":")) for s in a.sizes.split(",")]
    print(f"exp=e20_resamplers version={VERSION} task={a.task}/{a.shards} "
          f"kernels={kernels} n_ratios={len(pairs)}")

    rows = [("kernel", "n_in", "n_out", "ratio", "dyadic", "bit_exact",
             "shortest_maxabs", "certified", "amplitude", "pix_diff")]
    for i, (n_in, n_out) in enumerate(pairs):
        if i % a.shards != a.task:
            continue
        ratio = n_in / n_out
        dy = abs(np.log2(ratio) - round(np.log2(ratio))) < 1e-9
        for name in kernels:
            try:
                K = reconstruct(name, n_in, n_out)
                ok = bit_exact(name, n_in, n_out, K, rng)
            except Exception as e:
                print(f"  {name:9} {n_in}->{n_out}: reconstruct failed {type(e).__name__}")
                continue
            short, cert, amp, pix = "", 0, 0, 0
            if ok:
                best = None
                for c0 in range(0, max(1, n_in - a.width), 16):
                    v = integer_kernel_vector(K, c0, a.width)
                    if v is not None and 0 < int(np.abs(v).max()) <= a.max_abs:
                        if best is None or int(np.abs(v).max()) < int(np.abs(best).max()):
                            best = v
                if best is not None:
                    short = int(np.abs(best).max())
                    cert, amp, pix = certify_pair(name, n_in, n_out, best, rng)
            rows.append((name, n_in, n_out, round(ratio, 4), int(dy), int(ok),
                         short, cert, amp, pix))
            print(f"  {name:9} {n_in:>5}->{n_out:<4} ratio={ratio:5.2f} dyadic={int(dy)} "
                  f"bit_exact={int(ok)} shortest={short or '-':>3} certified={cert} amp={amp}",
                  flush=True)

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    b = rows[1:]
    print(f"RESULT task={a.task} cases={len(b)} bit_exact={sum(r[5] for r in b)} "
          f"with_short_vector={sum(1 for r in b if r[6] != '')} certified={sum(r[7] for r in b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
