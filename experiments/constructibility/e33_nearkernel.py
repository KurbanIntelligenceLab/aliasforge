#!/usr/bin/env python3
"""E33: is exact kernel membership what makes a collision portable across images?

A second review observed that integer perturbations OUTSIDE the exact kernel can still
resize bit-identically, because the fixed-point operator emits clip8((acc + 2^21) >> 22)
and a small accumulator difference is absorbed by that quantization.  That is true, and
on a FLAT base it is common.  This experiment asks what such collisions depend on.

On a flat base every output accumulator sits at the same position inside its 2^22-wide
rounding bucket, so one perturbation collides at every pixel or at none.  On a natural
image each affected accumulator sits at its own position, and the perturbation must fit
inside the margin at EVERY affected output.  An exact-kernel difference has zero
accumulator difference and so collides on every base at every offset, which is the
property the construction needs to place a legible fact on an arbitrary image.

Per (ratio, base kind) we draw random sparse integer perturbations of sup-norm <= 3 on
one row, keep those outside the exact kernel, and count bit-identical resizes.  We also
apply the exact-kernel carrier at random valid offsets and amplitudes and count the same.
Clipping breaks exactness for any mechanism, so trials whose perturbed pixel leaves
[0, 255] are counted separately and excluded from the rate.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import pil_coeffs                    # noqa: E402
from aliasforge.items import ImageStore, load_items         # noqa: E402

VERSION = "e33.2"
RATIOS = ((672, 448, "1.5x"), (896, 448, "2.0x"))
COLUMNS = ("ratio", "base", "mechanism", "n_bases", "trials", "clipped", "collide", "rate")

def resize(a, n_out):
    return np.asarray(Image.fromarray(a).resize((n_out, n_out), Image.BICUBIC))

def kernel_vectors(K, W=8):
    """Short exact-kernel vectors with their valid column offsets, found by brute force
    over sup-norm <= 3 supports of width 4; enough to exhibit the mechanism."""
    out = []
    n_in = K.shape[1]
    rng = np.random.default_rng(1)
    for _ in range(4000):
        w = rng.integers(-3, 4, size=4)
        if not w.any(): continue
        offs = [c for c in range(0, n_in - 4) if not (K[:, c:c + 4] @ w).any()]
        if offs: out.append((w, offs))
        if len(out) >= 3: break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=40); ap.add_argument("--trials", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    print(f"exp=e33_nearkernel version={VERSION} n_images={a.n_images} trials={a.trials}", flush=True)
    items = load_items(a.data, limit=None, single_image_only=True)
    store = ImageStore(a.data)
    naturals = []
    for it in items.values() if isinstance(items, dict) else items:
        try: img = np.asarray(store.load(it.image_path).convert("RGB"))
        except Exception: continue
        naturals.append(img)
        if len(naturals) >= a.n_images: break
    print(f"loaded natural images n={len(naturals)}", flush=True)
    rng = np.random.default_rng(0)
    rows = []
    for n_in, n_out, lab in RATIOS:
        K = pil_coeffs(n_in, n_out)
        kv = kernel_vectors(K)
        bases = {
            "flat": [np.full((n_in, n_in, 3), 128, np.uint8)],
            "noise": [rng.integers(0, 256, size=(n_in, n_in, 3)).astype(np.uint8) for _ in range(8)],
            "natural": [np.asarray(Image.fromarray(im).resize((n_in, n_in), Image.BICUBIC)) for im in naturals],
        }
        for bname, blist in bases.items():
            ref = [resize(b, n_out) for b in blist]
            # out-of-kernel perturbations
            trials = clipped = collide = 0
            while trials < a.trials and trials + clipped < 40 * a.trials:
                w = rng.integers(-3, 4, size=4)
                if not w.any(): continue
                c0 = int(rng.integers(0, n_in - 4))
                if not (K[:, c0:c0 + 4] @ w).any(): continue          # exact kernel: not this arm
                bi = int(rng.integers(0, len(blist))); b = blist[bi]
                row = int(rng.integers(8, n_in - 8))
                d = b.astype(int); d[row, c0:c0 + 4, :] += w[:, None]
                if d.min() < 0 or d.max() > 255: clipped += 1; continue
                trials += 1
                collide += int(np.array_equal(resize(d.astype(np.uint8), n_out), ref[bi]))
            rows.append((lab, bname, "outside exact kernel", len(blist), trials, clipped, collide, round(collide / max(trials, 1), 4)))
            print(f"RESULT ratio={lab} base={bname} mech=outside trials={trials} clipped={clipped} collide={collide} rate={collide/max(trials,1):.4f}", flush=True)
            # exact-kernel carrier at random valid offsets and amplitudes
            trials = clipped = collide = 0
            while kv and trials < a.trials and trials + clipped < 40 * a.trials:
                w, offs = kv[int(rng.integers(0, len(kv)))]
                c0 = int(offs[int(rng.integers(0, len(offs)))]); m = int(rng.choice([1, 5, 20]))
                bi = int(rng.integers(0, len(blist))); b = blist[bi]
                row = int(rng.integers(8, n_in - 8))
                d = b.astype(int); d[row, c0:c0 + 4, :] += m * w[:, None]
                if d.min() < 0 or d.max() > 255: clipped += 1; continue
                trials += 1
                collide += int(np.array_equal(resize(d.astype(np.uint8), n_out), ref[bi]))
            rows.append((lab, bname, "exact kernel", len(blist), trials, clipped, collide, round(collide / max(trials, 1), 4)))
            print(f"RESULT ratio={lab} base={bname} mech=exact trials={trials} clipped={clipped} collide={collide} rate={collide/max(trials,1):.4f}", flush=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(COLUMNS); wr.writerows(rows)
    print(f"RESULT rows={len(rows)} elapsed_s={time.time()-t0:.0f}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
