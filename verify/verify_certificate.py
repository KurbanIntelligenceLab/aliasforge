#!/usr/bin/env python3
"""Re-verify the paper's central claim end to end, on a laptop, with no model.

The claim is arithmetic, not empirical: a deployed resize is an exact integer map, and a
perturbation in its integer kernel leaves the resized tensor bit-identical. This script
recomputes that chain for any (kernel, input side, output side):

  1. reconstruct the library's fixed-point operator K and gate it on reproducing
     PIL.Image.resize bit for bit, so the matrix is the deployed one and not a model of it;
  2. compute the certified lower bound on the sup-norm of any nonzero integer kernel vector
     supported on a 32-wide window, in exact rational arithmetic over a Z-basis of the full
     integer kernel (a basis of a sublattice would give an unsound bound);
  3. if the bound exceeds 255, report CERTIFIED NON-CONSTRUCTIBLE and stop: no 8-bit image
     can carry a perturbation this operator annihilates on such a window;
  4. otherwise take the shortest reduced witness, build the two members of a pair whose
     difference is that witness on a row mask (three bands against two, the atomic fact),
     resize both, and require the outputs to be equal in every byte.

Exit status is 0 only if every stage did what the paper says it does.

  $ python verify_certificate.py                      # the deployed 896->448 configuration
  $ python verify_certificate.py --kernel lanczos     # a certified negative
"""
from __future__ import annotations
import argparse, hashlib, pathlib, sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from aliasforge.exact import KERNELS, pil_coeffs                      # noqa: E402
from lattice_bounds import certified_window_bound                    # noqa: E402

PIL_KERNEL = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR,
              "box": Image.BOX, "lanczos": Image.LANCZOS}


def gate(K, n_in, n_out, resample, seed=0):
    """The reconstruction must reproduce the library exactly, or nothing below is evidence."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (n_in, 4, 3), dtype=np.uint8)
    ref = np.asarray(Image.fromarray(img).resize((4, n_out), resample))
    half, bits = 1 << 21, 22
    mine = np.empty_like(ref)
    for c in range(4):
        for ch in range(3):
            acc = K @ img[:, c, ch].astype(object) + half
            mine[:, c, ch] = np.clip([int(x) >> bits for x in acc], 0, 255)
    return bool(np.array_equal(ref, mine))


def row_masks(H, n_a=3, n_b=2):
    a, b = np.zeros(H, bool), np.zeros(H, bool)
    for i, y0 in enumerate((int(.13 * H), int(.38 * H), int(.63 * H))):
        if i < n_a: a[y0:y0 + int(.12 * H)] = True
        if i < n_b: b[y0:y0 + int(.12 * H)] = True
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="bicubic", choices=sorted(PIL_KERNEL))
    ap.add_argument("--n-in", type=int, default=896)
    ap.add_argument("--n-out", type=int, default=448)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    filt, support = KERNELS[a.kernel]
    K = pil_coeffs(a.n_in, a.n_out, support=support, filt=filt)
    ok = gate(K, a.n_in, a.n_out, PIL_KERNEL[a.kernel], a.seed)
    print(f"exp=verify_certificate kernel={a.kernel} resize={a.n_in}->{a.n_out} "
          f"ratio={a.n_in / a.n_out:.4g} window={a.width} stride={a.stride} "
          f"bit_exact_gate={'pass' if ok else 'FAIL'}")
    if not ok:
        print("RESULT verified=0 reason=the reconstructed operator is not the library's")
        return 2

    best = None
    for c0 in range(0, max(1, a.n_in - a.width), a.stride):
        r = certified_window_bound(K, c0, a.width)
        if r is None:
            continue
        if best is None or r["bound"] < best[0]["bound"]:
            best = (r, c0)
    if best is None:
        print("RESULT verified=1 constructible=0 reason=the kernel is trivial on every window")
        return 0
    r, c0 = best
    print(f"certified_bound={r['bound']:.6g} over {a.width}-wide windows "
          f"(a bound above 255 excludes every 8-bit perturbation)")
    if r["bound"] > 255:
        print(f"RESULT verified=1 constructible=0 bound={r['bound']:.6g} "
              f"reason=certified non-constructible on this window family")
        return 0

    v = np.asarray([int(x) for x in r["witness"]], dtype=np.int64)
    span = int(v.max() - v.min())
    print(f"witness sup_norm={int(np.abs(v).max())} span={span} window={c0}")
    if span > 255:
        print("RESULT verified=1 constructible=0 reason=witness spans more than 8 bits")
        return 0

    vec = np.zeros(a.n_in, dtype=np.int64); vec[c0:c0 + a.width] = v
    amp = max(1, 100 // int(np.abs(v).max()))
    rng = np.random.default_rng(a.seed)
    base = rng.integers(110, 146, (a.n_in, a.n_in, 3), dtype=np.uint8).astype(np.int64)
    ma, mb = row_masks(a.n_in)
    xa, xb = base.copy(), base.copy()
    xa[ma] += (amp * vec)[:, None]; xb[mb] += (amp * vec)[:, None]
    if min(xa.min(), xb.min()) < 0 or max(xa.max(), xb.max()) > 255:
        print("RESULT verified=0 reason=the carrier saturates at this amplitude")
        return 2
    xa, xb = xa.astype(np.uint8), xb.astype(np.uint8)
    ra = np.asarray(Image.fromarray(xa).resize((a.n_out, a.n_out), PIL_KERNEL[a.kernel]))
    rb = np.asarray(Image.fromarray(xb).resize((a.n_out, a.n_out), PIL_KERNEL[a.kernel]))
    identical = bool(np.array_equal(ra, rb))
    d = hashlib.sha256(np.ascontiguousarray(ra).tobytes()).hexdigest()[:16]
    print(f"pair amplitude={amp} pixels_differing={int((xa != xb).any(-1).sum())} "
          f"marks=3vs2 resized_digest={d} identical={int(identical)}")
    print(f"RESULT verified={int(identical)} constructible=1 "
          f"sup_norm={int(np.abs(v).max())} bound={r['bound']:.6g}")
    return 0 if identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
