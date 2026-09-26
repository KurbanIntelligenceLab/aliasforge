#!/usr/bin/env python3
"""E16: does the certified construction survive on NATURAL images?

Every certified pair in this project so far was built on a synthetic base -- a mid-grey
sinusoid plus noise -- with the atomic fact rendered as horizontal bands.  The preprocessing
under attack is real, so "exact collisions exist in deployed pipelines" is defensible about
the PIPELINE, and the open question is whether it transfers to real photographs.
Nothing in the record answers that, and the honest answer decides how the contribution is
worded.

The obstacle is not the algebra, which is content-independent: `amplitude * v` is in the
kernel whatever the image underneath.  It is SATURATION.  `make_exact_pair` refuses a pair
that would clip, because clipping is nonlinear and destroys the exact cancellation.  A
synthetic base at 128 +/- 24 has room for a +60 carrier everywhere; a photograph with
highlights and shadows may have room almost nowhere.  So the question is empirical and it is
about headroom, not about the certificate.

Three arms, in increasing cleverness, so the result separates "cannot be done" from "cannot
be done naively":

  fixed   the published construction verbatim -- the same row bands, one amplitude.
  placed  choose WHERE the bands go by local headroom, since the mask is unconstrained and
          the fact is "how many bands", not "bands at these heights".
  signed  choose the SIGN of the carrier per band: -v is in the kernel exactly when v is,
          so a bright region can take a negative carrier and a dark one a positive.
  ranged  compress the photograph's dynamic range into [pad, 255-pad] before adding the
          carrier, which GUARANTEES headroom everywhere.  The cost is a mild contrast
          reduction and the result is still a photograph; local piloting suggests this is
          the only arm that survives a high-dynamic-range image, so the honest headline is
          likely 'natural images admit certified aliases after a bounded contrast
          reduction', and the size of that reduction is the number to report.

Certification here is the interface digest, which is preprocessing only, so this needs no
GPU and no model weights.  Amplitude is swept, because a lower amplitude buys headroom at
the cost of legibility and the trade-off is the actual finding.

Output: one row per (image, arm, amplitude) -> $OUT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import find_short_vector, pil_coeffs  # noqa: E402
from aliasforge.items import ImageStore, load_items  # noqa: E402

VERSION = "e16.1"


def band_rows(H, y0_frac, thickness_frac=0.12):
    m = np.zeros(H, bool)
    y0 = int(y0_frac * H)
    m[y0:y0 + int(thickness_frac * H)] = True
    return m


def headroom(base, rows, add):
    """Largest and smallest value the carrier would produce on these rows."""
    sel = base[rows].astype(np.int64)
    hi = int((sel + add).max()) if sel.size else 10**9
    lo = int((sel + add).min()) if sel.size else -10**9
    return lo, hi


def try_pair(base, vec, amp, mask_a, mask_b, signs_a, signs_b):
    """Build the pair with a per-band sign, refusing anything that would clip."""
    H, W, _ = base.shape
    xa = base.astype(np.int64).copy()
    xb = base.astype(np.int64).copy()
    for x, masks, signs in ((xa, mask_a, signs_a), (xb, mask_b, signs_b)):
        for m, s in zip(masks, signs):
            x[m] += (s * amp * vec)[:, None]
    lo = min(int(xa.min()), int(xb.min()))
    hi = max(int(xa.max()), int(xb.max()))
    if lo < 0 or hi > 255:
        return None, None, dict(saturated=1, lo=lo, hi=hi)
    return xa.astype(np.uint8), xb.astype(np.uint8), dict(saturated=0, lo=lo, hi=hi)


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--side", type=int, default=896)
    ap.add_argument("--target", type=int, default=448)
    ap.add_argument("--amps", default="20,10,5,2")
    a = ap.parse_args()

    amps = [int(x) for x in a.amps.split(",")]
    K = pil_coeffs(a.side, a.target)
    vec, meta = find_short_vector(K, width=64, stride=16, max_abs=8)
    if vec is None:
        print("RESULT no kernel vector"); return 1
    maxv = int(np.abs(vec).max())

    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0))
             if i % a.shards == a.task]
    store = ImageStore(a.data)
    print(f"exp=e16_natural version={VERSION} task={a.task}/{a.shards} items={len(items)} "
          f"side={a.side}->{a.target} max_abs_v={maxv} amps={amps}")

    from PIL import Image
    rows_out = [("key", "arm", "amplitude", "certified", "saturated", "pix_diff",
                 "peak_change", "img_lo", "img_hi", "contrast_kept")]
    ARMS = ("fixed", "placed", "signed", "ranged")
    tally = {(arm, amp): [0, 0] for arm in ARMS for amp in amps}
    t0 = time.time()
    for n, it in enumerate(items):
        try:
            img = Image.fromarray(np.asarray(store.load(it.image_path))).convert("RGB")
            img = img.resize((a.side, a.side), Image.BICUBIC)
            base = np.asarray(img)
        except Exception:
            continue
        H = base.shape[0]
        thirds = (0.13, 0.38, 0.63)
        for amp in amps:
            add = amp * vec
            # --- fixed: published masks, positive carrier -------------------
            ma = [band_rows(H, y) for y in thirds]
            mb = [band_rows(H, y) for y in thirds[:2]]
            variants = {"fixed": (ma, mb, [1] * 3, [1] * 2)}
            # --- placed: pick the three bands with the most headroom --------
            cands = [(y, band_rows(H, y)) for y in np.arange(0.02, 0.86, 0.04)]
            scored = []
            for y, m in cands:
                lo, hi = headroom(base, m, add[:, None])
                scored.append((min(255 - hi, lo), y, m))
            scored.sort(reverse=True, key=lambda t: t[0])
            top = [t for t in scored[:3]]
            if len(top) == 3:
                pa = [t[2] for t in top]; pb = [t[2] for t in top[:2]]
                variants["placed"] = (pa, pb, [1] * 3, [1] * 2)
            # --- signed: fixed masks, per-band sign chosen by local mean ----
            sa, sb = [], []
            for m in ma:
                sa.append(-1 if base[m].mean() > 127 else 1)
            sb = sa[:2]
            variants["signed"] = (ma, mb, sa, sb)
            # --- ranged: guarantee headroom by compressing the dynamic range --
            variants["ranged"] = (ma, mb, [1] * 3, [1] * 2)

            for arm, (A, B, SA, SB) in variants.items():
                src = base
                if arm == "ranged":
                    pad = amp * maxv
                    src = (base.astype(np.float64) / 255.0 * (255 - 2 * pad) + pad
                           ).astype(np.uint8)
                xa, xb, info = try_pair(src, vec, amp, A, B, SA, SB)
                cert = 0; pix = 0
                if xa is not None:
                    # the certificate: the preprocessed interface must be bit-identical
                    da = digest(np.asarray(Image.fromarray(xa).resize(
                        (a.target, a.target), Image.BICUBIC)))
                    db = digest(np.asarray(Image.fromarray(xb).resize(
                        (a.target, a.target), Image.BICUBIC)))
                    cert = int(da == db)
                    pix = int((xa != xb).any(axis=2).sum())
                kept = round((255 - 2 * amp * maxv) / 255.0, 4) if arm == "ranged" else 1.0
                rows_out.append((it.key, arm, amp, cert, info["saturated"], pix,
                                 amp * maxv, info["lo"], info["hi"], kept))
                tally[(arm, amp)][0] += cert
                tally[(arm, amp)][1] += 1
        if (n + 1) % 20 == 0:
            el = time.time() - t0
            best = max(amps)
            s = " ".join(f"{arm}@{best}={tally[(arm,best)][0]}/{tally[(arm,best)][1]}"
                         for arm in ARMS)
            print(f"  progress {n+1}/{len(items)} elapsed_s={el:.0f} yield {s}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows_out)

    parts = []
    for arm in ARMS:
        for amp in amps:
            c, t = tally[(arm, amp)]
            parts.append(f"{arm}_a{amp}={c}/{t}")
    print("RESULT task=%d " % a.task + " ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
