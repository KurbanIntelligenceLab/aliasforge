#!/usr/bin/env python3
"""E0a: certified aliases against a REAL deployed interface, by exact integer algebra.

This is the construction that works.  Every earlier route is superseded and the
reasons are recorded in the module docstrings:

  * the dead-region (crop) route certified against `AutoProcessor`'s CLIP feature
    extractor, which is not the path VisualPRM is deployed with -- retracted;
  * the approximate null-space route leaves a ~1% per-output mismatch floor that
    is flat in the downscale factor, 6000x too high to certify 602112 outputs
    (0/72 measured over factors 1.41x-8x).

Here the perturbation is an integer vector in the EXACT integer kernel of
Pillow's fixed-point coefficient matrix, so the resampler's accumulator is
bit-identical before it is ever shifted.  Nothing is rounded and nothing is
searched.

Configuration.  The verifier is Qwen2.5-VL under a resolution budget of
`max_pixels = 448*448`, which is a legitimate resolution-adaptive policy and the
regime such a router operates in; at the stock 12.8 MP budget `smart_resize` is
the identity and no alias can exist, which is itself reported as the control.
At this budget an 896x896 input maps to exactly 448x448 -- an exact factor of 2,
where the integer kernel admits a vector with max|v| = 3, so the perturbation is
a tiny pixel offset that can be scaled to any legible amplitude and remain
exactly in the kernel.

Controls: a random-direction pair of the same amplitude that must NOT certify,
and rejection of any pixel-identical pair.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs, validate  # noqa: E402
from aliasforge.interface import capture, per_field_report, state_digest  # noqa: E402
from aliasforge.qwen import MAX_PIXELS_STOCK, smart_resize  # noqa: E402

SIDE = 896
TARGET = 448
BUDGET_ROUTER = TARGET * TARGET          # exact 2x downscale
QUESTION = "How many horizontal marks appear in the image?"


def build_base(H, W, rng):
    return np.clip(
        128 + 18 * np.sin(np.mgrid[0:H, 0:W][1] / 11.0)[..., None]
        + rng.integers(-6, 7, (H, W, 3)), 0, 255).astype(np.uint8)


def row_masks(H, n_a=3, n_b=2):
    """The atomic fact: how many horizontal marks are present."""
    a = np.zeros(H, bool)
    b = np.zeros(H, bool)
    for i, y0 in enumerate((int(.13 * H), int(.38 * H), int(.63 * H))):
        if i < n_a:
            a[y0:y0 + int(.12 * H)] = True
        if i < n_b:
            b[y0:y0 + int(.12 * H)] = True
    return a, b


def run(model_dir, trials, out, seed=0):
    from transformers import AutoProcessor

    rows = []
    print(f"exp=e0a_exact model={pathlib.Path(model_dir).name} side={SIDE} target={TARGET} "
          f"trials={trials} seed={seed}")
    for name, ok, detail in validate():
        print(f"coeff_check {name} pass={int(ok)} maxdiff={detail}")

    K = pil_coeffs(SIDE, TARGET)
    vec, meta = find_short_vector(K, width=64, stride=16, max_abs=8)
    if vec is None:
        print("RESULT no short integer kernel vector; aborting")
        return []
    print(f"kernel_vector max_abs={int(np.abs(vec).max())} nonzeros={int((vec!=0).sum())} "
          f"window={meta.get('c0')}")

    budgets = [("router", BUDGET_ROUTER), ("stock", MAX_PIXELS_STOCK)]
    for bname, mp in budgets:
        h_bar, w_bar = smart_resize(SIDE, SIDE, max_pixels=mp)
        identity = (h_bar, w_bar) == (SIDE, SIDE)
        print(f"budget={bname} max_pixels={mp} smart_resize({SIDE},{SIDE})=({h_bar},{w_bar}) "
              f"identity={identity}")
        proc = AutoProcessor.from_pretrained(model_dir, max_pixels=mp, trust_remote_code=True)

        for trial in range(trials):
            rng = np.random.default_rng(seed * 100 + trial)
            base = build_base(SIDE, SIDE, rng)
            ma, mb = row_masks(SIDE)

            for arm in ("exact_kernel", "negative_control"):
                if arm == "exact_kernel":
                    a, b, mm = make_exact_pair(base, vec, 20, ma, mb)
                else:
                    off = np.zeros_like(vec)
                    nz = np.flatnonzero(vec)
                    off[nz] = rng.integers(1, 4, size=nz.size)   # same support, wrong direction
                    a, b, mm = make_exact_pair(base, off, 20, ma, mb)
                if a is None or mm.get("pix_diff", 0) == 0:
                    rows.append(dict(budget=bname, max_pixels=mp, trial=trial, arm=arm,
                                     certified=0, pix_diff=0, amplitude=0,
                                     identity_resize=int(identity), fields_differ="degenerate"))
                    continue

                from PIL import Image
                sa = capture(proc, Image.fromarray(a), QUESTION)
                sb = capture(proc, Image.fromarray(b), QUESTION)
                rep = per_field_report(sa, sb)
                cert = int(state_digest(sa) == state_digest(sb))
                rows.append(dict(budget=bname, max_pixels=mp, trial=trial, arm=arm,
                                 certified=cert, pix_diff=mm["pix_diff"],
                                 amplitude=mm["max_amplitude"],
                                 identity_resize=int(identity),
                                 fields_differ=",".join(rep["differ"]) or "none"))
                print(f"budget={bname} trial={trial} arm={arm} certified={cert} "
                      f"pix_diff={mm['pix_diff']} amplitude={mm['max_amplitude']} "
                      f"fields_differ={','.join(rep['differ']) or 'none'}")

    if out and rows:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    for bname, mp in budgets:
        ek = [r for r in rows if r["budget"] == bname and r["arm"] == "exact_kernel"]
        nc = [r for r in rows if r["budget"] == bname and r["arm"] == "negative_control"]
        print(f"BUDGET_SUMMARY budget={bname} certified={sum(r['certified'] for r in ek)}/{len(ek)} "
              f"control_certified={sum(r['certified'] for r in nc)}")
    ek = [r for r in rows if r["arm"] == "exact_kernel" and r["budget"] == "router"]
    nc = [r for r in rows if r["arm"] == "negative_control"]
    print(f"RESULT certified={sum(r['certified'] for r in ek)}/{len(ek)} "
          f"control_false_certifications={sum(r['certified'] for r in nc)} (must be 0)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()
    run(a.model, a.trials, a.out, a.seed)
