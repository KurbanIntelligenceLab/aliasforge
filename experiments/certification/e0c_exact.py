#!/usr/bin/env python3
"""E0b + E0c: does the certificate hold on the REAL model, and can the test detect it if not?

E0a established that the exact integer-kernel construction makes the processor's
interface state bit-identical (15/15, controls clean).  That is a statement about
preprocessing.  E0c asks the question that actually matters: does the FROZEN
VERIFIER then produce the same output?  Theorem 1 says it must, exactly, because
the model reads the image only through that state.

We measure the decision-position logit vector rather than pair-balanced accuracy.
Accuracy needs many samples to resolve 1/2 to any precision and converts an exact
prediction into a noisy one; the logits are a deterministic function of the
interface and can be compared directly.

Three arms, because the certified number alone proves nothing:

  SELF      the same image pushed through twice.  GPU kernels are not always
            bit-reproducible, so this is the floor the pair difference must be
            judged against -- not zero.
  CERTIFIED the exact-kernel pair.  Prediction: difference AT the self floor.
  CONTROL   the same construction in a non-kernel direction, identical support and
            amplitude.  Prediction: difference far ABOVE the floor.  Without this
            arm a null result on the certified pair could just mean the model
            ignores the image, or that the harness fed the same tensor twice.

A certified pair above the floor while the control separates would mean an
image-dependent field is not being hashed -- the enumeration gap Appendix B warns
about -- and that is exactly what E0b is for.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.qwen import MAX_PIXELS_STOCK, smart_resize  # noqa: E402

SIDE, TARGET = 896, 448
BUDGET_ROUTER = TARGET * TARGET
QUESTION = "How many horizontal marks appear in the image?"
STEP = "The image contains exactly three horizontal marks."


def build_base(H, W, rng):
    return np.clip(
        128 + 18 * np.sin(np.mgrid[0:H, 0:W][1] / 11.0)[..., None]
        + rng.integers(-6, 7, (H, W, 3)), 0, 255).astype(np.uint8)


def row_masks(H, n_a=3, n_b=2):
    a, b = np.zeros(H, bool), np.zeros(H, bool)
    for i, y0 in enumerate((int(.13 * H), int(.38 * H), int(.63 * H))):
        if i < n_a:
            a[y0:y0 + int(.12 * H)] = True
        if i < n_b:
            b[y0:y0 + int(.12 * H)] = True
    return a, b


def run(model_dir, trials, out, seed=0):
    from aliasforge.verifier import Verifier

    rows = []
    print(f"exp=e0c_exact model={pathlib.Path(model_dir).name} side={SIDE} "
          f"target={TARGET} trials={trials} seed={seed}")

    K = pil_coeffs(SIDE, TARGET)
    vec, meta = find_short_vector(K, width=64, stride=16, max_abs=8)
    if vec is None:
        print("RESULT no kernel vector; aborting")
        return []
    print(f"kernel_vector max_abs={int(np.abs(vec).max())} nonzeros={int((vec!=0).sum())}")

    for bname, mp in (("router", BUDGET_ROUTER), ("stock", MAX_PIXELS_STOCK)):
        hb, wb = smart_resize(SIDE, SIDE, max_pixels=mp)
        print(f"budget={bname} max_pixels={mp} smart_resize=({hb},{wb}) "
              f"identity={(hb,wb)==(SIDE,SIDE)}")
        v = Verifier(model_dir, max_pixels=mp)
        print(f"  loaded device={v.device} yes_ids={v.yes_ids} no_ids={v.no_ids}")

        for trial in range(trials):
            rng = np.random.default_rng(seed * 100 + trial)
            base = build_base(SIDE, SIDE, rng)
            ma, mb = row_masks(SIDE)

            for arm in ("certified", "control"):
                if arm == "certified":
                    a, b, mm = make_exact_pair(base, vec, 20, ma, mb)
                else:
                    off = np.zeros_like(vec)
                    nz = np.flatnonzero(vec)
                    off[nz] = rng.integers(1, 4, size=nz.size)
                    a, b, mm = make_exact_pair(base, off, 20, ma, mb)
                if a is None or mm.get("pix_diff", 0) == 0:
                    continue

                r = v.pair_test(a, b, QUESTION, STEP)
                floor = max(r["self_max_abs_logit_diff"], 0.0)
                pair = r["pair_max_abs_logit_diff"]
                rows.append(dict(
                    budget=bname, trial=trial, arm=arm,
                    digests_collide=int(r["digests_collide"]),
                    self_diff=floor, pair_diff=pair,
                    at_floor=int(pair <= floor),
                    ratio=(pair / floor) if floor > 0 else (float("inf") if pair > 0 else 1.0),
                    ba=r["pair_balanced_accuracy"],
                    p_a=r["p_correct_a"], p_b=r["p_correct_b"],
                    pix_diff=mm["pix_diff"]))
                print(f"budget={bname} trial={trial} arm={arm} "
                      f"collide={int(r['digests_collide'])} self={floor:.3e} "
                      f"pair={pair:.3e} at_floor={int(pair<=floor)} "
                      f"BA={r['pair_balanced_accuracy']:.4f}")

    if out and rows:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    cert = [r for r in rows if r["budget"] == "router" and r["arm"] == "certified"]
    ctrl = [r for r in rows if r["budget"] == "router" and r["arm"] == "control"]
    at_floor = sum(r["at_floor"] for r in cert)
    sep = sum(1 for r in ctrl if r["pair_diff"] > max(r["self_diff"], 1e-12) * 10)
    print(f"RESULT certified_at_noise_floor={at_floor}/{len(cert)} "
          f"controls_separated={sep}/{len(ctrl)} "
          f"(theorem predicts {len(cert)}/{len(cert)} and {len(ctrl)}/{len(ctrl)})")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()
    run(a.model, a.trials, a.out, a.seed)
