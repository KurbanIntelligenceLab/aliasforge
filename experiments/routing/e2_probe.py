#!/usr/bin/env python3
"""E2: is interface accessibility measurable, and does the certified set anchor it?

The score is the calibrated probability that an auxiliary decoder recovers an
atomic fact from the interface state alone.  E2 asks whether that decoder works,
and the certified set supplies the check that makes the answer falsifiable:

    On a certified alias the fact is PROVABLY not decodable from the interface --
    both members induce the same state, so no function of that state can tell
    them apart.  The decoder must therefore sit at chance.  Not "near" chance:
    exactly chance in expectation, and any systematic departure means the probe
    is reading something other than the interface.

That is a genuinely falsifiable prediction, which is rare for a probe. Most
probing results have no point at which they are required to fail.

Three arms per base image, all built by the same constructor so only the
direction of the perturbation differs:

    accessible  the fact is written in a NON-kernel direction, so it survives
                into the interface.  The decoder should recover it -- if it
                cannot, the probe is simply weak and nothing else here means
                anything.
    certified   the fact is written in the exact integer kernel, so it provably
                does NOT survive.  The decoder MUST be at chance.
    shuffled    the accessible arm with labels permuted within the fold.  This is
                the manuscript's leakage control: it must also collapse to
                chance, and if it does not, the pipeline is leaking the label
                through something other than the pixels.

Cross-fitting is grouped by BASE IMAGE, not by item: the two members of a pair
share a base, so an item-level split would put near-identical images on both
sides of the fold and inflate the probe -- the standard way this measurement is
made to look better than it is.
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
from aliasforge.qwen import smart_resize  # noqa: E402

SIDE, TARGET = 896, 448
BUDGET = TARGET * TARGET
QUESTION = "How many horizontal marks appear in the image?"


def build_base(rng):
    return np.clip(
        128 + 18 * np.sin(np.mgrid[0:SIDE, 0:SIDE][1] / 11.0)[..., None]
        + rng.integers(-6, 7, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)


def row_masks(n):
    """The atomic fact: how many marks are present.  Finite value set {2, 3}."""
    m = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13 * SIDE), int(.38 * SIDE), int(.63 * SIDE))):
        if i < n:
            m[y0:y0 + int(.12 * SIDE)] = True
    return m


def features(proc, img):
    """The interface state itself, reduced to a fixed-length vector.

    The decoder must read the INTERFACE, not the image, so features come from the
    processor output rather than from pixels.  Reduction is a fixed random
    projection: it is label-independent and identical across arms, so it cannot
    manufacture a difference between them.
    """
    from PIL import Image

    out = proc(images=Image.fromarray(img), text=QUESTION, return_tensors="np")
    pv = np.asarray(out["pixel_values"]).ravel().astype(np.float32)
    rng = np.random.default_rng(12345)                     # fixed, shared by all arms
    P = rng.normal(size=(pv.size, 128)).astype(np.float32) / np.sqrt(pv.size)
    return pv @ P


def cross_fit_auc(X, y, groups, n_folds=5, seed=0):
    """Grouped cross-fitted AUC + Brier, with isotonic calibration fitted out of fold."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    folds = np.array([fold_of[g] for g in groups])

    p = np.zeros(len(y), dtype=float)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            p[te] = 0.5
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        raw_tr = clf.predict_proba(X[tr])[:, 1]
        raw_te = clf.predict_proba(X[te])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_tr, y[tr])
        p[te] = iso.predict(raw_te)

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(np.mean((p - y) ** 2))
    # ECE, 10 equal-width bins
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return auc, brier, float(ece), p


def run(model_dir, n_base, out, seed=0):
    from transformers import AutoProcessor

    print(f"exp=e2_probe model={pathlib.Path(model_dir).name} n_base={n_base} seed={seed}")
    hb, wb = smart_resize(SIDE, SIDE, max_pixels=BUDGET)
    print(f"budget={BUDGET} smart_resize=({hb},{wb}) identity={(hb,wb)==(SIDE,SIDE)}")

    proc = AutoProcessor.from_pretrained(model_dir, max_pixels=BUDGET, trust_remote_code=True)
    K = pil_coeffs(SIDE, TARGET)
    vec, _ = find_short_vector(K, width=64, stride=16, max_abs=8)
    if vec is None:
        print("RESULT no kernel vector; aborting")
        return []
    print(f"kernel_vector max_abs={int(np.abs(vec).max())} nonzeros={int((vec!=0).sum())}")

    m3, m2 = row_masks(3), row_masks(2)
    data = {a: {"X": [], "y": [], "g": []} for a in ("accessible", "certified")}

    for b in range(n_base):
        rng = np.random.default_rng(seed * 1000 + b)
        base = build_base(rng)
        # non-kernel direction: same support and amplitude, but it SURVIVES the resize
        off = np.zeros_like(vec)
        nz = np.flatnonzero(vec)
        off[nz] = rng.integers(1, 4, size=nz.size)

        for arm, v in (("accessible", off), ("certified", vec)):
            a_img, b_img, meta = make_exact_pair(base, v, 20, m3, m2)
            if a_img is None:
                continue
            for img, lab in ((a_img, 1), (b_img, 0)):        # label = "three marks"
                data[arm]["X"].append(features(proc, img))
                data[arm]["y"].append(lab)
                data[arm]["g"].append(b)                      # group = base image
        if (b + 1) % 10 == 0:
            print(f"progress base={b+1}/{n_base}")

    rows = []
    for arm in ("accessible", "certified"):
        X = np.array(data[arm]["X"])
        y = np.array(data[arm]["y"])
        g = np.array(data[arm]["g"])
        if len(y) < 10:
            print(f"arm={arm} too few items ({len(y)})")
            continue
        auc, brier, ece, _ = cross_fit_auc(X, y, g, seed=seed)
        rows.append(dict(arm=arm, n=len(y), auc=round(auc, 4), brier=round(brier, 4),
                         ece=round(ece, 4)))
        print(f"arm={arm} n={len(y)} AUC={auc:.4f} Brier={brier:.4f} ECE={ece:.4f}")

        if arm == "accessible":                              # leakage control
            ysh = y.copy()
            r = np.random.default_rng(seed + 7)
            for gg in np.unique(g):                          # permute WITHIN group
                m = g == gg
                ysh[m] = r.permutation(ysh[m])
            a2, b2, e2 = cross_fit_auc(X, ysh, g, seed=seed)[:3]
            rows.append(dict(arm="shuffled", n=len(ysh), auc=round(a2, 4),
                             brier=round(b2, 4), ece=round(e2, 4)))
            print(f"arm=shuffled n={len(ysh)} AUC={a2:.4f} Brier={b2:.4f} ECE={e2:.4f}")

    if out and rows:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    got = {r["arm"]: r["auc"] for r in rows}
    acc, cert, shuf = got.get("accessible"), got.get("certified"), got.get("shuffled")
    # The anchor: certified must be at chance while accessible is not.
    anchor_ok = cert is not None and abs(cert - 0.5) < 0.10
    probe_ok = acc is not None and acc > 0.70
    shuf_ok = shuf is not None and abs(shuf - 0.5) < 0.10
    print(f"RESULT accessible_auc={acc} certified_auc={cert} shuffled_auc={shuf} "
          f"probe_works={int(probe_ok)} anchor_at_chance={int(anchor_ok)} "
          f"shuffle_at_chance={int(shuf_ok)}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--n-base", type=int, default=60)
    a = ap.parse_args()
    run(a.model, a.n_base, a.out, a.seed)
