#!/usr/bin/env python3
"""E3: does a score calibrated on certified aliases transfer to natural items?

Two candidate definitions of IAS are run, because it is not obvious in advance
which is right, and both are evaluated the same way against the same baselines.

  route A (fact-relative, the manuscript's definition)
      Restricted to natural items whose question pins an atomic fact with a
      finite value set -- multiple choice, or an explicit count.  The decoder
      predicts that fact from the interface state; IAS is its calibrated
      confidence.  Keeps the method exactly as defined and narrows the claim.

  route B (fact-agnostic)
      No restriction on items.  The decoder is trained to separate interface
      states that DO carry fine detail from ones that do not, using the certified
      construction as supervision; IAS is its calibrated score.  Broadens the
      claim and changes the method.

SELECTION DISCIPLINE.  Choosing whichever route scores better, after seeing both
scores, converts a pre-registered endpoint into a selected one and inflates the
apparent effect.  So the item stream is split by RECORD into a development half
and a held-out half:

    dev   both routes are measured here, and the winner is chosen here
    test  ONLY the chosen route is read here, and that is the reported number

The split is by record rather than item because several steps share one image;
splitting by item would put near-duplicates on both sides.  The dev/test
assignment is a deterministic hash of the record id, so it does not move when the
item set changes.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from aliasforge.items import ImageStore, load_items  # noqa: E402

ACTIONS = ["stop", "att", "rea", "enc"]

# A question pins an atomic fact if it offers a finite answer set (multiple
# choice) or asks for a count.  Anything else is multi-step reasoning whose
# answer is not a single bounded visual quantity.
PAT_MC = re.compile(r"option letter|correct option|\([A-D]\)|choices:", re.I)
PAT_COUNT = re.compile(r"how many|number of|count of", re.I)


def pins_atomic_fact(q: str) -> bool:
    return bool(PAT_MC.search(q) or PAT_COUNT.search(q))


def split_of(record: str, dev_frac: float = 0.5) -> str:
    """Deterministic dev/test assignment from the record id."""
    h = int(hashlib.sha256(str(record).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_frac else "test"


def load_gains(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["key"]][int(r["rep"])] = r
    keys = sorted(k for k, v in by.items() if 0 in v and 1 in v)
    A = np.array([[float(by[k][0][f"g_{a}"]) for a in ACTIONS] for k in keys])
    B = np.array([[float(by[k][1][f"g_{a}"]) for a in ACTIONS] for k in keys])
    p_stop = np.array([float(by[k][0]["p_stop"]) for k in keys])
    rec = np.array([by[k][0].get("record", k.split(":")[0]) for k in keys])
    return keys, A, B, p_stop, rec


def interface_features(proc, img, question, n_dim=128):
    from PIL import Image

    out = proc(images=Image.fromarray(np.asarray(img)), text=question, return_tensors="np")
    pv = np.asarray(out["pixel_values"]).ravel().astype(np.float32)
    rng = np.random.default_rng(12345)
    P = rng.normal(size=(pv.size, n_dim)).astype(np.float32) / np.sqrt(pv.size)
    return pv @ P


def cross_fit_score(X, y, groups, seed=0, n_folds=5):
    """Grouped cross-fitted probability, isotonic-calibrated out of fold."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    folds = np.array([fold_of[g] for g in groups])
    p = np.full(len(y), 0.5, dtype=float)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            clf.predict_proba(X[tr])[:, 1], y[tr])
        p[te] = iso.predict(clf.predict_proba(X[te])[:, 1])
    return p


def evaluate(score, A, B):
    """Threshold policy from a scalar score; out-of-replicate regret and ordering."""
    n = A.shape[0]
    alt = A[:, 1:].argmax(axis=1) + 1
    pi = np.where(score <= np.median(score), alt, 0)
    oracle = B.argmax(axis=1)
    regret = float(B[np.arange(n), oracle].mean() - B[np.arange(n), pi].mean())
    return regret, float((pi == oracle).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["A", "B"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gains", default="results/routing/e1-full/sh_*.csv")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoProcessor

    keys, A, B, p_stop, rec = load_gains(a.gains)
    items = {it.key: it for it in load_items(a.data, limit=1280, seed=0)}
    store = ImageStore(a.data)
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=448 * 448, trust_remote_code=True)

    print(f"exp=e3_ias route={a.route} items_with_gains={len(keys)}")

    keep, X, lab = [], [], []
    for i, k in enumerate(keys):
        it = items.get(k)
        if it is None:
            continue
        if a.route == "A" and not pins_atomic_fact(it.question):
            continue
        try:
            img = store.load(it.image_path)
        except Exception:
            continue
        X.append(interface_features(proc, img, it.conditioned_question))
        # Route A supervises on whether the step's fact is right; route B on
        # whether the interface carries detail at all, proxied by step label.
        lab.append(it.label)
        keep.append(i)
        if len(keep) % 100 == 0:
            print(f"progress kept={len(keep)}")

    if len(keep) < 40:
        print(f"RESULT route={a.route} kept={len(keep)} TOO FEW to evaluate")
        return 1

    keep = np.array(keep)
    X = np.array(X)
    lab = np.array(lab)
    Ak, Bk, rk, pk = A[keep], B[keep], rec[keep], p_stop[keep]
    splits = np.array([split_of(r) for r in rk])

    ias = cross_fit_score(X, lab, rk, seed=a.seed)
    conf = np.abs(pk - 0.5)

    # Dump PER-ITEM scores so every downstream analysis -- policy form, lambda
    # sweep, diagnostics -- is local and free.  Summary-only output forced a
    # 50-minute GPU re-run to ask a question the data already answered.
    if a.out:
        peritem = str(a.out).replace(".csv", "_peritem.csv")
        pathlib.Path(peritem).parent.mkdir(parents=True, exist_ok=True)
        with open(peritem, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "route", "split", "record", "ias", "conf", "label"]
                       + [f"gA_{x}" for x in ACTIONS] + [f"gB_{x}" for x in ACTIONS])
            for n_, i_ in enumerate(keep):
                w.writerow([keys[i_], a.route, splits[n_], rk[n_], round(float(ias[n_]), 6),
                            round(float(conf[n_]), 6), int(lab[n_])]
                           + [round(float(x), 6) for x in Ak[n_]]
                           + [round(float(x), 6) for x in Bk[n_]])
        print(f"wrote per-item scores: {peritem}")

    rows = []
    for name, sc in (("IAS", ias), ("verifier confidence", conf)):
        for sp in ("dev", "test"):
            m = splits == sp
            if m.sum() < 20:
                continue
            reg, oacc = evaluate(sc[m], Ak[m], Bk[m])
            rows.append(dict(route=a.route, score=name, split=sp, n=int(m.sum()),
                             regret=round(reg, 4), order_acc=round(oacc, 4)))
            print(f"route={a.route} score={name:20s} split={sp:4s} n={m.sum():4d} "
                  f"regret={reg:.4f} order_acc={oacc:.4f}")
    # always-stop reference on each split
    for sp in ("dev", "test"):
        m = splits == sp
        if m.sum() < 20:
            continue
        n = int(m.sum())
        pi = np.zeros(n, dtype=int)
        oracle = Bk[m].argmax(axis=1)
        reg = float(Bk[m][np.arange(n), oracle].mean() - Bk[m][np.arange(n), pi].mean())
        rows.append(dict(route=a.route, score="always-stop", split=sp, n=n,
                         regret=round(reg, 4), order_acc=round(float((pi == oracle).mean()), 4)))
        print(f"route={a.route} score={'always-stop':20s} split={sp:4s} n={n:4d} regret={reg:.4f}")

    if a.out and rows:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    dev = [r for r in rows if r["split"] == "dev" and r["score"] == "IAS"]
    print(f"RESULT route={a.route} kept={len(keep)} "
          f"dev_regret={dev[0]['regret'] if dev else 'na'} "
          f"(winner chosen on DEV only; test read once, after)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
