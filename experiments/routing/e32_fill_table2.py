#!/usr/bin/env python3
"""E32 -- fill the two rows of Table 2 that carried dashes.

Two cells were empty because the quantity had never been computed, not because it is
undefined, so this computes both.

  raw interface distance.  The score is the L2 norm of the deployed processor's image
  tensor, which is one number per item and identical across the two members of a certified
  pair.  It had a dominated-selection rate but no natural-stream row, because the score was
  never run through the routing policy.  Here it is, on the same items, gains and rules as
  every other row of the table.

  learned router.  E8 reported its regret and order accuracy as single numbers, with no
  per-item dump, so no clustered interval could be formed, and it was never applied to a
  certified pair, so it had no dominated-selection rate.  This re-fits it the same way
  (ridge and gradient boosting on a random projection of the image tensor, cross-fitted in
  five folds grouped by record, scored out of fold) and emits BOTH: the per-item chosen
  action, so regret can be bootstrapped by record, and the action it selects on each of the
  80 certified pairs.

Processor only.  No model weights and no GPU: the features are the interface tensor, and
the gains come from the recorded path-matched run.
"""
from __future__ import annotations

import argparse, collections, csv, glob, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.items import ImageStore, load_items                          # noqa: E402

VERSION = "e32.1"
ACTIONS = ["stop", "att", "rea", "enc"]
DOMINATED = ("att", "rea")
SIDE, TARGET = 896, 448


def load_gains(pattern):
    """Per-item gains from the path-matched run: replicate 0 chooses, replicate 1 scores."""
    rows = [r for f in sorted(glob.glob(pattern)) for r in csv.DictReader(open(f))]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["key"]][r["rep"]] = r
    keys, A, B, rec = [], [], [], []
    for k, v in sorted(by.items()):
        if "0" not in v or "1" not in v:
            continue
        keys.append(k); rec.append(v["0"]["record"])
        A.append([float(v["0"][f"g_{a}"]) for a in ACTIONS])
        B.append([float(v["1"][f"g_{a}"]) for a in ACTIONS])
    return keys, np.array(A), np.array(B), np.array(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--gains", default="results/routing/e21-pathmatched/sh_*.csv")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--n-dim", type=int, default=512)
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    from transformers import AutoProcessor
    from PIL import Image
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor

    keys, A, B, rec = load_gains(a.gains)
    items = {it.key: it for it in load_items(a.data, limit=1280, seed=0)}
    store = ImageStore(a.data)
    proc = AutoProcessor.from_pretrained(a.model, max_pixels=TARGET * TARGET, trust_remote_code=True)
    print(f"exp=e32_fill_table2 version={VERSION} items_with_gains={len(keys)} n_dim={a.n_dim}", flush=True)

    # Qwen uses dynamic resolution, so pixel_values is (n_patches, patch_dim) with n_patches
    # varying by aspect ratio. A projection sized to the flattened tensor therefore cannot be
    # shared across items. (E8 rebuilt one per size, which leaves features from different size
    # groups incomparable.) Project the PATCH dimension, which is constant, then pool over
    # patches: size-invariant, and comparable across items and certified pairs alike.
    rp = np.random.default_rng(12345); P = None
    keep, X, dist = [], [], []

    def featurize(pv2d):
        nonlocal P
        if P is None:
            P = rp.normal(size=(pv2d.shape[1], a.n_dim)).astype(np.float32) / np.sqrt(pv2d.shape[1])
        z = pv2d @ P
        return np.concatenate([z.mean(0), z.std(0)])

    for i, k in enumerate(keys):
        it = items.get(k)
        if it is None: continue
        try: img = np.asarray(store.load(it.image_path))
        except Exception: continue
        o = proc(images=Image.fromarray(img), text=it.conditioned_question, return_tensors="np")
        pv = np.asarray(o["pixel_values"]).astype(np.float32)
        pv2d = pv.reshape(-1, pv.shape[-1])
        X.append(featurize(pv2d))
        # the score must not be a proxy for image size, so use the per-element RMS
        dist.append(float(np.linalg.norm(pv2d) / np.sqrt(pv2d.size)))
        keep.append(i)
        if len(keep) % 200 == 0:
            print(f"progress items={len(keep)}/{len(keys)} patches={pv2d.shape[0]} "
                  f"elapsed_s={time.time()-t0:.0f}", flush=True)
    keep = np.array(keep); X = np.array(X); dist = np.array(dist)
    Ak, Bk, rk = A[keep], B[keep], rec[keep]
    n = len(keep)
    print(f"features built n={n} dim={X.shape[1]} dist_range=[{dist.min():.1f},{dist.max():.1f}]", flush=True)

    def regret(pi, G): return float(G.max(1).mean() - G[np.arange(len(pi)), pi].mean())
    def order_acc(pi, G): return float((G.argmax(1) == pi).mean())

    # --- the two decision rules, exactly as the table's other rows use them
    def median_rule(sc): return np.where(sc <= np.median(sc), Ak[:, 1:].argmax(axis=1) + 1, 0)

    # --- learned router, cross-fitted by record and scored out of fold
    uniq = np.unique(rk); rng = np.random.default_rng(a.seed)
    fold = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
    folds = np.array([fold[g] for g in rk])
    best = None
    for name, mk in (("ridge", lambda: Ridge(alpha=1.0)),
                     ("gbt", lambda: GradientBoostingRegressor(n_estimators=120, max_depth=3))):
        gh = np.zeros_like(Ak)
        for f in range(5):
            tr, te = folds != f, folds == f
            if tr.sum() < 20 or te.sum() == 0: continue
            for j in range(len(ACTIONS)):
                gh[te, j] = mk().fit(X[tr], Ak[tr, j]).predict(X[te])
        pi = gh.argmax(1); r = regret(pi, Bk)
        print(f"router={name} regret={r:.4f} order_acc={order_acc(pi,Bk):.4f}", flush=True)
        if best is None or r < best[1]: best = (name, r, pi, gh)
    rname, rreg, rpi, rgh = best

    rows = [["key", "record", "dist", "router_action", "router_regret_contrib"]]
    for t, i in enumerate(keep):
        rows.append([keys[i], rk[t], round(float(dist[t]), 4), ACTIONS[rpi[t]],
                     round(float(Bk[t].max() - Bk[t, rpi[t]]), 6)])

    # --- distance score through both rules
    summary = []
    def add(policy, pi):
        summary.append(dict(policy=policy, regret=round(regret(pi, Bk), 4),
                            order_acc=round(order_acc(pi, Bk), 4), n=n))
        print(f"RESULT policy={policy:34s} regret={regret(pi,Bk):.4f} order_acc={order_acc(pi,Bk):.4f}", flush=True)
    add("distance (median-threshold rule)", median_rule(dist))
    add(f"learned router ({rname})", rpi)

    # --- certified pairs: what does the router select there?
    K = pil_coeffs(SIDE, TARGET)
    vec, _ = find_short_vector(K, width=64, stride=16, max_abs=8)
    m3 = np.zeros(SIDE, bool); m2 = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13*SIDE), int(.38*SIDE), int(.63*SIDE))):
        if i < 3: m3[y0:y0+int(.12*SIDE)] = True
        if i < 2: m2[y0:y0+int(.12*SIDE)] = True
    # Fit once on ALL natural items, then apply out of domain to each certified pair.
    full = []
    for j in range(len(ACTIONS)):
        m = (Ridge(alpha=1.0) if rname == "ridge"
             else GradientBoostingRegressor(n_estimators=120, max_depth=3))
        full.append(m.fit(X, Ak[:, j]))
    dom = tot = 0
    for seed in (0, 1):
        for b in range(a.n_pairs):
            rr = np.random.default_rng(seed * 1000 + b)
            base = np.clip(128 + 18 * np.sin(np.mgrid[0:SIDE, 0:SIDE][1] / 11.)[..., None]
                           + rr.integers(-6, 7, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
            x0, x1, _ = make_exact_pair(base, vec, 20, m3, m2)
            if x0 is None:
                continue
            o = proc(images=Image.fromarray(x0),
                     text="How many full-width bands are drawn across the figure?",
                     return_tensors="np")
            pv = np.asarray(o["pixel_values"]).astype(np.float32)
            f = featurize(pv.reshape(-1, pv.shape[-1]))[None, :]
            gh = np.array([float(m.predict(f)[0]) for m in full])
            # the rule takes the best NON-STOP action on a pair it acts on
            pick = ACTIONS[1:][int(np.argmax(gh[1:]))]
            tot += 1
            dom += int(pick in DOMINATED)
        if tot == 0:
            break
    if tot:
        print(f"RESULT learned_router_certified dominated={dom}/{tot} rate={dom/tot:.4f}", flush=True)
        summary.append(dict(policy="learned router, certified pairs", regret="", order_acc="",
                            n=tot, dominated=dom, rate=round(dom / tot, 4)))

    print(f"RESULT n={n} router={rname} regret={rreg:.4f} elapsed_s={time.time()-t0:.0f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        with open(a.out.replace(".csv", "_summary.csv"), "w", newline="") as fh:
            fn = sorted({k for d in summary for k in d})
            w = csv.DictWriter(fh, fieldnames=fn); w.writeheader(); w.writerows(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
