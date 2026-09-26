#!/usr/bin/env python3
"""E34b: the strong router's behavior on the 80 certified pairs, natively.

E34 saved the vision-tower features of every natural item beside its shard.  This refits the
routers on those features (no natural-item extraction), extracts the same features for the 80
certified pairs, and records for each router BOTH the action it takes natively (argmax over
all four actions, stop included) and the best non-stop action the median rule would take.
The first is what an action-rate comparison against natural items needs; the second is what
the old dominated-selection column used.  GPU, for the vision tower only.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.verifier import Verifier                                     # noqa: E402

VERSION = "e34b.3"
ACTIONS = ["stop", "att", "rea", "enc"]
SIDE, TARGET = 896, 448
QUESTION = "How many full-width bands are drawn across the figure?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", default="results/routing/e34-strong-probe/sh_0000_features.npz")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--max-pixels", type=int, default=TARGET * TARGET)
    ap.add_argument("--n-pairs", type=int, default=40)
    a = ap.parse_args(); t0 = time.time()
    import torch
    from PIL import Image
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor

    z = np.load(a.features, allow_pickle=True)
    XP, XE, A = z["proj"].astype(np.float32), z["enc"].astype(np.float32), z["A"]
    feats = {"proj": XP, "enc": XE, "proj+enc": np.concatenate([XP, XE], 1)}
    # same learners as the interval script, so the certified-pair rows match the natural rows
    makers = {"ridge": lambda: RidgeCV(alphas=(1e-1, 1, 10, 100, 1e3, 1e4)),
              "gbt": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                           early_stopping=True, random_state=0)}
    print(f"exp=e34b_probe_pairs version={VERSION} natural_items={len(A)} dims={ {k: v.shape[1] for k, v in feats.items()} }", flush=True)

    v = Verifier(a.model, max_pixels=a.max_pixels); model, proc = v.model, v.processor
    visual = getattr(getattr(model, "model", model), "visual", None) or getattr(model, "visual")
    device = next(model.parameters()).device
    @torch.no_grad()
    def featurize(img):
        o = proc(images=Image.fromarray(np.asarray(img)), text=QUESTION, return_tensors="pt")
        pv = o["pixel_values"].to(device=device, dtype=next(visual.parameters()).dtype)
        out = visual(pv, grid_thw=o["image_grid_thw"].to(device))
        if isinstance(out, (tuple, list)):
            merged, pre = out[0], out[0]
        else:
            merged, pre = out.pooler_output, out.last_hidden_state
        merged, pre = merged.float(), pre.float()
        return (torch.cat([merged.mean(0), merged.amax(0)]).cpu().numpy(),
                torch.cat([pre.mean(0), pre.std(0)]).cpu().numpy())

    # the certified pairs, built exactly as the screen measurement built them
    K = pil_coeffs(SIDE, TARGET); vec, _ = find_short_vector(K, width=64, stride=16, max_abs=8)
    m3 = np.zeros(SIDE, bool); m2 = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13*SIDE), int(.38*SIDE), int(.63*SIDE))):
        if i < 3: m3[y0:y0+int(.12*SIDE)] = True
        if i < 2: m2[y0:y0+int(.12*SIDE)] = True
    pair_feats = []
    for seed in (0, 1):
        for b in range(a.n_pairs):
            rr = np.random.default_rng(seed * 1000 + b)
            base = np.clip(128 + 18 * np.sin(np.mgrid[0:SIDE, 0:SIDE][1] / 11.)[..., None]
                           + rr.integers(-6, 7, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
            x0, x1, _ = make_exact_pair(base, vec, 20, m3, m2)
            if x0 is None: continue
            fp0, fe0 = featurize(x0); fp1, fe1 = featurize(x1)
            same = bool(np.allclose(fp0, fp1) and np.allclose(fe0, fe1))
            pair_feats.append((seed, b, fp0, fe0, same))
    print(f"pairs built={len(pair_feats)} features_identical_across_members={sum(p[4] for p in pair_feats)}/{len(pair_feats)} elapsed_s={time.time()-t0:.0f}", flush=True)

    rows = [["router", "features", "seed", "pair", "native_action", "nonstop_action"]]
    summary = []
    for fname, X in feats.items():
        mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
        for mname, mk in makers.items():
            full = [mk().fit(Xs, A[:, j]) for j in range(len(ACTIONS))]
            nat_pi = np.array([m.predict(Xs) for m in full]).T.argmax(1)        # in-sample, for the action-rate reference only
            acts = nonstop = 0
            for seed, b, fp, fe, _ in pair_feats:
                f = {"proj": fp, "enc": fe, "proj+enc": np.concatenate([fp, fe])}[fname]
                gh = np.array([float(m.predict(((f - mu) / sd)[None, :])[0]) for m in full])
                native = ACTIONS[int(gh.argmax())]; ns = ACTIONS[1:][int(gh[1:].argmax())]
                rows.append([mname, fname, seed, b, native, ns]); acts += native != "stop"; nonstop += ns in ("att", "rea")
            n = len(pair_feats)
            summary.append(dict(router=mname, features=fname, pairs=n, acts_on_pairs=acts, act_rate_pairs=round(acts / n, 4),
                                nonstop_dominated=nonstop, natural_act_rate_insample=round(float((nat_pi != 0).mean()), 4)))
            print(f"RESULT router={mname} features={fname} acts_on_pairs={acts}/{n} nonstop_dominated={nonstop}/{n} natural_act_rate_insample={(nat_pi!=0).mean():.3f}", flush=True)
    with open(a.out, "w", newline="") as fh: csv.writer(fh).writerows(rows)
    stem = os.path.splitext(a.out)[0]
    with open(stem + "_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(f"RESULT pairs={len(pair_feats)} routers={len(summary)} elapsed_s={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
