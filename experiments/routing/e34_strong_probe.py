#!/usr/bin/env python3
"""E34: the strong Phi-measurable probe.

A review objected, correctly, that V_Phi ~ 0 in Table 2 is a failure to fit rather than a
measurement: any fitted score only lower-bounds the supremum Proposition 4 talks about, and the
two probes tried read a random projection of the pixel tensor.  The obvious strong probe reads
the vision tower itself.  Its outputs are deterministic functions of the interface tensor z, so
they are Phi-measurable and Proposition 4 applies to them unchanged; they are also the richest
function of z the model computes before the text touches anything.

This job runs the frozen verifier's vision tower on every natural item, pools the merged visual
tokens (the projector output the language model actually reads) and the pre-merger patch
features, fits the same cross-fitted router as the fill run on each, and scores it on the
held-out replicate.  It then applies the router to the 80 certified pairs.  Per-item rows are
written in the fill run's format so the table script consumes them with --fill.  Features are
saved beside the shard so the routers can be re-fit on CPU without the model.
"""
from __future__ import annotations
import argparse, collections, csv, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.items import ImageStore, load_items                          # noqa: E402
from aliasforge.verifier import Verifier                                     # noqa: E402

VERSION = "e34.2"
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
    ap.add_argument("--max-pixels", type=int, default=TARGET * TARGET)
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    import torch
    from PIL import Image
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import GradientBoostingRegressor

    keys, A, B, rec = load_gains(a.gains)
    items = {it.key: it for it in load_items(a.data, limit=1280, seed=0)}
    store = ImageStore(a.data)
    v = Verifier(a.model, max_pixels=a.max_pixels)
    model, proc = v.model, v.processor
    visual = getattr(getattr(model, "model", model), "visual", None) or getattr(model, "visual")
    device = next(model.parameters()).device
    print(f"exp=e34_strong_probe version={VERSION} items_with_gains={len(keys)} "
          f"max_pixels={a.max_pixels} visual={type(visual).__name__}", flush=True)

    @torch.no_grad()
    def featurize(img, question):
        o = proc(images=Image.fromarray(np.asarray(img)), text=question, return_tensors="pt")
        pv = o["pixel_values"].to(device=device, dtype=next(visual.parameters()).dtype)
        thw = o["image_grid_thw"].to(device)
        out = visual(pv, grid_thw=thw)
        # transformers 5.16 returns both tensors: pooler_output is the merger output the
        # language model reads, last_hidden_state the pre-merger patch features.
        if isinstance(out, (tuple, list)):
            merged, pre = out[0], out[0]
        else:
            merged, pre = out.pooler_output, out.last_hidden_state
        merged, pre = merged.float(), pre.float()
        f_proj = torch.cat([merged.mean(0), merged.amax(0)]).cpu().numpy()
        f_enc = torch.cat([pre.mean(0), pre.std(0)]).cpu().numpy()
        return f_proj, f_enc

    keep, XP, XE = [], [], []
    for i, k in enumerate(keys):
        it = items.get(k)
        if it is None: continue
        try: img = np.asarray(store.load(it.image_path).convert("RGB"))
        except Exception: continue
        fp, fe = featurize(img, it.conditioned_question)
        XP.append(fp); XE.append(fe); keep.append(i)
        if len(keep) % 200 == 0:
            print(f"progress items={len(keep)}/{len(keys)} elapsed_s={time.time()-t0:.0f}", flush=True)
    keep = np.array(keep); XP = np.array(XP, dtype=np.float32); XE = np.array(XE, dtype=np.float32)
    Ak, Bk, rk = A[keep], B[keep], rec[keep]; n = len(keep)
    print(f"features built n={n} dim_proj={XP.shape[1]} dim_enc={XE.shape[1]} elapsed_s={time.time()-t0:.0f}", flush=True)
    stem = os.path.splitext(a.out)[0]
    np.savez_compressed(stem + "_features.npz", keys=np.array([keys[i] for i in keep]), rec=rk,
                        proj=XP, enc=XE, A=Ak, B=Bk)

    def regret(pi, G): return float(G.max(1).mean() - G[np.arange(len(pi)), pi].mean())
    def order_acc(pi, G): return float((G.argmax(1) == pi).mean())

    # --- cross-fitted routers, folds grouped by record, scored out of fold on the held-out replicate
    uniq = np.unique(rk); rng = np.random.default_rng(a.seed)
    fold = {g: i % 5 for i, g in enumerate(rng.permutation(uniq))}
    folds = np.array([fold[g] for g in rk])
    feats = {"proj": XP, "enc": XE, "proj+enc": np.concatenate([XP, XE], 1)}
    makers = {"ridge": lambda: RidgeCV(alphas=(1e-1, 1, 10, 100, 1e3, 1e4)),
              "gbt": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, subsample=0.8, random_state=0)}
    summary, best = [], None
    for fname, X in feats.items():
        mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
        for mname, mk in makers.items():
            gh = np.zeros_like(Ak)
            for f in range(5):
                tr, te = folds != f, folds == f
                for j in range(len(ACTIONS)):
                    gh[te, j] = mk().fit(Xs[tr], Ak[tr, j]).predict(Xs[te])
            pi = gh.argmax(1); r, o = regret(pi, Bk), order_acc(pi, Bk)
            gain = float(Bk[np.arange(n), pi].mean() - Bk[:, 0].mean())
            name = f"learned router ({mname}, {fname})"
            summary.append(dict(policy=name, regret=round(r, 4), order_acc=round(o, 4), gain_over_stop=round(gain, 4), n=n))
            print(f"RESULT policy={name:40s} regret={r:.4f} order_acc={o:.4f} gain_over_stop={gain:+.4f}", flush=True)
            if best is None or r < best[1]: best = (name, r, pi, fname, mname)
    bname, breg, bpi, bfeat, bmodel = best
    print(f"best router: {bname} regret={breg:.4f}", flush=True)

    rows = [["key", "record", "router_action", "router_regret_contrib"]]
    for t, i in enumerate(keep):
        rows.append([keys[i], rk[t], ACTIONS[bpi[t]], round(float(Bk[t].max() - Bk[t, bpi[t]]), 6)])
    with open(a.out, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    # --- certified pairs: fit the best router on all natural items, apply out of domain
    X = feats[bfeat]; mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
    full = [makers[bmodel]().fit(Xs, Ak[:, j]) for j in range(len(ACTIONS))]
    K = pil_coeffs(SIDE, TARGET)
    vec, _ = find_short_vector(K, width=64, stride=16, max_abs=8)
    m3 = np.zeros(SIDE, bool); m2 = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13*SIDE), int(.38*SIDE), int(.63*SIDE))):
        if i < 3: m3[y0:y0+int(.12*SIDE)] = True
        if i < 2: m2[y0:y0+int(.12*SIDE)] = True
    dom = tot = 0
    for seed in (0, 1):
        for b in range(a.n_pairs):
            rr = np.random.default_rng(seed * 1000 + b)
            base = np.clip(128 + 18 * np.sin(np.mgrid[0:SIDE, 0:SIDE][1] / 11.)[..., None]
                           + rr.integers(-6, 7, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
            x0, x1, _ = make_exact_pair(base, vec, 20, m3, m2)
            if x0 is None: continue
            fp, fe = featurize(x0, "How many full-width bands are drawn across the figure?")
            f = {"proj": fp, "enc": fe, "proj+enc": np.concatenate([fp, fe])}[bfeat]
            f = ((f - mu) / sd)[None, :]
            gh = np.array([float(m.predict(f)[0]) for m in full])
            pick = ACTIONS[1:][int(np.argmax(gh[1:]))]   # the rule takes the best non-stop action
            tot += 1; dom += int(pick in DOMINATED)
    if tot:
        print(f"RESULT learned_router_certified dominated={dom}/{tot} rate={dom/tot:.4f}", flush=True)
        summary.append(dict(policy=f"{bname}, certified pairs", regret="", order_acc="", gain_over_stop="", n=tot, dominated=dom, rate=round(dom / tot, 4)))
    with open(stem + "_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["policy", "regret", "order_acc", "gain_over_stop", "n", "dominated", "rate"])
        w.writeheader(); w.writerows(summary)
    print(f"RESULT n={n} best={bname} regret={breg:.4f} elapsed_s={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
