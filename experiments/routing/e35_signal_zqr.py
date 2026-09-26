#!/usr/bin/env python3
"""E35: a router whose signal reads the full interface state (z, Q, R).

Proposition 4 bounds any router whose signal is a function of the interface state.  The
vision-tower routers of E34 read z alone.  The richest signal the frozen verifier computes
from (z, Q, R) is the language model's hidden state at the decision position, the vector the
Yes/No logits are a linear readout of.  This job extracts it on every natural item and on
both members of the 80 certified pairs.

It writes features only.  The routers are fit on CPU by analysis/e35_intervals.py with
the same cross-fitting, folds and record-clustered bootstrap as the E34 rows, so the rows are
comparable.  The pair features also test the theorem directly: the two members of a certified
pair share (z, Q, R), so their hidden states must be bit-identical.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.items import ImageStore, load_items                          # noqa: E402
from aliasforge.verifier import Verifier                                     # noqa: E402
from e34_strong_probe import load_gains                                     # noqa: E402

VERSION = "e35.1"
SIDE, TARGET = 896, 448
QUESTION = "How many full-width bands are drawn across the figure?"
STEP = "The figure shows three full-width bands, so the count is three."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--gains", default="results/routing/e21-pathmatched/sh_*.csv")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--max-pixels", type=int, default=TARGET * TARGET)
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N natural items only")
    a = ap.parse_args(); t0 = time.time()
    import torch

    keys, A, B, rec = load_gains(a.gains)
    items = {it.key: it for it in load_items(a.data, limit=1280, seed=0)}
    store = ImageStore(a.data)
    v = Verifier(a.model, max_pixels=a.max_pixels)
    model = v.model
    print(f"exp=e35_signal_zqr version={VERSION} items_with_gains={len(keys)} max_pixels={a.max_pixels} "
          f"signal=decision_position_hidden_state limit={a.limit}", flush=True)

    @torch.no_grad()
    def featurize(img, question, step):
        enc = v._inputs(img, question, step)
        enc = {k: t.to(v.device) if hasattr(t, "to") else t for k, t in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        last = hs[-1][0, -1, :].float().cpu().numpy()            # what the Yes/No logits read
        mid = hs[len(hs) // 2][0, -1, :].float().cpu().numpy()   # a mid-depth view of the same position
        lg = out.logits[0, -1, :].float().cpu().numpy()
        y = float(np.logaddexp.reduce(lg[v.yes_ids])); n = float(np.logaddexp.reduce(lg[v.no_ids]))
        return last, mid, float(1.0 / (1.0 + np.exp(n - y)))

    keep, XL, XM, P = [], [], [], []
    for i, k in enumerate(keys):
        if a.limit and len(keep) >= a.limit: break
        it = items.get(k)
        if it is None: continue
        try: img = np.asarray(store.load(it.image_path).convert("RGB"))
        except Exception: continue
        last, mid, p = featurize(img, it.conditioned_question, it.step)
        XL.append(last); XM.append(mid); P.append(p); keep.append(i)
        if len(keep) % 200 == 0:
            print(f"progress items={len(keep)}/{len(keys)} elapsed_s={time.time()-t0:.0f}", flush=True)
    keep = np.array(keep); XL = np.array(XL, dtype=np.float32); XM = np.array(XM, dtype=np.float32)
    print(f"natural features n={len(keep)} dim={XL.shape[1]} elapsed_s={time.time()-t0:.0f}", flush=True)

    # --- certified pairs: same construction and seeds as E34
    K = pil_coeffs(SIDE, TARGET)
    vec, _ = find_short_vector(K, width=64, stride=16, max_abs=8)
    m3 = np.zeros(SIDE, bool); m2 = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13*SIDE), int(.38*SIDE), int(.63*SIDE))):
        if i < 3: m3[y0:y0+int(.12*SIDE)] = True
        if i < 2: m2[y0:y0+int(.12*SIDE)] = True
    PL, PM, PP, pid, maxdiff, same_digest = [], [], [], [], 0.0, 0
    for seed in (0, 1):
        for b in range(a.n_pairs if not a.limit else 2):
            rr = np.random.default_rng(seed * 1000 + b)
            base = np.clip(128 + 18 * np.sin(np.mgrid[0:SIDE, 0:SIDE][1] / 11.)[..., None]
                           + rr.integers(-6, 7, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
            x0, x1, _ = make_exact_pair(base, vec, 20, m3, m2)
            if x0 is None: continue
            l0, m0, p0 = featurize(x0, QUESTION, STEP)
            l1, m1, p1 = featurize(x1, QUESTION, STEP)
            maxdiff = max(maxdiff, float(np.abs(l0 - l1).max()), float(np.abs(m0 - m1).max()), abs(p0 - p1))
            same_digest += int(v.interface_digest(x0, QUESTION, STEP) == v.interface_digest(x1, QUESTION, STEP))
            PL.append(l0); PM.append(m0); PP.append(p0); pid.append(f"s{seed}b{b}")
    print(f"RESULT certified_pairs n={len(pid)} same_digest={same_digest} max_abs_signal_diff={maxdiff:.3g} "
          f"elapsed_s={time.time()-t0:.0f}", flush=True)

    stem = os.path.splitext(a.out)[0]
    np.savez_compressed(stem + "_features.npz", keys=np.array([keys[i] for i in keep]), rec=rec[keep],
                        last=XL, mid=XM, conf=np.array(P), A=A[keep], B=B[keep],
                        pair_id=np.array(pid), pair_last=np.array(PL, dtype=np.float32),
                        pair_mid=np.array(PM, dtype=np.float32), pair_conf=np.array(PP),
                        pair_max_abs_signal_diff=maxdiff, pair_same_digest=same_digest)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key", "record", "conf"])
        for t, i in enumerate(keep):
            w.writerow([keys[i], rec[i], round(P[t], 6)])
    print(f"RESULT n={len(keep)} pairs={len(pid)} dim={XL.shape[1]} max_abs_signal_diff={maxdiff:.3g} "
          f"elapsed_s={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
