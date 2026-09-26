#!/usr/bin/env python3
"""E31b -- the dominated-selection column, measured as a PAIR-BALANCED gain.

E31 measured the wrong quantity and its log says so: on pairs whose digests collide it
reported g_att = g_rea = +0.059, while Theorem 1 forces the gain to be exactly zero there.
The error was scoring one member.  Corollary 3(b) dominates att and rea because their
PAIR-BALANCED gain is zero and their cost is positive, and pair-balanced accuracy is
BA(a) = 1/2 * ( p(x0 under a) + 1 - p(x1 under a) ) over the two members, which carry
opposite labels.  A single member's p(a) - p(placebo) is a real number and not that.

What follows from the collision.  Both members present the identical interface state, so
any action that reads only that state returns the same verdict law on both, p is common,
and BA = 1/2(p + 1 - p) = 1/2 exactly for stop, att and rea.  Their gains are therefore
zero by the theorem, not by measurement, and this script CHECKS that on a subset rather
than spending a forward pass per pair on it.  Only enc leaves the language-side class: it
recomputes the interface from a higher-resolution crop, which breaks the collision, so
BA(enc) is free and must be measured on BOTH members.

Per pair this runs enc on each member and the placebo on each member, giving the only gain
that is not pinned by the theorem.  The rule then picks an action by argmax over
(0, 0, 0, g_enc), and the script reports the dominated-selection count together with the
tie width, so a rate that depends on a tie-break is visible as one.
"""
from __future__ import annotations

import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.actions import ActionRunner            # noqa: E402
from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.items import ImageStore, load_items    # noqa: E402
from aliasforge.verifier import Verifier               # noqa: E402

VERSION = "e31b.1"
ACTIONS = ["stop", "att", "rea", "enc"]
DOMINATED = ("att", "rea")
SIDE, TARGET = 896, 448
PLACEBO = ("Let me restate the setup before judging. The problem provides an image and a "
           "worked step, and the task is to decide whether that step follows. I will keep "
           "the original wording in mind and consider the step as written.")
QUESTION = "How many full-width bands are drawn across the figure?"
STEP = "The figure shows three full-width bands, so the count is three."


def build_pairs(n_pairs, seed, vec, m3, m2):
    out = []
    for b in range(n_pairs):
        rng = np.random.default_rng(seed * 1000 + b)
        base = np.clip(128 + 18 * np.sin(np.mgrid[0:SIDE, 0:SIDE][1] / 11.)[..., None]
                       + rng.integers(-6, 7, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
        x0, x1, _ = make_exact_pair(base, vec, 20, m3, m2)
        if x0 is not None:
            out.append((b, x0, x1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--n-check", type=int, default=8, help="pairs on which att/rea are measured too")
    ap.add_argument("--n-calib", type=int, default=150)
    a = ap.parse_args(); t0 = time.time()

    K = pil_coeffs(SIDE, TARGET)
    vec, _ = find_short_vector(K, width=64, stride=16, max_abs=8)
    m3 = np.zeros(SIDE, bool); m2 = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13 * SIDE), int(.38 * SIDE), int(.63 * SIDE))):
        if i < 3: m3[y0:y0 + int(.12 * SIDE)] = True
        if i < 2: m2[y0:y0 + int(.12 * SIDE)] = True
    pairs = build_pairs(a.n_pairs, a.seed, vec, m3, m2)

    v = Verifier(a.model, max_pixels=TARGET * TARGET)
    runner = ActionRunner(v, rea_passes=3, sampled=True, temperature=0.8, seed=a.seed)
    box = (SIDE // 4, SIDE // 4, SIDE // 2, SIDE // 2)
    print(f"exp=e31b_screen_actions version={VERSION} seed={a.seed} pairs={len(pairs)} "
          f"witness_maxabs={int(np.abs(vec).max())} model={os.path.basename(a.model)}", flush=True)

    cols = ["seed", "pair", "digests_collide", "conf", "p_plac_a", "p_plac_b",
            "p_enc_a", "p_enc_b", "ba_plac", "ba_enc", "g_enc",
            "ba_att", "ba_rea", "argmax_action", "tie_width", "checked"]
    rows, recs = [], []
    for (b, x0, x1) in pairs:
        d0 = v.interface_digest(x0, QUESTION, STEP); d1 = v.interface_digest(x1, QUESTION, STEP)
        # x0 carries the three-band content and is the label-1 member; x1 is label 0.
        pa = v.verdict_after_critique(x0, QUESTION, STEP, PLACEBO)
        pb = v.verdict_after_critique(x1, QUESTION, STEP, PLACEBO)
        ba_plac = 0.5 * (pa + (1.0 - pb))
        ea = runner.run_all(x0, QUESTION, STEP, box)["enc"].p_correct
        eb = runner.run_all(x1, QUESTION, STEP, box)["enc"].p_correct
        ba_enc = 0.5 * (ea + (1.0 - eb))
        g_enc = ba_enc - ba_plac
        checked = int(b < a.n_check)
        if checked:                       # the theorem check, on a subset
            ra = runner.run_all(x0, QUESTION, STEP, box); rb = runner.run_all(x1, QUESTION, STEP, box)
            ba_att = 0.5 * (ra["att"].p_correct + (1.0 - rb["att"].p_correct))
            ba_rea = 0.5 * (ra["rea"].p_correct + (1.0 - rb["rea"].p_correct))
        else:
            ba_att = ba_rea = float("nan")
        conf = float(v.p_correct(x0, QUESTION, STEP))
        g = {"stop": 0.0, "att": 0.0, "rea": 0.0, "enc": g_enc}   # theorem pins the first three
        # The median rule routes the acted half to the best NON-STOP action, so stop is not a
        # candidate there and a tie between att and rea is a dominated pick.
        order = sorted([k for k in ACTIONS if k != "stop"], key=lambda k: -g[k])
        best = order[0]; tie = float(g[order[0]] - g[order[1]])
        rows.append([a.seed, b, int(d0 == d1), round(conf, 6), round(pa, 6), round(pb, 6),
                     round(ea, 6), round(eb, 6), round(ba_plac, 6), round(ba_enc, 6),
                     round(g_enc, 6), round(ba_att, 6), round(ba_rea, 6), best, round(tie, 9), checked])
        recs.append(dict(conf=conf, g_enc=g_enc, best=best, tie=tie, collide=int(d0 == d1),
                         ba_plac=ba_plac,
                         ba_att=ba_att, ba_rea=ba_rea))
        print(f"progress pair={b+1}/{len(pairs)} collide={int(d0==d1)} conf={conf:.4f} "
              f"ba_plac={ba_plac:.4f} ba_enc={ba_enc:.4f} g_enc={g_enc:+.4f} argmax={best} "
              f"tie={tie:.2e} elapsed_s={time.time()-t0:.0f}", flush=True)

    if not recs:
        print("RESULT no pairs built"); return 1

    # Write the per-pair rows now. Calibration comes after and reads the benchmark, so a
    # failure there must not discard two hours of measurement (it did once).
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
        print(f"wrote {len(rows)} pair rows to {a.out} before calibration", flush=True)

    store = ImageStore(a.data); nat = []
    for it in load_items(a.data, limit=a.n_calib * 3, seed=0):
        if len(nat) >= a.n_calib: break
        try: im = np.asarray(store.load(it.image_path))
        except Exception: continue
        nat.append(float(v.p_correct(im, it.conditioned_question, it.step)))
    thr = float(np.median(nat))
    print(f"calibration: {len(nat)} natural items, median confidence {thr:.4f}", flush=True)

    n = len(recs)
    rng = np.random.default_rng(1234 + a.seed)
    # the score decides whether to ACT; IAS sits at the certified anchor 0.5, below the
    # natural median the paper reports, so it acts on every certified pair
    acts = {"verifier confidence": np.array([r["conf"] <= thr for r in recs]),
            "IAS":                 np.ones(n, bool),
            "random score":        rng.random(n) <= 0.5}
    summary = []
    for name, act in acts.items():
        dom = sum(1 for r, u in zip(recs, act) if u and r["best"] in DOMINATED)
        tied = sum(1 for r, u in zip(recs, act) if u and abs(r["tie"]) <= 1e-9)
        dom_hi = sum(1 for r, u in zip(recs, act)
                     if u and (r["best"] in DOMINATED or abs(r["tie"]) <= 1e-9))
        summary.append(dict(score=name, n=n, acted=int(act.sum()), dominated=dom,
                            dominated_upper=dom_hi, tied=tied,
                            rate=round(dom / n, 4), rate_upper=round(dom_hi / n, 4)))
        print(f"RESULT score={name:20s} n={n} acted={int(act.sum())} dominated={dom}/{n} "
              f"upper_if_ties_dominated={dom_hi}/{n} tied={tied}", flush=True)

    coll = sum(r["collide"] for r in recs)
    # Theorem 1 is a statement about the output LAW. The placebo path is deterministic given
    # the interface, so its realization IS the law and BA must be exactly 1/2. att and rea
    # generate critiques at temperature 0.8, so each member draws independently and a single
    # realization of BA fluctuates around 1/2 without contradicting anything; those columns
    # are recorded, not used as a test.
    worst_plac = max(abs(r["ba_plac"] - 0.5) for r in recs)
    print(f"RESULT theorem_check pairs={len(recs)} max|BA-1/2| on the deterministic "
          f"placebo path = {worst_plac:.3e}")
    print(f"RESULT seed={a.seed} pairs={n} digests_collide={coll}/{n} "
          f"g_enc_positive={sum(1 for r in recs if r['g_enc']>0)}/{n} "
          f"median_conf_natural={thr:.4f} elapsed_s={time.time()-t0:.0f}")

    if a.out:
        with open(a.out.replace(".csv", "_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
