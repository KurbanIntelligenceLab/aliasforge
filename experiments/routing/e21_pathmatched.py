#!/usr/bin/env python3
"""E21: E1 re-measured against a PATH-MATCHED baseline.

E1's gains are g(a) = correct(p_a) - correct(p_stop), where `stop` is a direct verdict and
every action is a verdict AFTER a generated critique.  E19 measured what that path costs
with no content in it: -0.152 for a fluent placebo, -0.096 for the bare template.  Against
that baseline att and rea are indistinguishable from the placebo (CI spans zero).  So E1's
"every action hurts, best fixed action is stop" is mostly a statement about prompt format.

This does not touch Theorem 1, which is exact and path-independent, and it does not touch
the fibre argument.  It touches three reported numbers: the per-action means, the identity
of the best fixed action, and V -- which cannot be corrected by hand, because
V = oracle - best_fixed and a common shift moves both terms.  So we re-measure.

Same items, same replicates, same actions, same runner configuration as E1, plus one rung:
a content-free placebo critique through the same path.  Two gain sets are emitted per row:

    g_*    against the PLACEBO   -- path held fixed; the headline
    gd_*   against DIRECT stop   -- E1's convention, for reconciliation

Columns match E1's, so e1_gate.py runs unchanged on `g_*` and reports the path-matched V.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.actions import ActionRunner  # noqa: E402
from aliasforge.items import ImageStore, load_items  # noqa: E402
from aliasforge.verifier import Verifier  # noqa: E402

VERSION = "e21.1"
ACTIONS = ["stop", "att", "rea", "enc"]
PLACEBO = ("Let me restate the setup before judging. The problem provides an image and a "
           "worked step, and the task is to decide whether that step follows. I will keep "
           "the original wording in mind and consider the step as written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--limit", type=int, default=1280)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    a = ap.parse_args()

    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0))
             if i % a.shards == a.task]
    store = ImageStore(a.data)
    v = Verifier(a.model, max_pixels=a.max_pixels)
    runner = ActionRunner(v, rea_passes=3, sampled=True, temperature=0.8, seed=a.task)
    print(f"exp=e21_pathmatched version={VERSION} task={a.task}/{a.shards} items={len(items)} "
          f"baseline=placebo model={os.path.basename(a.model)}")

    cols = (["seed", "shard", "key", "record", "source", "label", "rep", "wall_s",
             "p_stop_direct", "p_placebo"]
            + [f"p_{k}" for k in ACTIONS] + [f"g_{k}" for k in ACTIONS]
            + [f"gd_{k}" for k in ACTIONS] + ["rea_spread"])
    rows = [cols]
    acc = {k: [] for k in ("placebo_off",) + tuple(f"g_{k}" for k in ACTIONS)}
    t0 = time.time()
    # Resume: a walltime kill keeps the rows already written (the shard is rewritten every
    # few items below), so a `fix` continues from the last checkpoint instead of restarting.
    done_keys = set()
    if a.out and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, newline="") as fh:
            old = list(csv.reader(fh))
        if old and old[0] == list(cols):
            rows = old
            done_keys = {(r[2], r[6]) for r in old[1:]}
            print(f"resume: {len(old) - 1} rows already in shard ({len(done_keys) // 2} items done)")

    def checkpoint():
        if a.out:
            tmp = a.out + ".tmp"
            with open(tmp, "w", newline="") as fh:
                csv.writer(fh).writerows(rows)
            os.replace(tmp, a.out)
    n_ok = 0
    for n, it in enumerate(items):
        if (it.key, "0") in done_keys and (it.key, "1") in done_keys:
            n_ok += 1
            continue
        try:
            img = np.asarray(store.load(it.image_path))
        except Exception:
            continue
        H, W = img.shape[:2]
        box = (H // 4, W // 4, H // 2, W // 2)
        correct = (lambda p: p) if it.label == 1 else (lambda p: 1.0 - p)
        for rep in (0, 1):
            t1 = time.time()
            p_direct = v.p_correct(img, it.conditioned_question, it.step)
            p_plac = v.verdict_after_critique(img, it.conditioned_question, it.step, PLACEBO)
            res = runner.run_all(img, it.conditioned_question, it.step, box)
            p = {k: res[k].p_correct for k in ACTIONS}
            # path-matched: the "stop" rung IS the placebo, so its gain is 0 by definition
            base_p, base_d = correct(p_plac), correct(p_direct)
            g = {"stop": 0.0, **{k: correct(p[k]) - base_p for k in ACTIONS if k != "stop"}}
            gd = {k: correct(p[k]) - base_d for k in ACTIONS}
            rows.append([a.task, a.task, it.key, it.key.split(":")[0], it.source, it.label,
                         rep, round(time.time() - t1, 2), round(p_direct, 6), round(p_plac, 6)]
                        + [round(p[k], 6) for k in ACTIONS]
                        + [round(g[k], 6) for k in ACTIONS]
                        + [round(gd[k], 6) for k in ACTIONS]
                        + [round(res["rea"].extra["spread"], 6)])
            acc["placebo_off"].append(base_p - base_d)
            for k in ACTIONS:
                acc[f"g_{k}"].append(g[k])
        n_ok += 1
        if n_ok % 5 == 0:
            checkpoint()
        if n_ok % 10 == 0:
            el = time.time() - t0
            print(f"progress items={n_ok}/{len(items)} elapsed_s={el:.0f} per_item_s={el/n_ok:.1f} "
                  f"eta_s={el/n_ok*(len(items)-n_ok):.0f} placebo_off={np.mean(acc['placebo_off']):+.4f} "
                  + " ".join(f"g_{k}={np.mean(acc[f'g_{k}']):+.4f}" for k in ("att", "rea", "enc")))

    checkpoint()
    # summary over every row in the shard (resumed rows included), not only this process's
    ci = {c: i for i, c in enumerate(rows[0])}
    body = rows[1:]
    def col(name):
        return [float(r[ci[name]]) for r in body] if body else []
    lab = [int(float(r[ci["label"]])) for r in body]
    def corr(vals):
        return [v if l == 1 else 1.0 - v for v, l in zip(vals, lab)]
    m = {"placebo_off": float(np.mean(np.subtract(corr(col("p_placebo")), corr(col("p_stop_direct"))))) if body else float("nan")}
    for k in ("att", "rea", "enc"):
        m[f"g_{k}"] = float(np.mean(col(f"g_{k}"))) if body else float("nan")
    print(f"RESULT task={a.task} items={n_ok} placebo_offset={m['placebo_off']:+.4f} "
          + " ".join(f"mean_g_{k}_vs_placebo={m[f'g_{k}']:+.4f}" for k in ("att", "rea", "enc")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
