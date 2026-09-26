#!/usr/bin/env python3
"""E19: is "every action hurts" a property of the actions, or of the path they take?

E1's headline is that all four actions have negative mean gain and the best fixed action is
`stop`.  Every gain is g(a) = p_a - p_stop.  But `stop` and the language-side actions do not
take the same route through the model:

    stop        p_correct(image, question, step)
    att / rea   critique(...) -> verdict_after_critique(...), which is p_correct on
                f"{question}\\n\\nCritique of the step:\\n{critique_text}"

So g(a) mixes the action's INFORMATIONAL effect with a fixed offset between two different
prompts.  E5 already hints that the offset dominates: with a null instruction the critique
path still costs -0.056, while the entire spread across five instructions is 0.017.  If that
offset is the bulk of the effect, then "no language-side repair helps" is partly a statement
about prompt formatting, not about re-attention or extra reasoning.

A placebo ladder separates the three effects, each step adding exactly one thing:

    stop         direct verdict.                              (E1's baseline)
    header       verdict through the critique template with an EMPTY critique.
                 Isolates the cost of the template itself.
    placebo      verdict through the template with a fluent critique that makes NO visual
                 claim and no judgement.  Isolates the cost of "there is text here".
    att / rea    the real actions.

g(att) against `placebo` is the informational effect of re-attention with the path held
fixed; g(att) against `stop` is what E1 reported.  The difference between them is the
confound, measured rather than argued.

This does not touch Theorem 1, whose zero-gain claim is exact and path-independent, nor V,
which depends on how the best action VARIES by item rather than on any common offset.
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

VERSION = "e19.1"

# Fluent, on-register, and deliberately empty of visual or evaluative content.
PLACEBO = ("Let me restate the setup before judging. The problem provides an image and a "
           "worked step, and the task is to decide whether that step follows. I will keep "
           "the original wording in mind and consider the step as written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--limit", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    a = ap.parse_args()

    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0))
             if i % a.shards == a.task]
    store = ImageStore(a.data)
    v = Verifier(a.model, max_pixels=a.max_pixels)
    runner = ActionRunner(v, rea_passes=3, sampled=True, temperature=0.8, seed=a.task)

    print(f"exp=e19_placebo version={VERSION} task={a.task}/{a.shards} items={len(items)} "
          f"reps={a.reps} ladder=stop,header,placebo,att,rea model={os.path.basename(a.model)}")

    cols = ("key", "record", "label", "rep",
            "p_stop", "p_header", "p_placebo", "p_att", "p_rea")
    rows = [cols]
    acc = {k: [] for k in ("stop", "header", "placebo", "att", "rea")}
    t0 = time.time()
    for n, it in enumerate(items):
        try:
            img = np.asarray(store.load(it.image_path))
        except Exception:
            continue
        q, st = it.conditioned_question, it.step
        for rep in range(a.reps):
            p_stop = v.p_correct(img, q, st)
            p_header = v.verdict_after_critique(img, q, st, "")
            p_placebo = v.verdict_after_critique(img, q, st, PLACEBO)
            p_att = runner.att(img, q, st).p_correct
            p_rea = runner.rea(img, q, st).p_correct
            rows.append((it.key, it.key.split(":")[0], it.label, rep,
                         round(p_stop, 6), round(p_header, 6), round(p_placebo, 6),
                         round(p_att, 6), round(p_rea, 6)))
            for k, p in (("stop", p_stop), ("header", p_header), ("placebo", p_placebo),
                         ("att", p_att), ("rea", p_rea)):
                acc[k].append(p)
        if (n + 1) % 20 == 0:
            el = time.time() - t0
            m = {k: float(np.mean(x)) for k, x in acc.items()}
            print(f"  progress {n+1}/{len(items)} elapsed_s={el:.0f} per_item_s={el/(n+1):.1f} "
                  f"g_att_vs_stop={m['att']-m['stop']:+.4f} "
                  f"g_att_vs_placebo={m['att']-m['placebo']:+.4f} "
                  f"path_offset={m['placebo']-m['stop']:+.4f}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

    m = {k: float(np.mean(x)) if x else float("nan") for k, x in acc.items()}
    print("RESULT task=%d n=%d " % (a.task, len(items)) +
          " ".join(f"mean_p_{k}={m[k]:.4f}" for k in ("stop", "header", "placebo", "att", "rea")) +
          f" header_offset={m['header']-m['stop']:+.4f}"
          f" placebo_offset={m['placebo']-m['stop']:+.4f}"
          f" g_att_vs_stop={m['att']-m['stop']:+.4f}"
          f" g_att_vs_placebo={m['att']-m['placebo']:+.4f}"
          f" g_rea_vs_stop={m['rea']-m['stop']:+.4f}"
          f" g_rea_vs_placebo={m['rea']-m['placebo']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
