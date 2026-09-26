#!/usr/bin/env python3
"""E1: measure Delta_i(a) for all four actions on the powered item stream.

The powered run behind E1's pre-registered gate.  Sharded because one GPU needs
roughly eight hours for the full stream at ~22 s per item, which is what the
critique generation costs once `rea` is a real procedure rather than a re-read.

Each shard measures a disjoint slice and writes PER-ITEM gains.  It deliberately
does NOT compute the gate: V is a property of the whole stream, and a shard that
reported its own V would invite reading a partial answer as the result.  The gate
is computed once, over the merged shards, by `e1_gate.py`.

Each item is measured TWICE with independent sampling seeds.  That is not
redundancy: the split-sample estimator needs two independent replicates, because
the plug-in V is upward biased whenever the per-item gains carry measurement
noise -- and with generated critiques they do (checks C9/C9b).  Measured on the
pilot: plug-in 0.1026 against split-sample 0.0792, the gap being the bias.

Clustering is by RECORD, not by item: several steps come from one image and one
question, so their gains are correlated and item-level resampling would be
anti-conservative.
"""
from __future__ import annotations
import argparse, csv, os, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from aliasforge.actions import ActionRunner, gains            # noqa: E402
from aliasforge.items import ImageStore, load_items, summarize  # noqa: E402
from aliasforge.metrics import routing_value, routing_value_crossfit  # noqa: E402

ACTIONS = ["stop", "att", "rea", "enc"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--n-items", type=int, default=1280)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    a = ap.parse_args()

    from aliasforge.verifier import Verifier

    all_items = load_items(a.data, limit=a.n_items, seed=0)   # seed 0: one fixed stream
    items = all_items[a.shard::a.n_shards]                    # disjoint slice
    store = ImageStore(a.data)
    print(f"exp=e1_full shard={a.shard}/{a.n_shards} seed={a.seed} n_items={len(items)} of {len(all_items)} max_pixels={a.max_pixels}")
    print("items " + " ".join(f"{k}={v}" for k, v in summarize(items).items() if k != "top_sources"))

    v = Verifier(a.model, max_pixels=a.max_pixels)
    runner = ActionRunner(v, rea_passes=3, sampled=True,
                          temperature=0.8, seed=a.seed)
    rows, G = [], {r: [] for r in (0, 1)}
    kept = []

    t_start = time.time()
    for n, it in enumerate(items):
        try:
            img = np.asarray(store.load(it.image_path))
        except Exception as exc:                      # noqa: BLE001
            print(f"item={it.key} SKIP image_load {type(exc).__name__}")
            continue
        H, W = img.shape[:2]
        box = (H // 4, W // 4, H // 2, W // 2)        # deployable centre crop, no oracle region
        per_rep = {}
        for rep in (0, 1):                            # two replicates -> split-sample V
            t0 = time.time()
            res = runner.run_all(img, it.conditioned_question, it.step, box)
            g = gains(res, it.label)
            per_rep[rep] = [g[k] for k in ACTIONS]
            rows.append(dict(seed=a.seed, shard=a.shard, key=it.key, record=it.key.split(":")[0],
                             source=it.source, label=it.label,
                             rep=rep, wall_s=round(time.time() - t0, 2),
                             **{f"p_{k}": round(res[k].p_correct, 6) for k in ACTIONS},
                             **{f"g_{k}": round(g[k], 6) for k in ACTIONS},
                             rea_spread=round(res["rea"].extra["spread"], 6),
                             decode_tokens=sum(res[k].decode_tokens for k in ACTIONS),
                             g_ctrl_same=round(g["ctrl_same_interface"], 8),
                             digest_stable=int(res["ctrl_same_interface"].extra["digest_stable"])))
        for rep in (0, 1):
            G[rep].append(per_rep[rep])
        kept.append(it)
        if (n + 1) % 10 == 0:
            # cumulative + per-item, NOT the last replicate's time: reporting the
            # latter made a ~100 s/item job look like ~22 s/item and produced a
            # 4x-optimistic estimate of the whole run.
            el = time.time() - t_start
            done = max(1, len(kept))
            print(f"progress items={done}/{len(items)} elapsed_s={el:.0f} "
                  f"per_item_s={el/done:.1f} eta_s={el/done*(len(items)-done):.0f}")

    if not kept:
        print("RESULT no items measured"); return

    A, B = np.array(G[0]), np.array(G[1])
    best_fixed = int(np.argmax(A.mean(axis=0)))
    ctrl = np.array([r["g_ctrl_same"] for r in rows])

    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    ident = int(np.array_equal(A, B))
    print(f"replicates_identical={ident} (must be 0: the split-sample estimator needs independence)")
    print("mean gain per action: " + " ".join(f"{k}={A.mean(axis=0)[i]:+.4f}" for i, k in enumerate(ACTIONS)))
    print(f"best_fixed_action={ACTIONS[best_fixed]}")
    # Theorem 1 forces this control to zero; a non-zero value is an apparatus fault.
    print(f"ctrl_same_interface max|gain|={np.abs(ctrl).max():.3e} "
          f"digest_stable_all={int(all(r['digest_stable'] for r in rows))}")
    print(f"RESULT shard={a.shard} items={len(kept)} records={len({r['record'] for r in rows})} "
          f"(per-item gains only; the gate is computed over merged shards by e1_gate.py)")

if __name__ == "__main__":
    main()
