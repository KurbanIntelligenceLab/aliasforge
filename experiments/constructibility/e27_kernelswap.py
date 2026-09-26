#!/usr/bin/env python3
"""E27: what the Lanczos defence costs the frozen verifier, and whether it stops the collisions.

E20 established that Pillow's Lanczos kernel admits no realizable integer kernel vector at
any ratio swept, so the paper names the resampling kernel as the defence parameter.  Two
things follow that E20 does not answer:

  1. the defence is a change to the verifier's interface, so it may move accuracy.  The
     model was trained behind a bicubic resize; feeding it Lanczos-resized images is a
     distribution shift.  What does it cost, end to end, on the same 1280-item stream E21
     scores?
  2. the certified pairs that collide under bicubic must stop colliding under Lanczos on
     the REAL processor, not just in the kernel algebra.

So each task scores ONLY the `stop` action (the direct verdict, `Verifier.p_correct`)
under two verifiers that share one set of weights and differ only in the processor's
resampling kernel: bicubic (PIL int 3) and Lanczos (PIL int 1).  One model is loaded once
and the processor is swapped, because two 7B copies do not fit one 32 GB card.  Task 0
additionally rebuilds four certified pairs with E0c's construction (896 -> 448, exact
integer kernel of the bicubic operator) and records whether `interface_digest` agrees
between the members under each kernel.  Expected: yes under bicubic, no under Lanczos.

Row schema (one row per item; pair rows carry key `__pair_<i>__` and fill only the
`collide_*` / `gap_*` columns):
  key record source label p_stop_bicubic p_stop_lanczos correct_bicubic correct_lanczos
  hard_bicubic hard_lanczos wall_s collide_bicubic collide_lanczos gap_bicubic gap_lanczos
with correct(p) = p if label == 1 else 1 - p, and hard = [p > 0.5] == label.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "certification"))
from aliasforge.items import ImageStore, load_items  # noqa: E402
from aliasforge.verifier import Verifier  # noqa: E402

VERSION = "e27.1"
KERNELS = (("bicubic", 3), ("lanczos", 1))   # PIL resampling ints
N_PAIRS = 4
COLS = ["key", "record", "source", "label", "p_stop_bicubic", "p_stop_lanczos",
        "correct_bicubic", "correct_lanczos", "hard_bicubic", "hard_lanczos", "wall_s",
        "collide_bicubic", "collide_lanczos", "gap_bicubic", "gap_lanczos"]


def swap_kernel(v: Verifier, resample: int) -> Verifier:
    """A second verifier over the SAME weights whose processor uses another kernel.

    A shallow copy shares `model`, `yes_ids`, `no_ids` and the device; only the processor
    is replaced, so the two verifiers differ in exactly one thing.
    """
    from transformers import AutoProcessor

    w = copy.copy(v)
    kw = {"trust_remote_code": True, "resample": resample}
    if v.max_pixels is not None:
        kw["max_pixels"] = v.max_pixels
    if v.backend is not None:
        kw["backend"] = v.backend
    w.processor = AutoProcessor.from_pretrained(v.model_dir, **kw)
    w.resample = resample
    print(w.describe_image_processor())
    return w


def build_pairs(n_pairs: int = N_PAIRS, seed: int = 0):
    """E0c's certified pairs, verbatim construction: (a, b, meta) per trial."""
    from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs
    from e0c_exact import SIDE, TARGET, build_base, row_masks

    K = pil_coeffs(SIDE, TARGET)
    vec, meta = find_short_vector(K, width=64, stride=16, max_abs=8)
    if vec is None:
        return []
    out = []
    for trial in range(n_pairs):
        rng = np.random.default_rng(seed * 100 + trial)
        base = build_base(SIDE, SIDE, rng)
        ma, mb = row_masks(SIDE)
        a, b, mm = make_exact_pair(base, vec, 20, ma, mb)
        if a is None or mm.get("pix_diff", 0) == 0:
            continue
        out.append((a, b, mm))
    return out


def pair_rows(digests: dict, probs: dict | None, pairs) -> list[list]:
    """One row per pair.  `digests[kernel](img) -> str`; `probs[kernel](img) -> float` or None."""
    rows = []
    for i, (a, b, mm) in enumerate(pairs):
        t1 = time.time()
        col, gap = {}, {}
        for name, _ in KERNELS:
            col[name] = int(digests[name](a) == digests[name](b))
            gap[name] = (round(abs(probs[name](a) - probs[name](b)), 6)
                         if probs is not None else "")
        rows.append([f"__pair_{i}__", "__pair__", "e0c", "", "", "", "", "", "", "",
                     round(time.time() - t1, 2),
                     col["bicubic"], col["lanczos"], gap["bicubic"], gap["lanczos"]])
        print(f"pair={i} pix_diff={mm['pix_diff']} collide_bicubic={col['bicubic']} "
              f"collide_lanczos={col['lanczos']} gap_bicubic={gap['bicubic']} "
              f"gap_lanczos={gap['lanczos']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=1280)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--backend", default=None, help="pil | torchvision; default leaves the library's choice")
    ap.add_argument("--progress-every", type=int, default=40)
    a = ap.parse_args()

    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0))
             if i % a.shards == a.task]
    store = ImageStore(a.data)
    print(f"exp=e27_kernelswap version={VERSION} task={a.task}/{a.shards} items={len(items)} "
          f"max_pixels={a.max_pixels} backend={a.backend or 'default'} "
          f"kernels=bicubic:3,lanczos:1 action=stop model={os.path.basename(a.model)}")

    v = {"bicubic": Verifier(a.model, max_pixels=a.max_pixels, resample=3, backend=a.backend)}
    v["lanczos"] = swap_kernel(v["bicubic"], 1)

    rows = [COLS]
    acc = {f"{m}_{k}": [] for m in ("correct", "hard") for k, _ in KERNELS}
    flips = 0
    t0 = time.time()
    n_ok = 0
    for it in items:
        try:
            img = np.asarray(store.load(it.image_path))
        except Exception:
            continue
        correct = (lambda p: p) if it.label == 1 else (lambda p: 1.0 - p)
        t1 = time.time()
        p = {k: v[k].p_correct(img, it.conditioned_question, it.step) for k, _ in KERNELS}
        c = {k: correct(p[k]) for k in p}
        h = {k: int((p[k] > 0.5) == (it.label == 1)) for k in p}
        rows.append([it.key, it.key.split(":")[0], it.source, it.label,
                     round(p["bicubic"], 6), round(p["lanczos"], 6),
                     round(c["bicubic"], 6), round(c["lanczos"], 6),
                     h["bicubic"], h["lanczos"], round(time.time() - t1, 2), "", "", "", ""])
        for k in p:
            acc[f"correct_{k}"].append(c[k])
            acc[f"hard_{k}"].append(h[k])
        flips += int(h["bicubic"] != h["lanczos"])
        n_ok += 1
        if n_ok % a.progress_every == 0:
            el = time.time() - t0
            print(f"progress items={n_ok}/{len(items)} elapsed_s={el:.0f} per_item_s={el/n_ok:.1f} "
                  f"eta_s={el/n_ok*(len(items)-n_ok):.0f} "
                  f"correct_bicubic={np.mean(acc['correct_bicubic']):.4f} "
                  f"correct_lanczos={np.mean(acc['correct_lanczos']):.4f} flips={flips}")

    n_col = {}
    if a.task == 0:
        from e0c_exact import QUESTION, STEP
        pairs = build_pairs(N_PAIRS, seed=0)
        digests = {k: (lambda img, k=k: v[k].interface_digest(img, QUESTION, STEP)) for k, _ in KERNELS}
        probs = {k: (lambda img, k=k: v[k].p_correct(img, QUESTION, STEP)) for k, _ in KERNELS}
        prs = pair_rows(digests, probs, pairs)
        rows.extend(prs)
        n_col = {k: sum(r[COLS.index(f"collide_{k}")] for r in prs) for k, _ in KERNELS}
        n_col["n"] = len(prs)

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    m = {k: float(np.mean(x)) if x else float("nan") for k, x in acc.items()}
    print(f"RESULT task={a.task} items={n_ok} "
          f"mean_correct_bicubic={m['correct_bicubic']:.4f} mean_correct_lanczos={m['correct_lanczos']:.4f} "
          f"delta_lanczos_minus_bicubic={m['correct_lanczos']-m['correct_bicubic']:+.4f} "
          f"hard_acc_bicubic={m['hard_bicubic']:.4f} hard_acc_lanczos={m['hard_lanczos']:.4f} "
          f"flips={flips}"
          + (f" pairs_collide_bicubic={n_col['bicubic']}/{n_col['n']} "
             f"pairs_collide_lanczos={n_col['lanczos']}/{n_col['n']}" if n_col else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
