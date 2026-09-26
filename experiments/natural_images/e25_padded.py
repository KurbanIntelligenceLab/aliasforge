#!/usr/bin/env python3
"""E25: E23 with the image PADDED onto the 896 canvas instead of stretched.

Every original pixel is untouched when the image fits; when it does not, it is scaled with
aspect preserved and then padded. Marks are confined to the image region so the border can
never carry them. Reports which case each item hit.

Original E23 docstring follows.

E18 chose mark rows by intensity headroom alone. On line-art that search finds its headroom
on existing dark lines, and the carrier then modulates an edge that is already there: the
pair is certified, the pixels differ by 60/255, and no reader would count three marks
against two. So E18's yields are of certified pairs, not of legible ones, and the paper
must not conflate them.

This adds a legibility constraint to the same construction: a row may carry the mark only
if the carrier's neighbourhood is FLAT (low variance across the carrier columns and a margin
either side) and MID-TONE (room for a +-60 dash to read as a dash in either direction).
The yield of certified-and-legible pairs is the honest number; the assets of the most
legible pair are saved for the figure.

Preprocessing only: no GPU, no weights.
"""
from __future__ import annotations
import argparse, csv, hashlib, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from aliasforge.items import ImageStore, load_items             # noqa: E402
from e18_freeplace import carriers, row_feasible, pick_marks     # noqa: E402

VERSION = "e25.1"


def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def row_legible(base, v, margin=6, max_std=10.0, lo=40, hi=215):
    """Rows where a mark on this carrier would read as a distinct dash on a flat field."""
    nz = np.flatnonzero(v); c0, c1 = max(0, nz.min() - margin), min(base.shape[1], nz.max() + margin + 1)
    win = base[:, c0:c1].astype(np.float64)                      # (H, w, 3)
    std = win.std(axis=(1, 2)); mean = win.mean(axis=(1, 2))
    return (std < max_std) & (mean > lo) & (mean < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--side", type=int, default=896); ap.add_argument("--target", type=int, default=448)
    ap.add_argument("--amp", type=int, default=20); ap.add_argument("--need", type=float, default=0.90)
    ap.add_argument("--only-key", default=""); ap.add_argument("--pad-color", default="128",
                    help="integer grey, or 'auto' = median of the image border")
    a = ap.parse_args()
    from PIL import Image
    items = [it for i, it in enumerate(load_items(a.data, limit=a.limit, seed=0)) if i % a.shards == a.task]
    if a.only_key: items = [it for it in load_items(a.data, limit=a.limit, seed=0) if it.key == a.only_key]
    store = ImageStore(a.data)
    cands = carriers(a.side, a.target); signed = [(v, s) for v in cands for s in (1, -1)]
    T = int(0.12 * a.side)
    print(f"exp=e25_padded version={VERSION} task={a.task}/{a.shards} items={len(items)} amp={a.amp} need={a.need}")
    rows = [("key", "source", "fit", "native_w", "native_h", "pad_color", "certified_any", "certified_legible", "rows_kept", "local_contrast", "pix_diff")]
    n_any = n_leg = 0; best = None
    for it in items:
        try:
            im = Image.fromarray(np.asarray(store.load(it.image_path))).convert("RGB")
        except Exception:
            continue
        w, h = im.size; fit = "native" if max(w, h) <= a.side else "scaled"
        if fit == "scaled":
            sc = a.side / max(w, h); im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BICUBIC); w, h = im.size
        if a.pad_color == "auto":
            e = np.asarray(im); border = np.concatenate([e[0], e[-1], e[:, 0], e[:, -1]]); pc = tuple(int(x) for x in np.median(border, axis=0))
        else:
            pc = (int(a.pad_color),) * 3
        canvas = Image.new("RGB", (a.side, a.side), pc); ox, oy = (a.side - w) // 2, (a.side - h) // 2
        canvas.paste(im, (ox, oy)); base = np.asarray(canvas)
        inside_rows = np.zeros(a.side, bool); inside_rows[oy:oy + h] = True
        def in_cols(v):
            nz = np.flatnonzero(v); return bool(nz.min() >= ox + 6 and nz.max() < ox + w - 6)
        feas_any = np.stack([row_feasible(base, v, a.amp, s) & inside_rows & in_cols(v) for v, s in signed])
        leg = np.stack([row_legible(base, v) for v, _ in signed])
        feas_leg = feas_any & leg
        def aligned(feas):
            bst = None
            for ci in range(len(signed)):
                sel = pick_marks(feas[ci:ci + 1], T, 3, a.need, T // 2)
                if sel is None: continue
                cov = float(np.mean([c[2] for c in sel]))
                if bst is None or cov > bst[0]: bst = (cov, ci, sel)
            return bst
        any_ok = aligned(feas_any) is not None
        r = aligned(feas_leg)
        cert_leg, cov, contrast, pix = 0, 0.0, 0.0, 0
        if r is not None:
            cov, ci, sel = r; v, s = signed[ci]; add = (s * a.amp * v)[:, None]
            xa = base.astype(np.int64).copy(); xb = base.astype(np.int64).copy()
            for i, (_, y, _) in enumerate(sel):
                rr = np.flatnonzero(feas_leg[ci][y:y + T]) + y
                xa[rr] += add
                if i < 2: xb[rr] += add
            if min(xa.min(), xb.min()) >= 0 and max(xa.max(), xb.max()) <= 255:
                xa, xb = xa.astype(np.uint8), xb.astype(np.uint8)
                za = np.asarray(Image.fromarray(xa).resize((a.target, a.target), Image.BICUBIC))
                zb = np.asarray(Image.fromarray(xb).resize((a.target, a.target), Image.BICUBIC))
                if digest(za) == digest(zb):
                    cert_leg = 1; pix = int((xa != xb).any(axis=2).sum())
                    y3 = sel[2][1]; nz = np.flatnonzero(v)
                    contrast = float(np.abs(xa[y3:y3 + T, nz].astype(int) - xb[y3:y3 + T, nz].astype(int)).max())
                    if best is None or (cov, contrast) > (best[0], best[1]):
                        best = (cov, contrast, it.key, xa, xb, za, sel, nz)
        n_any += any_ok; n_leg += cert_leg
        rows.append((it.key, it.source, fit, w, h, str(pc), int(any_ok), cert_leg, round(cov, 4), contrast, pix))
    if a.out:
        with open(a.out, "w", newline="") as fh: csv.writer(fh).writerows(rows)
        if best is not None:
            cov, contrast, key, xa, xb, za, sel, nz = best
            d = os.path.dirname(a.out) or "."
            Image.fromarray(xa).save(os.path.join(d, f"best_t{a.task}_x0_three.png"))
            Image.fromarray(xb).save(os.path.join(d, f"best_t{a.task}_x1_two.png"))
            Image.fromarray(za).save(os.path.join(d, f"best_t{a.task}_interface448.png"))
            # zoomed crop of the third mark, both members side by side
            y = sel[2][1]; c = int(nz.mean()); box = (max(0, c - 40), max(0, y - 20), min(896, c + 40), min(896, y + T + 20))
            A = Image.fromarray(xa).crop(box).resize(((box[2]-box[0])*4, (box[3]-box[1])*4), Image.NEAREST)
            B = Image.fromarray(xb).crop(box).resize(A.size, Image.NEAREST)
            canvas = Image.new("RGB", (A.width * 2 + 24, A.height), "white"); canvas.paste(A, (0, 0)); canvas.paste(B, (A.width + 24, 0))
            canvas.save(os.path.join(d, f"best_t{a.task}_zoom.png"))
            with open(os.path.join(d, f"best_t{a.task}.txt"), "w") as fh:
                fh.write(f"key={key} rows_kept={cov:.4f} contrast={contrast:.0f} cols={nz.tolist()} marks={[(s[1], s[1]+T) for s in sel]}\n")
    print(f"RESULT task={a.task} items={len(items)} certified_any={n_any} certified_legible={n_leg} "
          f"best={(best[2], round(best[0],3), best[1]) if best else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
