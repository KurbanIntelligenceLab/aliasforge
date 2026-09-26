#!/usr/bin/env python3
"""E24: how many benchmark images are eligible at their NATIVE size?

E16-E23 resized every image to 896x896 before constructing, which forces the deployed 2x
regime. That is a modification of the input, and the paper must say whether it is needed.
Under the routing budget the processor's own policy scales by beta = sqrt(hw / max_pixels),
which lands on a dyadic ratio only for particular native areas. This reads each image's
native size (header only), applies the policy, and reports the ratio and whether bicubic's
criterion admits it. CPU, no weights, seconds.
"""
from __future__ import annotations
import argparse, csv, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.items import ImageStore, load_items   # noqa: E402
from aliasforge.qwen import smart_resize               # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--limit", type=int, default=200); ap.add_argument("--max-pixels", type=int, default=448 * 448)
    a = ap.parse_args()
    from PIL import Image
    store = ImageStore(a.data)
    rows = [("key", "source", "native_h", "native_w", "out_h", "out_w", "ratio_h", "ratio_w", "isotropic", "dyadic", "identity")]
    n = elig = ident = 0
    for it in load_items(a.data, limit=a.limit, seed=0):
        try:
            w, h = Image.fromarray(__import__("numpy").asarray(store.load(it.image_path))).size
        except Exception:
            continue
        hb, wb = smart_resize(h, w, max_pixels=a.max_pixels)
        rh, rw = h / hb, w / wb
        iso = abs(rh - rw) < 1e-9
        dy = iso and abs(math.log2(rh) - round(math.log2(rh))) < 1e-9 and rh > 1
        idn = (hb, wb) == (h, w)
        rows.append((it.key, it.source, h, w, hb, wb, round(rh, 4), round(rw, 4), int(iso), int(dy), int(idn)))
        n += 1; elig += dy; ident += idn
    if a.out:
        with open(a.out, "w", newline="") as fh: csv.writer(fh).writerows(rows)
    print(f"RESULT items={n} max_pixels={a.max_pixels} native_dyadic_eligible={elig} identity_resize={ident}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
