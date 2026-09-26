#!/usr/bin/env python3
"""Do certified pairs survive other resize implementations and numeric modes?

Two pairs are built at 896 -> 448 on the synthetic base of the certification experiments:
one from the bicubic kernel vector (1,-3,3,-1) and one from the box kernel vector (1,-1).
Each pair is resized by every available library mode, and the two outputs are compared
exactly.  A mode is `identical` when the outputs agree bit for bit in the mode's own output
dtype, and `identical after rounding` when a floating-point output agrees only once it is
rounded to 8 bits.  Pillow and torchvision are tested when importable, OpenCV and TensorFlow
likewise; a missing library is reported as such and never simulated.

Writes library_transfer.csv beside the other generated tables.
"""
from __future__ import annotations
import csv, pathlib, sys
import numpy as np

_here = pathlib.Path(__file__).resolve()
for _c in (_here.parents[3] / "Code" / "src" if len(_here.parents) > 3 else None, _here.parents[1] / "src"):
    if _c is not None and (_c / "aliasforge").is_dir():
        sys.path.insert(0, str(_c)); break
from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402

SIDE, TARGET, N_PAIRS, AMP = 896, 448, 5, 20


def pairs():
    K = pil_coeffs(SIDE, TARGET)
    bic, _ = find_short_vector(K, width=64, stride=16, max_abs=8)
    box = np.zeros(SIDE, dtype=np.int64)
    for c in range(64, SIDE - 64, 16):           # (1,-1) on aligned column pairs, interior only
        box[c], box[c + 1] = 1, -1
    m3 = np.zeros(SIDE, bool); m2 = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13*SIDE), int(.38*SIDE), int(.63*SIDE))):
        m3[y0:y0+int(.12*SIDE)] = True
        if i < 2: m2[y0:y0+int(.12*SIDE)] = True
    out = {"bicubic carrier": [], "box carrier": []}
    for b in range(N_PAIRS):
        rr = np.random.default_rng(b)
        base = np.clip(128 + 18 * np.sin(np.mgrid[0:SIDE, 0:SIDE][1] / 11.)[..., None]
                       + rr.integers(-6, 7, (SIDE, SIDE, 3)), 0, 255).astype(np.uint8)
        for name, vec in (("bicubic carrier", np.asarray(bic, dtype=np.int64)), ("box carrier", box)):
            x0, x1, _ = make_exact_pair(base, vec, AMP, m3, m2)
            assert x0 is not None and (x0 != x1).any()
            out[name].append((x0, x1))
    return out


def modes():
    """(library, mode, numeric type, fn: uint8 HxWx3 -> array) for every importable library."""
    M = []
    from PIL import Image
    import PIL
    for nm, flt in (("bicubic", Image.BICUBIC), ("bilinear", Image.BILINEAR), ("box", Image.BOX), ("Lanczos", Image.LANCZOS)):
        M.append((f"Pillow {PIL.__version__}", nm, "fixed-point",
                  lambda x, flt=flt: np.asarray(Image.fromarray(x).resize((TARGET, TARGET), flt))))
    try:
        import torch, torchvision
        torch.set_num_threads(1)   # one thread: two OpenMP runtimes in one process abort on some hosts
        from torchvision.transforms.v2 import functional as F
        from torchvision.transforms import InterpolationMode as IM
        def tv(x, mode, as_float, aa):
            t = torch.from_numpy(x).permute(2, 0, 1)
            if as_float: t = t.float()
            return F.resize(t, [TARGET, TARGET], interpolation=mode, antialias=aa).numpy()
        lib = f"torchvision {torchvision.__version__}"
        for nm, mode in (("bicubic", IM.BICUBIC), ("bilinear", IM.BILINEAR)):
            M.append((lib, f"{nm}, antialias", "uint8", lambda x, mode=mode: tv(x, mode, False, True)))
            M.append((lib, f"{nm}, antialias", "float32", lambda x, mode=mode: tv(x, mode, True, True)))
            M.append((lib, f"{nm}, no antialias", "float32", lambda x, mode=mode: tv(x, mode, True, False)))
        M.append((f"torch {torch.__version__}", "area", "float32",
                  lambda x: torch.nn.functional.interpolate(torch.from_numpy(x).permute(2, 0, 1)[None].float(),
                                                            size=(TARGET, TARGET), mode="area")[0].numpy()))
    except ImportError:
        M.append(("torchvision", "not installed", "", None))
    try:
        import cv2
        for nm, flag in (("area", cv2.INTER_AREA), ("bicubic", cv2.INTER_CUBIC), ("bilinear", cv2.INTER_LINEAR), ("Lanczos", cv2.INTER_LANCZOS4)):
            M.append((f"OpenCV {cv2.__version__}", nm, "uint8", lambda x, flag=flag: cv2.resize(x, (TARGET, TARGET), interpolation=flag)))
    except ImportError:
        M.append(("OpenCV", "not installed", "", None))
    try:
        import tensorflow as tf
        for nm in ("bilinear", "bicubic", "area", "lanczos3"):
            for aa in (False, True):
                M.append((f"TensorFlow {tf.__version__}", f"{nm}, {'antialias' if aa else 'no antialias'}", "float32",
                          lambda x, nm=nm, aa=aa: tf.image.resize(x, (TARGET, TARGET), method=nm, antialias=aa).numpy()))
    except ImportError:
        M.append(("TensorFlow", "not installed", "", None))
    return M


def main():
    P = pairs(); rows = []
    for lib, mode, num, fn in modes():
        row = dict(library=lib, mode=mode, numeric=num)
        for cname, prs in P.items():
            if fn is None:
                row[cname] = "not run"; continue
            exact = rounded = 0; worst = 0.0
            for x0, x1 in prs:
                a, b = fn(x0), fn(x1)
                exact += int(np.array_equal(a, b))
                worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
                rounded += int(np.array_equal(np.clip(np.rint(a), 0, 255), np.clip(np.rint(b), 0, 255)))
            row[cname] = (f"identical {exact}/{len(prs)}" if exact == len(prs)
                          else f"identical after rounding {rounded}/{len(prs)} (max diff {worst:.2g})" if rounded == len(prs)
                          else f"differs (identical {exact}/{len(prs)}, after rounding {rounded}/{len(prs)}, max diff {worst:.3g})")
        rows.append(row)
        print(f"RESULT {lib:22s} {mode:24s} {num:12s} | bicubic carrier: {row['bicubic carrier']:58s} | box carrier: {row['box carrier']}", flush=True)
    out_dir = _here.parents[1] / "generated"
    if not out_dir.is_dir():                      # supplementary layout
        out_dir = _here.parents[1] / "analysis" / "output"; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "library_transfer.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
