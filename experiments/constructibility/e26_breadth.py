#!/usr/bin/env python3
"""E26: processor-level certification across MORE deployed pipelines.

The paper certifies two architectures end to end.  A processor-level certificate needs only
the processor, never the weights: build a pair from the integer kernel of the deployed
resize, run both members through the real processor, and compare digests over every field
it emits.  The screen table (resize_configs.txt) PREDICTS eligibility from the ratio; this
experiment tests that prediction on real preprocessing code, one processor per task.

Per processor, in order:
  1. load it (AutoProcessor, then AutoImageProcessor; trust_remote_code) and record the
     classes, the resample attribute and the resize keys of preprocessor_config.json;
  2. recover the resize POLICY empirically: square synthetic images of many sides go through
     the processor and the emitted geometry is read back (a (B,3,H,W) tensor, a tiled
     (B,N,3,h,w) tensor, a Qwen-style patch grid via image_grid_thw, a padded canvas via
     image_unpadded_heights);
  3. choose an input side whose PRIMARY resize is an exact dyadic downscale (2x preferred),
     else the nearest ratio the policy allows, and report the certified Gram-Schmidt lower
     bound on the shortest kernel vector in a 32-wide window (tools/lattice_bounds.py);
  4. build seeded pairs with the existing exact construction and certify them against the
     processor, with a random-direction control that must NOT certify, plus a check of
     whether the processor's resize reproduces Pillow's fixed-point path.

transformers 5 ships every image processor with a torchvision backend by default and a
Pillow one behind `backend="pil"`; both are measured when both load, as separate variants
(the image_processor_class column tells them apart).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from aliasforge.exact import KERNELS, PRECISION_BITS, find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.interface import per_field_report, state_digest  # noqa: E402
from aliasforge.lattice import kernel_basis  # noqa: E402

VERSION = "e26.1"
PROBE_SIDES = [224, 336, 384, 448, 512, 672, 768, 896, 980, 1024, 1120, 1344, 1456, 1536,
               1792, 1960, 2048]
RESAMPLE_NAMES = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}
QUESTION = "How many horizontal marks appear in the image?"
COLS = ["processor", "processor_class", "image_processor_class", "resample", "policy", "n_in",
        "n_out", "ratio", "dyadic", "gs_bound", "trial", "arm", "certified", "fields_differ",
        "n_fields", "pix_diff", "error"]
CONFIG_KEYS = ("size", "patch", "merge", "pixel", "resample", "crop", "tile", "split", "thumb",
               "longest", "shortest", "num_crops", "pinpoints", "pan", "scan")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _lb_module():
    here = pathlib.Path(__file__).resolve().parents[2] / "verify" / "lattice_bounds.py"
    spec = importlib.util.spec_from_file_location("lattice_bounds", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_base(H, W, rng):
    return np.clip(128 + 18 * np.sin(np.mgrid[0:H, 0:W][1] / 11.0)[..., None]
                   + rng.integers(-6, 7, (H, W, 3)), 0, 255).astype(np.uint8)


def row_masks(H, n_a=3, n_b=2):
    """The atomic fact: how many horizontal marks are present (e0a's masks)."""
    a, b = np.zeros(H, bool), np.zeros(H, bool)
    for i, y0 in enumerate((int(.13 * H), int(.38 * H), int(.63 * H))):
        if i < n_a:
            a[y0:y0 + int(.12 * H)] = True
        if i < n_b:
            b[y0:y0 + int(.12 * H)] = True
    return a, b


def probe_image(side, seed=0):
    from PIL import Image
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (side, side, 3), dtype=np.uint8))


def is_dyadic(r):
    return r > 1 and abs(math.log2(r) - round(math.log2(r))) < 1e-9


def resample_int(x):
    try:
        return int(x)
    except Exception:
        return {"NEAREST": 0, "LANCZOS": 1, "BILINEAR": 2, "BICUBIC": 3, "BOX": 4,
                "HAMMING": 5}.get(str(x).split(".")[-1].upper(), -1)


def pil_resample(code):
    from PIL import Image
    return {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR, 3: Image.BICUBIC,
            4: Image.BOX, 5: Image.HAMMING}[code]


def coeffs_for(code, n_in, n_out):
    name = RESAMPLE_NAMES.get(code)
    if name not in KERNELS:
        raise ValueError(f"resample {code} ({name}) has no exact fixed-point kernel here")
    f, supp = KERNELS[name]
    return pil_coeffs(n_in, n_out, support=supp, filt=f)


def bit_exact(code, n_in, n_out, K, rng):
    """Gate (as e20): the transcribed operator must reproduce Pillow bit-for-bit."""
    from PIL import Image
    img = rng.integers(0, 256, (n_in, 4, 3), dtype=np.uint8)
    ref = np.asarray(Image.fromarray(img).resize((4, n_out), pil_resample(code)))
    half = 1 << (PRECISION_BITS - 1)
    mine = np.empty_like(ref)
    for c in range(4):
        for ch in range(3):
            ss = K @ img[:, c, ch].astype(object) + half
            mine[:, c, ch] = np.clip([int(x) >> PRECISION_BITS for x in ss], 0, 255)
    return bool(np.array_equal(ref, mine))


def gs_bound(K, width=32, stride=16, max_windows=4):
    """Min over windows of the certified lower bound on the shortest kernel vector."""
    lb = _lb_module()
    n_in = K.shape[1]
    starts = list(range(0, max(1, n_in - width), stride))
    if len(starts) > max_windows:
        starts = [starts[int(i * (len(starts) - 1) / (max_windows - 1))] for i in range(max_windows)]
    bounds = []
    for c0 in starts:
        B = kernel_basis(K, c0, width)
        if B:
            b = lb.gs_bound_exact(B)
            if b is not None:
                bounds.append(b)
    return min(bounds) if bounds else None


# ---------------------------------------------------------------------------
# processor I/O
# ---------------------------------------------------------------------------

def _flatten(out):
    """Every emitted field as a digestable value; lists of arrays become indexed fields.

    A list of tensors would otherwise be digested through its repr, which torch truncates,
    and a truncated repr can collide when the tensors do not.
    """
    flat = {}

    def put(k, v):
        if isinstance(v, (list, tuple)) and v and all(hasattr(x, "shape") or isinstance(x, (list, tuple)) for x in v):
            for i, x in enumerate(v):
                put(f"{k}[{i}]", x)
        else:
            flat[k] = v
    for k in list(out.keys()):
        put(k, out[k])
    return flat


def call_any(proc, image, text=QUESTION):
    """Run one image through, falling back across text/no-text and np/pt return types."""
    attempts = []
    ip = getattr(proc, "image_processor", None)
    for label, fn in (
        ("images+text/np", lambda: proc(images=image, text=text, return_tensors="np")),
        ("images+text/pt", lambda: proc(images=image, text=text, return_tensors="pt")),
        ("images/np", lambda: proc(images=image, return_tensors="np")),
        ("images/pt", lambda: proc(images=image, return_tensors="pt")),
        ("image_processor/np", lambda: ip(images=image, return_tensors="np") if ip else None),
        ("image_processor/pt", lambda: ip(images=image, return_tensors="pt") if ip else None),
    ):
        try:
            out = fn()
            if out is None:
                continue
            return _flatten(out), label
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{label}: {type(exc).__name__}: {str(exc)[:80]}")
    raise RuntimeError("processor call failed on every path: " + " | ".join(attempts))


def _arr(v):
    return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)


def geometry(out, patch):
    """Read the resize geometry back from what the processor emitted, for a SQUARE input.

    Returns {n_out, secondary, tiles, mode, shape} or None.  n_out is the side of the
    PRIMARY resize (the one every emitted pixel passes through); `secondary` is a second,
    independent resize of the original implied by a thumbnail in a tiled layout.
    """
    if "image_grid_thw" in out:
        g = _arr(out["image_grid_thw"]).reshape(-1, 3)[0]
        p = patch if isinstance(patch, int) else 14
        return dict(n_out=int(g[1]) * p, secondary=0, tiles=1, mode="grid",
                    shape=str(tuple(_arr(out[next(k for k in out if "pixel" in k)]).shape)))
    if "image_unpadded_heights" in out:
        h = int(_arr(out["image_unpadded_heights"]).reshape(-1)[0])
        return dict(n_out=h, secondary=0, tiles=1, mode="pad_canvas",
                    shape=str(tuple(_arr(out["images"]).shape)) if "images" in out else "")
    key = next((k for k in ("pixel_values", "image_pixel_values", "images")
                if k in out), None)
    if key is None:
        key = next((k for k in out if k.endswith("pixel_values")), None)
    if key is None:
        return None
    a = _arr(out[key])
    if a.ndim == 4:
        return dict(n_out=int(a.shape[-2]), secondary=0, tiles=1, mode="single",
                    shape=str(tuple(a.shape)))
    if a.ndim == 5:
        n, h = int(a.shape[1]), int(a.shape[-2])
        if n == 1:
            return dict(n_out=h, secondary=0, tiles=1, mode="single", shape=str(tuple(a.shape)))
        g = int(round(math.sqrt(n - 1)))
        if g * g == n - 1:       # g x g crops of a resized image, plus one thumbnail
            return dict(n_out=g * h, secondary=h, tiles=n, mode="tiles+thumb", shape=str(tuple(a.shape)))
        g = int(round(math.sqrt(n)))
        if g * g == n:
            return dict(n_out=g * h, secondary=0, tiles=n, mode="tiles", shape=str(tuple(a.shape)))
        return dict(n_out=h, secondary=0, tiles=n, mode="tiles?", shape=str(tuple(a.shape)))
    return None


def probe_policy(proc, patch):
    entries, errs, unknown = [], [], None
    for side in PROBE_SIDES:
        try:
            out, _ = call_any(proc, probe_image(side))
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{side}!{type(exc).__name__}")
            continue
        g = geometry(out, patch)
        if g is None:
            unknown = unknown or {k: str(tuple(_arr(v).shape)) if hasattr(v, "shape") else type(v).__name__
                                  for k, v in out.items()}
            errs.append(f"{side}?")
            continue
        entries.append(dict(side=side, **g))
    return entries, errs, unknown


def policy_string(entries, errs, backend):
    if not entries:
        return f"backend={backend};unrecognized;" + ",".join(errs)
    outs = {(e["n_out"], e["secondary"], e["tiles"]) for e in entries}
    if len(outs) == 1:
        e = entries[0]
        s = f"fixed:{e['n_out']}"
        if e["tiles"] > 1:
            s += f"+thumb{e['secondary']}x{e['tiles']}" if e["secondary"] else f"x{e['tiles']}"
    else:
        s = "map:" + ",".join(
            f"{e['side']}>{e['n_out']}" + (f"+{e['secondary']}" if e["secondary"] else "")
            + (f"x{e['tiles']}" if e["tiles"] > 1 else "") for e in entries)
    mode = sorted({e["mode"] for e in entries})
    s = f"backend={backend};mode={'/'.join(mode)};{s}"
    if errs:
        s += ";err=" + ",".join(errs)
    return s


def choose_geometry(entries):
    """Prefer an exact 2x primary downscale, then any dyadic, then the ratio nearest 2x."""
    cands = [(e, e["side"] / e["n_out"]) for e in entries if e["n_out"] > 0]
    if not cands:
        return None, None, "no recognized geometry"
    exact2 = [(e, r) for e, r in cands if e["side"] == 2 * e["n_out"]]
    if exact2:
        e, r = min(exact2, key=lambda t: (t[0]["tiles"], t[0]["side"]))
        return e, r, ""
    dy = [(e, r) for e, r in cands if is_dyadic(r)]
    if dy:
        e, r = min(dy, key=lambda t: (t[1], t[0]["tiles"], t[0]["side"]))
        return e, r, ""
    down = [(e, r) for e, r in cands if r > 1 + 1e-9]
    if down:
        e, r = min(down, key=lambda t: (abs(math.log2(t[1]) - 1), t[0]["tiles"]))
        return e, r, "no dyadic ratio at default policy"
    e, r = min(cands, key=lambda t: t[0]["side"])
    return e, r, "no downscale at default policy (identity or upscale only)"


def certify(proc, a, b):
    from PIL import Image
    sa, mode = call_any(proc, Image.fromarray(a))
    sb, _ = call_any(proc, Image.fromarray(b))
    rep = per_field_report(sa, sb)
    return dict(certified=int(rep["collides"] and state_digest(sa) == state_digest(sb)),
                fields_differ=",".join(rep["differ"] + rep["missing"]) or "none",
                n_fields=len(sa), mode=mode)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def describe(proc):
    ip = getattr(proc, "image_processor", proc)
    return dict(processor_class=type(proc).__name__, image_processor_class=type(ip).__name__,
                resample=resample_int(getattr(ip, "resample", -1)),
                backend=str(getattr(ip, "backend", "unknown")),
                patch=getattr(ip, "patch_size", None))


def load_variants(path, overrides):
    """[(label, processor, how, error)] for the default backend and, when it differs, Pillow's."""
    from transformers import AutoImageProcessor, AutoProcessor
    variants, errors = [], []
    proc = None
    for how, fn in (("AutoProcessor", lambda: AutoProcessor.from_pretrained(path, trust_remote_code=True, **overrides)),
                    ("AutoImageProcessor", lambda: AutoImageProcessor.from_pretrained(path, trust_remote_code=True, **overrides))):
        try:
            proc = fn()
            variants.append(("default", proc, how, ""))
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{how}: {type(exc).__name__}: {str(exc)[:160]}")
    if proc is None:
        return [("default", None, "", "no processor could be loaded: " + " | ".join(errors))]
    if describe(proc)["backend"] == "torchvision":
        try:
            pil = AutoImageProcessor.from_pretrained(path, trust_remote_code=True, backend="pil", **overrides)
            variants.append(("pil", pil, "AutoImageProcessor[backend=pil]", ""))
        except Exception as exc:  # noqa: BLE001
            variants.append(("pil", None, "AutoImageProcessor[backend=pil]",
                             f"pil backend unavailable: {type(exc).__name__}: {str(exc)[:160]}"))
    return variants


def config_keys(path):
    fp = pathlib.Path(path) / "preprocessor_config.json"
    if not fp.exists():
        return "preprocessor_config.json: absent"
    try:
        blob = json.loads(fp.read_text())
    except Exception as exc:  # noqa: BLE001
        return f"preprocessor_config.json: unreadable ({type(exc).__name__})"
    keep = {k: v for k, v in blob.items() if any(t in k for t in CONFIG_KEYS)}
    return json.dumps(keep, sort_keys=True, default=str)[:400]


# ---------------------------------------------------------------------------
# the core, per processor variant
# ---------------------------------------------------------------------------

def run_variant(name, proc, trials, seed, base_row):
    """All CSV rows for one loaded processor.  Never raises: failures become rows."""
    rows = []
    d = describe(proc)
    row = dict(base_row, processor_class=d["processor_class"],
               image_processor_class=d["image_processor_class"], resample=d["resample"])

    def emit(**kw):
        r = dict(row, **kw)
        rows.append({c: r.get(c, "") for c in COLS})
        return r

    t0 = time.time()
    entries, errs, unknown = probe_policy(proc, d["patch"])
    row["policy"] = policy_string(entries, errs, d["backend"])
    print(f"variant={d['image_processor_class']} backend={d['backend']} resample={d['resample']} "
          f"policy={row['policy']} probe_s={time.time()-t0:.0f}")
    if not entries:
        emit(trial=0, arm="exact_kernel", certified=0,
             error=f"geometry not recognized; emitted={json.dumps(unknown, default=str)[:200]}")
        return rows

    e, ratio, note = choose_geometry(entries)
    n_in, n_out = e["side"], e["n_out"]
    row.update(n_in=n_in, n_out=n_out, ratio=round(ratio, 4), dyadic=int(is_dyadic(ratio)))
    ratios = [(n_out, "primary")] + ([(e["secondary"], "thumb")] if e["secondary"] else [])
    print(f"chosen n_in={n_in} n_out={n_out} ratio={ratio:.4f} dyadic={row['dyadic']} "
          f"mode={e['mode']} tiles={e['tiles']} secondary={e['secondary']} note='{note}'")

    rng = np.random.default_rng(seed)
    try:
        K = coeffs_for(d["resample"], n_in, n_out)
    except ValueError as exc:
        emit(trial=0, arm="exact_kernel", certified=0, error=f"{note}; {exc}".strip("; "))
        return rows
    if ratio > 1 + 1e-9 and not bit_exact(d["resample"], n_in, n_out, K, rng):
        emit(trial=0, arm="exact_kernel", certified=0,
             error=f"{note}; coefficient transcription not bit-exact vs Pillow".strip("; "))
        return rows

    t0 = time.time()
    try:
        gb = gs_bound(K) if ratio > 1 + 1e-9 else None
    except Exception as exc:  # noqa: BLE001
        gb = None
        note = f"{note}; gs_bound failed: {type(exc).__name__}".strip("; ")
    row["gs_bound"] = "" if gb is None else f"{gb:.4g}"
    print(f"gs_bound_32={row['gs_bound'] or 'none'} (window 32, min over windows) {time.time()-t0:.0f}s")

    # the Pillow fixed-point path check: proc(x) must equal proc(Pillow-resize(x))
    if ratio > 1 + 1e-9 and e["tiles"] == 1 and d["resample"] in (1, 2, 3, 4):
        from PIL import Image
        base = build_base(n_in, n_in, rng)
        small = np.asarray(Image.fromarray(base).resize((n_out, n_out), pil_resample(d["resample"])))
        try:
            c = certify(proc, base, small)
            emit(trial=0, arm="pillow_path", certified=c["certified"], fields_differ=c["fields_differ"],
                 n_fields=c["n_fields"], pix_diff="",
                 error="" if c["certified"] else "processor output differs from Pillow's fixed-point resize")
            print(f"pillow_path match={c['certified']} fields_differ={c['fields_differ']} call={c['mode']}")
        except Exception as exc:  # noqa: BLE001
            emit(trial=0, arm="pillow_path", certified=0, error=f"pillow_path check raised {type(exc).__name__}: {str(exc)[:120]}")
    else:
        emit(trial=0, arm="pillow_path", certified="", error="pillow_path check not defined (tiled policy, identity, or unsupported resample)")

    vec, meta = find_short_vector(K, width=64, stride=16, max_abs=255)
    if vec is None:
        why = ("resize is identity at default policy (null space {0})" if abs(ratio - 1) < 1e-9
               else "no integer kernel vector with max|v|<=255 in a 64-wide window")
        emit(trial=0, arm="exact_kernel", certified=0, error=f"{note}; {why}".strip("; "))
        return rows
    vmax = int(np.abs(vec).max())
    print(f"kernel_vector max_abs={vmax} nonzeros={int((vec != 0).sum())} window={meta.get('c0')}")
    vec_stacked = None
    if e["secondary"]:
        try:
            K2 = np.vstack([K, coeffs_for(d["resample"], n_in, e["secondary"])])
            vec_stacked, m2 = find_short_vector(K2, width=64, stride=16, max_abs=255)
            print(f"stacked_kernel(primary+thumb) found={vec_stacked is not None} "
                  f"max_abs={m2.get('best_max_abs')}")
        except Exception as exc:  # noqa: BLE001
            print(f"stacked_kernel failed {type(exc).__name__}")

    for trial in range(trials):
        rng = np.random.default_rng(seed * 100 + trial)
        base = build_base(n_in, n_in, rng)
        ma, mb = row_masks(n_in)
        arms = [("exact_kernel", vec)]
        if e["secondary"]:
            arms.append(("exact_kernel_stacked", vec_stacked))
        off = np.zeros_like(vec)
        nz = np.flatnonzero(vec)
        off[nz] = rng.integers(1, 4, size=nz.size)       # same support, wrong direction
        arms.append(("negative_control", off))
        for arm, v in arms:
            if v is None:
                emit(trial=trial, arm=arm, certified=0, error=f"{note}; no short vector in the stacked kernel".strip("; "))
                continue
            amp = max(1, min(20, 100 // max(1, int(np.abs(v).max()))))
            a, b, mm = make_exact_pair(base, v, amp, ma, mb)
            if a is None or mm.get("pix_diff", 0) == 0:
                emit(trial=trial, arm=arm, certified=0, pix_diff=0, error=f"{note}; degenerate pair".strip("; "))
                continue
            try:
                c = certify(proc, a, b)
            except Exception as exc:  # noqa: BLE001
                emit(trial=trial, arm=arm, certified=0, pix_diff=mm["pix_diff"],
                     error=f"{note}; processor raised {type(exc).__name__}: {str(exc)[:120]}".strip("; "))
                continue
            emit(trial=trial, arm=arm, certified=c["certified"], fields_differ=c["fields_differ"],
                 n_fields=c["n_fields"], pix_diff=mm["pix_diff"], error=note)
            print(f"trial={trial} arm={arm} certified={c['certified']} pix_diff={mm['pix_diff']} "
                  f"amplitude={mm['max_amplitude']} n_fields={c['n_fields']} fields_differ={c['fields_differ']}")
    return rows


def write_rows(out, rows):
    if not out:
        return
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def summarize(name, rows):
    ek = [r for r in rows if r["arm"] in ("exact_kernel", "exact_kernel_stacked")]
    nc = [r for r in rows if r["arm"] == "negative_control"]
    pp = [r for r in rows if r["arm"] == "pillow_path" and r["certified"] != ""]
    first = next((r for r in rows if r["n_in"] != ""), rows[0] if rows else {})
    print(f"RESULT processor={name} variants={len({r['image_processor_class'] for r in rows})} "
          f"n_in={first.get('n_in', '')} n_out={first.get('n_out', '')} ratio={first.get('ratio', '')} "
          f"dyadic={first.get('dyadic', '')} gs_bound={first.get('gs_bound', '')} "
          f"certified={sum(int(r['certified'] or 0) for r in ek)}/{len(ek)} "
          f"control_certified={sum(int(r['certified'] or 0) for r in nc)} "
          f"pillow_path={sum(int(r['certified']) for r in pp)}/{len(pp)} "
          f"error=\"{(first.get('error') or '')[:100]}\"")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def selftest(trials, seed):
    """No files: a CLIPImageProcessor from kwargs, 448 -> 224, both backends when present."""
    from transformers import CLIPImageProcessor
    kw = dict(size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224}, resample=3,
              do_normalize=False)
    procs = [CLIPImageProcessor(**kw)]
    try:
        from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil
        procs.append(CLIPImageProcessorPil(**kw))
    except Exception as exc:  # noqa: BLE001
        print(f"selftest: no Pillow-backend CLIP class ({type(exc).__name__})")
    rows = []
    for p in procs:
        rows += run_variant("selftest_clip224", p, trials, seed, dict(processor="selftest_clip224"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processor", default="", help="staged processor directory")
    ap.add_argument("--name", default="", help="repo id / label for the processor column")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("TASK_ID", 0) or 0))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--override", action="append", default=[],
                    help="key=value passed to from_pretrained (e.g. max_pixels=200704)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-comment", action="store_true", help="write a placeholder row for a comment line")
    a = ap.parse_args()

    name = a.name or pathlib.Path(a.processor).name
    print(f"exp=e26_breadth version={VERSION} processor={name} path={a.processor or '-'} "
          f"trials={a.trials} seed={a.seed} sides={PROBE_SIDES[0]}..{PROBE_SIDES[-1]} "
          f"overrides={a.override} selftest={int(a.selftest)}")

    if a.skip_comment:
        rows = [{c: "" for c in COLS} | dict(processor=name[:60], arm="skipped", certified=0,
                                            error="skipped: comment line")]
        write_rows(a.out, rows)
        print(f"RESULT processor=comment skipped=1")
        return 0

    overrides = {}
    for kv in a.override:
        k, _, v = kv.partition("=")
        try:
            overrides[k] = json.loads(v)
        except Exception:
            overrides[k] = v

    if a.selftest:
        rows = selftest(a.trials, a.seed)
        write_rows(a.out, rows)
        summarize("selftest_clip224", rows)
        return 0

    base_row = dict(processor=name)
    rows = []
    if not pathlib.Path(a.processor).is_dir():
        rows.append({c: "" for c in COLS} | dict(base_row, arm="exact_kernel", certified=0,
                                                 error=f"processor dir missing: {a.processor}"))
        write_rows(a.out, rows)
        summarize(name, rows)
        return 0
    print(f"config={config_keys(a.processor)}")
    for label, proc, how, err in load_variants(a.processor, overrides):
        if proc is None:
            rows.append({c: "" for c in COLS} | dict(base_row, arm="exact_kernel", certified=0,
                                                     image_processor_class=label, error=err))
            print(f"variant={label} load_error=\"{err[:160]}\"")
            continue
        print(f"variant={label} loaded_via={how}")
        try:
            rows += run_variant(name, proc, a.trials, a.seed, base_row)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rows.append({c: "" for c in COLS} | dict(base_row, arm="exact_kernel", certified=0,
                                                     image_processor_class=type(getattr(proc, "image_processor", proc)).__name__,
                                                     error=f"driver raised {type(exc).__name__}: {str(exc)[:160]}"))
    write_rows(a.out, rows)
    summarize(name, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
