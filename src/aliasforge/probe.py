#!/usr/bin/env python3
"""Discover a target verifier's visual interface: what fields exist, and how they move.

The certificate is only as strong as the enumeration of image-dependent state
(Appendix B), and that enumeration is an empirical fact about an implementation,
not something to assume.  This module runs real images through a real processor
and reports every tensor and every scalar the preprocessing emits, with dtype,
shape and -- crucially -- whether the field RESPONDS to a change in the image.

A field that never moves cannot carry information and need not be hashed; a field
that moves must be hashed or Theorem 1 is silently weakened.  The `responds`
column is what tells the two apart, and it is measured rather than declared.

Output: one CSV row per (model, probe image, field), plus a concise stdout digest.

Usage:
    python -m aliasforge.probe --model /path/to/model --out fields.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import traceback

import numpy as np

# Probe images are synthetic and generated here so the probe depends on no
# dataset.  Sizes are chosen to exercise the policies that matter: a square, a
# non-square needing aspect handling, one below and one above a typical
# resize threshold.
PROBE_SIZES = [(448, 448), (500, 333), (1024, 768), (224, 224)]


def make_probe_image(h: int, w: int, seed: int = 0):
    """A deterministic RGB image with structure at every spatial frequency.

    Flat images collide trivially under any resize, which would make the
    `responds` column meaningless, so this carries a broadband pattern.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = (
        80
        + 60 * np.sin(2 * np.pi * xx / 37.0)
        + 50 * np.cos(2 * np.pi * yy / 23.0)
        + 40 * np.sin(2 * np.pi * (xx + yy) / 7.0)  # near-Nyquist content
    )
    arr = np.clip(base[..., None] + rng.integers(-8, 9, size=(h, w, 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def describe(value):
    """(kind, dtype, shape, summary) for any field a processor might return."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        return ("array", str(arr.dtype), str(tuple(arr.shape)), f"mean={arr.astype(np.float64).mean():.6g}")
    if isinstance(value, (list, tuple)):
        return ("seq", type(value).__name__, str((len(value),)), str(value)[:80])
    return ("scalar", type(value).__name__, "()", str(value)[:80])


def fingerprint(value) -> str:
    """A content digest for one field, stable across runs and machines."""
    import hashlib

    h = hashlib.sha256()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    else:
        h.update(repr(value).encode())
    return h.hexdigest()[:16]


def load_processor(model_dir: str):
    """Return (processor, how) for a local model dir, trying the usual entry points."""
    from transformers import AutoProcessor, AutoImageProcessor

    errors = []
    for name, fn in (
        ("AutoProcessor", lambda: AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)),
        ("AutoImageProcessor", lambda: AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)),
    ):
        try:
            return fn(), name
        except Exception as exc:  # noqa: BLE001 - we want the reason, not a crash
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no processor could be loaded:\n  " + "\n  ".join(errors))


def call_processor(proc, image):
    """Run one image through, handling the processor/image-processor split."""
    # A full processor wants text alongside the image; an image processor does not.
    try:
        return proc(images=image, text="Describe the image.", return_tensors="np"), "images+text"
    except Exception:
        return proc(images=image, return_tensors="np"), "images"


def probe_model(model_dir: str) -> tuple[list[dict], dict]:
    """Enumerate the interface fields of one target."""
    rows: list[dict] = []
    proc, how = load_processor(model_dir)
    name = pathlib.Path(model_dir).name

    # The preprocessing policy itself is image-independent but determines the
    # interface, so record it: it is what a reader needs to reproduce a digest.
    cfg = {}
    for fname in ("preprocessor_config.json", "config.json"):
        fp = pathlib.Path(model_dir) / fname
        if fp.exists():
            try:
                blob = json.loads(fp.read_text())
                cfg[fname] = {
                    k: v for k, v in blob.items()
                    if isinstance(v, (int, float, str, bool, list)) and len(str(v)) < 200
                }
            except Exception:
                pass

    for (h, w) in PROBE_SIZES:
        img_a = make_probe_image(h, w, seed=0)
        img_b = make_probe_image(h, w, seed=1)  # a DIFFERENT image, same geometry
        try:
            out_a, mode = call_processor(proc, img_a)
            out_b, _ = call_processor(proc, img_b)
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "model": name, "probe_h": h, "probe_w": w, "field": "<ERROR>",
                "kind": "error", "dtype": "", "shape": "", "responds": "",
                "digest_a": "", "summary": f"{type(exc).__name__}: {exc}"[:200],
                "call_mode": "", })
            continue

        keys = sorted(set(out_a.keys()) | set(out_b.keys()))
        for k in keys:
            va, vb = out_a.get(k), out_b.get(k)
            kind, dtype, shape, summary = describe(va)
            da, db = fingerprint(va), fingerprint(vb)
            rows.append({
                "model": name, "probe_h": h, "probe_w": w, "field": k,
                "kind": kind, "dtype": dtype, "shape": shape,
                # responds=1 means this field carries image information and MUST
                # be hashed; responds=0 means it is geometry-only for this pair.
                "responds": int(da != db),
                "digest_a": da, "summary": summary, "call_mode": mode,
            })
    return rows, cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local model directory")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    args = ap.parse_args(argv)

    name = pathlib.Path(args.model).name
    print(f"exp=probe_interface model={name} path={args.model}")

    try:
        rows, cfg = probe_model(args.model)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"RESULT model={name} status=failed error=\"{type(exc).__name__}: {exc}\"")
        return 1

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # concise, parsable stdout: one line per distinct field
    seen = {}
    for r in rows:
        seen.setdefault(r["field"], []).append(r)
    for field, rs in sorted(seen.items()):
        responds = sorted({str(r["responds"]) for r in rs})
        shapes = sorted({r["shape"] for r in rs})
        print(
            f"field={field} kind={rs[0]['kind']} dtype={rs[0]['dtype']} "
            f"responds={','.join(responds)} shapes={'|'.join(shapes[:4])}"
        )
    for fname, blob in cfg.items():
        keep = {k: v for k, v in blob.items() if any(
            t in k for t in ("size", "patch", "merge", "pixel", "resample",
                             "crop", "tile", "rescale", "norm", "thumbnail")
        )}
        if keep:
            print(f"config={fname} {json.dumps(keep, sort_keys=True)[:600]}")

    n_resp = sum(1 for r in rows if r["responds"] == 1)
    print(
        f"RESULT model={name} status=ok fields={len(seen)} rows={len(rows)} "
        f"image_responsive_rows={n_resp}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
