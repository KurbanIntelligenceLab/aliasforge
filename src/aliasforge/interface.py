#!/usr/bin/env python3
"""Capture, serialize and digest a verifier's visual interface state.

Assumption A3 requires the collision to be verified by comparing cryptographic
digests of the serialized state "including dtype, shape and every item of
non-array metadata".  This module is that comparison.  It is deliberately
paranoid in one direction only: it hashes MORE than it needs to, because a field
omitted here silently weakens Theorem 1 while a redundant field costs nothing.

The certified stage is the processor output -- everything the language model
conditions on that depends on the image.  Colliding there is sound for
Assumption A1 without any architectural argument: the model reads the image only
through this tensor bundle, so if the bundle is bit-identical the whole forward
pass is, for fixed weights and shared text.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

# Fields a processor returns that do NOT depend on the image (they echo the text
# prompt).  They are still hashed -- Assumption A2 requires the members to share
# them -- but they are tracked separately so a mismatch is diagnosed correctly:
# a difference here is a construction bug, not a failed collision.
TEXT_FIELDS = {"input_ids", "attention_mask", "token_type_ids"}


def _canon(value) -> tuple[str, bytes]:
    """Canonical (type-tag, bytes) for one field.  Order-free and machine-stable."""
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        tag = f"ndarray|{arr.dtype.str}|{arr.shape}"
        return tag, arr.tobytes()
    if isinstance(value, (list, tuple)):
        tag = f"seq|{type(value).__name__}|{len(value)}"
        return tag, json.dumps(value, sort_keys=True, default=str).encode()
    tag = f"scalar|{type(value).__name__}"
    return tag, repr(value).encode()


def field_digest(value) -> str:
    tag, blob = _canon(value)
    h = hashlib.sha256()
    h.update(tag.encode())
    h.update(b"\x00")
    h.update(blob)
    return h.hexdigest()


def state_digest(state: dict) -> str:
    """Digest of the whole interface state.

    Keys are sorted, and each field contributes its name, its type tag and its
    bytes, so a renamed, retyped, reshaped or reordered field all change the
    digest.  This is the object Assumption A3 compares.
    """
    h = hashlib.sha256()
    for k in sorted(state):
        tag, blob = _canon(state[k])
        h.update(k.encode())
        h.update(b"\x00")
        h.update(tag.encode())
        h.update(b"\x00")
        h.update(blob)
        h.update(b"\xff")
    return h.hexdigest()


def capture(processor, image, text: str = "Describe the image.") -> dict:
    """Run one image through the processor and return every emitted field."""
    try:
        out = processor(images=image, text=text, return_tensors="np")
    except Exception:
        out = processor(images=image, return_tensors="np")
    return {k: out[k] for k in out}


def per_field_report(state_a: dict, state_b: dict) -> dict:
    """Which fields agree, which differ, and which are missing on one side.

    Used by E0b: perturbing an enumerated field in isolation must change the
    digest, and this is what names the field that moved.
    """
    keys = sorted(set(state_a) | set(state_b))
    same, differ, missing = [], [], []
    for k in keys:
        if k not in state_a or k not in state_b:
            missing.append(k)
        elif field_digest(state_a[k]) == field_digest(state_b[k]):
            same.append(k)
        else:
            differ.append(k)
    return {
        "same": same,
        "differ": differ,
        "missing": missing,
        "image_fields_differ": [k for k in differ if k not in TEXT_FIELDS],
        "text_fields_differ": [k for k in differ if k in TEXT_FIELDS],
        "collides": not differ and not missing,
    }
