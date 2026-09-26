#!/usr/bin/env python3
"""E26: the per-field interface schema, and the certificate re-checked under the SCORED prompt.

Two things the completeness appendix has to state and has so far only asserted.

  1. WHAT the certificate hashes.  `aliasforge.interface.state_digest` hashes every field
     the HF processor returns, with its name, dtype, shape and bytes.  This records
     that enumeration per model: field, dtype, shape, whether it depends on the image,
     and whether it is in the digest.  The `image_dependent` column is MEASURED (seed-0
     base vs seed-1 base at fixed geometry), not declared, and the `perturb_check` column
     confirms that flipping one element of each field moves the digest, so no field is
     hashed only nominally.

  2. WHICH prompt the hashed input_ids belong to.  E0a/E0_llava captured the processor
     output with a bare question (`capture(proc, image, QUESTION)`), while the scored
     forward pass in E0c goes through `Verifier._inputs`, which wraps the critic text in
     the chat template with an image placeholder (Qwen), or the literal `USER: <image>...`
     prompt of e0_llava (LLaVA).  The bare and scored prompts tokenize differently, so the
     bare certificate and the scored logits were not literally about the same input_ids.
     This re-runs the certificate under the exact scored prompt, so the hashed state IS
     the scored one.  Weights are never loaded: the processor is the whole interface.

Targets (chosen by $TASK_ID / --target): 0 = Qwen2.5-VL-7B-Instruct at max_pixels=448*448
with the 896->448 construction of e0a_exact.py; 1 = llava-1.5-7b-hf with the 672->336
construction of e0_llava.py.  Base images, row masks, kernel vector, amplitude and control
direction are the same builders with the same seeds, so the pairs are the ones already
certified, not new ones.

CPU only.  One CSV row per (model, prompt_mode, trial, arm, field) plus one `__state__`
summary row per (model, prompt_mode, trial, arm).
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from aliasforge.exact import find_short_vector, make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.interface import TEXT_FIELDS, field_digest, state_digest  # noqa: E402
from aliasforge.verifier import CRITIC_TEMPLATE  # noqa: E402

VERSION = "e26.1"
PROMPT_MODES = ("bare", "scored")
CONTROL_TRIAL = 0            # the one control pair, drawn from trial 0's rng as e0a does
AMPLITUDE = 20               # e0a_exact / e0_llava

# The two deployments already certified, with the exact geometry each used.
TARGETS = {
    "qwen": dict(model="Qwen__Qwen2.5-VL-7B-Instruct", side=896, target=448,
                 max_pixels=448 * 448),
    "llava": dict(model="llava-hf__llava-1.5-7b-hf", side=672, target=336,
                  max_pixels=None),
}
# Shared question / step (e0a_exact, e0c_exact, e0_llava all use these strings).
QUESTION = "How many horizontal marks appear in the image?"
STEP = "The image contains exactly three horizontal marks."

CSV_COLS = ["model", "prompt_mode", "trial", "arm", "field", "dtype", "shape",
            "image_dependent", "in_digest", "digest_a16", "digest_b16", "same",
            "perturb_check", "processor_class", "image_processor_class", "resample",
            "versions"]


# ----------------------------------------------------------------------------
# pair construction -- verbatim from e0a_exact.py / e0_llava.py
# ----------------------------------------------------------------------------
def build_base(H, W, rng):
    return np.clip(
        128 + 18 * np.sin(np.mgrid[0:H, 0:W][1] / 11.0)[..., None]
        + rng.integers(-6, 7, (H, W, 3)), 0, 255).astype(np.uint8)


def row_masks(H, n_a=3, n_b=2):
    a, b = np.zeros(H, bool), np.zeros(H, bool)
    for i, y0 in enumerate((int(.13 * H), int(.38 * H), int(.63 * H))):
        if i < n_a:
            a[y0:y0 + int(.12 * H)] = True
        if i < n_b:
            b[y0:y0 + int(.12 * H)] = True
    return a, b


def control_vector(vec, rng):
    """Same support as the kernel vector, random (non-kernel) direction."""
    off = np.zeros_like(vec)
    nz = np.flatnonzero(vec)
    off[nz] = rng.integers(1, 4, size=nz.size)
    return off


def build_pairs(side, target, seed, trials):
    """(trial, arm, a, b, meta) for the certified trials and the one control."""
    K = pil_coeffs(side, target)
    vec, meta = find_short_vector(K, width=64, stride=16, max_abs=8)
    if vec is None:
        raise RuntimeError(f"no short integer kernel vector for {side}->{target}")
    print(f"kernel_vector side={side} target={target} max_abs={int(np.abs(vec).max())} "
          f"nonzeros={int((vec != 0).sum())} window={meta.get('c0')}")
    ma, mb = row_masks(side)
    pairs = []
    for trial in range(trials):
        rng = np.random.default_rng(seed * 100 + trial)
        base = build_base(side, side, rng)
        a, b, mm = make_exact_pair(base, vec, AMPLITUDE, ma, mb)
        pairs.append((trial, "certified", a, b, mm))
        if trial == CONTROL_TRIAL:
            a2, b2, mm2 = make_exact_pair(base, control_vector(vec, rng), AMPLITUDE, ma, mb)
            pairs.append((trial, "control", a2, b2, mm2))
    return pairs


# ----------------------------------------------------------------------------
# prompt construction -- the two modes under comparison
# ----------------------------------------------------------------------------
def scored_prompt(processor, target_name, question=QUESTION, step=STEP):
    """The text the scored forward pass actually fed the processor.

    Qwen: `Verifier._inputs` -- chat template over [{image},{text}], generation prompt
    appended, falling back to the bare critic text if the template is unavailable.
    LLaVA: the literal prompt of e0_llava.py.
    Returns (prompt, how) so the log records which path was taken.
    """
    text = CRITIC_TEMPLATE.format(question=question, step=step)
    if target_name == "llava":
        return (f"USER: <image>\n{question}\nCandidate step: {step}\n"
                f"Is this step correct? ASSISTANT:"), "e0_llava_literal"
    try:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return prompt, "apply_chat_template"
    except Exception as exc:  # noqa: BLE001 -- Verifier swallows this too; we record it
        return text, f"fallback_bare({type(exc).__name__})"


def capture_state(processor, image, text):
    """`aliasforge.interface.capture` with the call path recorded."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    try:
        out = processor(images=image, text=text, return_tensors="np")
        how = "images+text"
    except Exception:  # noqa: BLE001 -- same fallback as interface.capture
        out = processor(images=image, return_tensors="np")
        how = "images"
    return {k: out[k] for k in out}, how


# ----------------------------------------------------------------------------
# schema, image dependence, and the adversarial per-field check
# ----------------------------------------------------------------------------
def describe(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return str(value.dtype), str(tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return type(value).__name__, str((len(value),))
    return type(value).__name__, "()"


def perturbed(value):
    """A copy of `value` with exactly one element changed; None if no change is possible."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        v = np.array(value, copy=True)
        idx = np.unravel_index(0, v.shape)
        if v.dtype.kind == "b":
            v[idx] = not v[idx]
        elif v.dtype.kind in "iu":
            v[idx] = v[idx] + 1 if v[idx] < np.iinfo(v.dtype).max else v[idx] - 1
        else:
            v[idx] = v[idx] + 1.0 if np.isfinite(v[idx]) else 0.0
        return v
    if isinstance(value, (list, tuple)):
        v = list(value)
        if v:
            v[0] = perturbed(v[0]) if isinstance(v[0], (np.ndarray, list, tuple)) else (
                v[0] + 1 if isinstance(v[0], (int, float)) else str(v[0]) + "x")
        else:
            v = [0]
        return type(value)(v)
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return str(value) + "x"


def perturb_check(state):
    """flip one element of each field in turn; the state digest must move every time."""
    base = state_digest(state)
    out = {}
    for k in state:
        p = perturbed(state[k])
        if p is None:
            out[k] = "empty"
            continue
        s2 = dict(state)
        s2[k] = p
        out[k] = int(state_digest(s2) != base)
    return out


def image_dependence(processor, side, text):
    """field -> 1 if its digest differs between two DIFFERENT base images, same geometry."""
    ia = build_base(side, side, np.random.default_rng(0))
    ib = build_base(side, side, np.random.default_rng(1))
    sa, _ = capture_state(processor, ia, text)
    sb, _ = capture_state(processor, ib, text)
    keys = sorted(set(sa) | set(sb))
    return {k: int(k not in sa or k not in sb
                   or field_digest(sa[k]) != field_digest(sb[k])) for k in keys}


# ----------------------------------------------------------------------------
# processor loading -- exactly as Verifier.__init__, but nothing else
# ----------------------------------------------------------------------------
def load_processor(model_dir, max_pixels):
    from transformers import AutoProcessor

    kw = {"trust_remote_code": True}
    if max_pixels is not None:
        kw["max_pixels"] = max_pixels
    return AutoProcessor.from_pretrained(model_dir, **kw)


def versions_string():
    import PIL
    import torch
    import torchvision
    import transformers

    return (f"transformers={transformers.__version__};torchvision={torchvision.__version__};"
            f"Pillow={PIL.__version__};numpy={np.__version__};torch={torch.__version__}")


def processor_facts(processor):
    ip = getattr(processor, "image_processor", processor)
    return dict(processor_class=type(processor).__name__,
                image_processor_class=type(ip).__name__,
                resample=str(getattr(ip, "resample", None)),
                size=str(getattr(ip, "size", None)))


# ----------------------------------------------------------------------------
def run(target_name, processor, side, target, seed, trials, out, prompt_modes=PROMPT_MODES,
        prompt_texts=None):
    facts = processor_facts(processor)
    vers = versions_string()
    print(f"exp=e26_fields version={VERSION} target={target_name} model={TARGETS[target_name]['model']} "
          f"side={side} target_px={target} max_pixels={TARGETS[target_name]['max_pixels']} "
          f"trials={trials} seed={seed} amplitude={AMPLITUDE} "
          f"processor={facts['processor_class']} image_processor={facts['image_processor_class']} "
          f"resample={facts['resample']} size={facts['size']} {vers}")

    pairs = build_pairs(side, target, seed, trials)
    rows = []
    summary = {}
    for mode in prompt_modes:
        if prompt_texts is not None and mode in prompt_texts:
            text, how = prompt_texts[mode]
        elif mode == "bare":
            text, how = QUESTION, "bare_question"
        else:
            text, how = scored_prompt(processor, target_name)
        print(f"prompt_mode={mode} construction={how} n_chars={len(text)}")

        dep = image_dependence(processor, side, text)
        # the adversarial check runs on ONE captured state per mode: trial 0, member a
        s0, call_mode = capture_state(processor, pairs[0][2], text)
        pchk = perturb_check(s0)
        print(f"prompt_mode={mode} call_mode={call_mode} fields={','.join(sorted(s0))} "
              f"image_dependent={','.join(k for k in sorted(dep) if dep[k]) or 'none'} "
              f"perturb_pass={sum(1 for v in pchk.values() if v == 1)}/{len(pchk)}")

        n_cert = n_cert_ok = 0
        n_ctrl = n_ctrl_ok = 0
        for trial, arm, a, b, mm in pairs:
            if a is None or mm.get("pix_diff", 0) == 0:
                print(f"prompt_mode={mode} trial={trial} arm={arm} degenerate pair, skipped")
                continue
            sa, _ = capture_state(processor, a, text)
            sb, _ = capture_state(processor, b, text)
            da, db = state_digest(sa), state_digest(sb)
            keys = sorted(set(sa) | set(sb))
            differ = []
            for k in keys:
                va, vb = sa.get(k), sb.get(k)
                fa = field_digest(va) if k in sa else ""
                fb = field_digest(vb) if k in sb else ""
                same = int(fa == fb and k in sa and k in sb)
                if not same:
                    differ.append(k)
                dtype, shape = describe(va if k in sa else vb)
                rows.append(dict(model=TARGETS[target_name]["model"], prompt_mode=mode,
                                 trial=trial, arm=arm, field=k, dtype=dtype, shape=shape,
                                 image_dependent=dep.get(k, ""), in_digest=1,
                                 digest_a16=fa[:16], digest_b16=fb[:16], same=same,
                                 perturb_check=pchk.get(k, ""),
                                 processor_class=facts["processor_class"],
                                 image_processor_class=facts["image_processor_class"],
                                 resample=facts["resample"], versions=vers))
            collide = int(da == db)
            rows.append(dict(model=TARGETS[target_name]["model"], prompt_mode=mode,
                             trial=trial, arm=arm, field="__state__", dtype="",
                             shape=f"({len(keys)} fields)",
                             image_dependent=int(any(dep.get(k, 0) == 1 for k in keys)),
                             in_digest=1, digest_a16=da[:16], digest_b16=db[:16],
                             same=collide,
                             perturb_check=int(all(pchk.get(k) == 1 for k in keys)),
                             processor_class=facts["processor_class"],
                             image_processor_class=facts["image_processor_class"],
                             resample=facts["resample"], versions=vers))
            if arm == "certified":
                n_cert += 1
                n_cert_ok += collide
            else:
                n_ctrl += 1
                n_ctrl_ok += collide
            print(f"prompt_mode={mode} trial={trial} arm={arm} collide={collide} "
                  f"pix_diff={mm['pix_diff']} amplitude={mm['max_amplitude']} "
                  f"fields_differ={','.join(differ) or 'none'} "
                  f"text_fields_differ={','.join(k for k in differ if k in TEXT_FIELDS) or 'none'}")
        summary[mode] = dict(certified=n_cert_ok, n_cert=n_cert, control_certified=n_ctrl_ok,
                             n_ctrl=n_ctrl, fields=len(s0),
                             image_dependent=sum(dep.values()),
                             perturb_pass=sum(1 for v in pchk.values() if v == 1),
                             construction=how)

    if out and rows:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLS)
            w.writeheader()
            w.writerows(rows)

    for mode, s in summary.items():
        print(f"SUMMARY target={target_name} prompt_mode={mode} construction={s['construction']} "
              f"certified={s['certified']}/{s['n_cert']} "
              f"control_certified={s['control_certified']}/{s['n_ctrl']} "
              f"fields={s['fields']} image_dependent_fields={s['image_dependent']} "
              f"perturb_pass={s['perturb_pass']}/{s['fields']}")
    sc = summary.get("scored", summary.get(prompt_modes[-1]))
    print(f"RESULT target={target_name} model={TARGETS[target_name]['model']} "
          f"scored_certified={sc['certified']}/{sc['n_cert']} "
          f"scored_control_false_certifications={sc['control_certified']} (must be 0) "
          f"fields={sc['fields']} perturb_pass={sc['perturb_pass']}/{sc['fields']} rows={len(rows)}")
    return rows


# ----------------------------------------------------------------------------
# local self-test: no processor files exist locally, so build the image processors from
# kwargs and wrap them in a stub that tokenizes text into a fake input_ids, exercising the
# whole pipeline (both prompt modes, text fields, schema, perturb check) without weights.
# ----------------------------------------------------------------------------
class _StubProcessor:
    def __init__(self, image_processor, template=True):
        self.image_processor = image_processor
        self._template = template

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        if not self._template:
            raise ValueError("no chat template")
        txt = "".join(c["text"] for m in msgs for c in m["content"] if c["type"] == "text")
        return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{txt}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, images=None, text=None, return_tensors="np"):
        out = dict(self.image_processor(images=images, return_tensors=return_tensors))
        if text is not None:
            ids = np.frombuffer(text.encode(), dtype=np.uint8).astype(np.int64)[None]
            out["input_ids"] = ids
            out["attention_mask"] = np.ones_like(ids)
        return out


def selftest(out):
    import warnings

    warnings.filterwarnings("ignore")
    from transformers import CLIPImageProcessor, Qwen2VLImageProcessor

    ip_q = Qwen2VLImageProcessor(max_pixels=448 * 448)
    ip_l = CLIPImageProcessor(size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336},
                              resample=3, do_center_crop=True, do_resize=True, do_normalize=True,
                              do_convert_rgb=True,
                              image_mean=[0.48145466, 0.4578275, 0.40821073],
                              image_std=[0.26862954, 0.26130258, 0.27577711])
    all_rows = []
    for name, ip in (("qwen", ip_q), ("llava", ip_l)):
        t = TARGETS[name]
        rows = run(name, _StubProcessor(ip), t["side"], t["target"], 0, 2,
                   out.replace(".csv", f"_{name}.csv") if out else "")
        all_rows += rows
    # a stub without a template must take Verifier's fallback path, and be recorded as such
    p, how = scored_prompt(_StubProcessor(ip_q, template=False), "qwen")
    print(f"selftest fallback_path={how} prompt_is_bare_text={int(p == CRITIC_TEMPLATE.format(question=QUESTION, step=STEP))}")
    st = [r for r in all_rows if r["field"] == "__state__"]
    print(f"SELFTEST rows={len(all_rows)} state_rows={len(st)} "
          f"certified_collide={sum(r['same'] for r in st if r['arm']=='certified')}/"
          f"{sum(1 for r in st if r['arm']=='certified')} "
          f"control_collide={sum(r['same'] for r in st if r['arm']=='control')}/"
          f"{sum(1 for r in st if r['arm']=='control')} "
          f"perturb_all_pass={int(all(r['perturb_check'] == 1 for r in st))} "
          f"scored_differs_from_bare={int(any(r['digest_a16'] != s['digest_a16'] for r in st for s in st if r['prompt_mode']=='bare' and s['prompt_mode']=='scored' and r['model']==s['model'] and r['trial']==s['trial'] and r['arm']==s['arm']))}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="", help="staged processor dir (configs/tokenizer only)")
    ap.add_argument("--target", default=None, choices=list(TARGETS),
                    help="which deployment; default from TASK_ID (0=qwen, 1=llava)")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.out)
    if not a.model_dir:
        ap.error("--model-dir is required (or --selftest)")
    target_name = a.target or list(TARGETS)[int(os.environ.get("TASK_ID", 0))]
    t = TARGETS[target_name]
    import faulthandler
    faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)   # where are we if we hang
    print(f"loading processor from {a.model_dir} (files: {sorted(os.listdir(a.model_dir))[:12]})", flush=True)
    processor = load_processor(a.model_dir, t["max_pixels"])
    print(f"processor loaded: {type(processor).__name__}", flush=True)
    run(target_name, processor, t["side"], t["target"], a.seed, a.trials, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
