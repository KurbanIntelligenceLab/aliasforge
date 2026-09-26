#!/usr/bin/env python3
"""E28b: the logit-gap arm for E28's certified alias on VisualPRM-8B's DEPLOYED path.

E28 built a certified pair at S=2688 on the sound integer-kernel lattice of the STACKED
deployed operator (rows of pil_coeffs(2688,1344) and pil_coeffs(2688,448): the 3x3 tile
grid and the thumbnail share the source columns and Pillow's horizontal pass runs
first, in fixed point), and showed that both members reach an identical
`internvl.interface_state` digest.  Table 1 reports, for the other two architectures,
"certified k/n" and "logit gap exactly 0" against a SELF control (the same image
twice, the hardware's own reproducibility floor) and a LIVE control (a differing pair
that must separate).  This experiment produces the same three numbers for VisualPRM-8B.

Per task (seed = --task), --trials trials.  CPU part: rebuild the S=2688 stacked-kernel
witness exactly as E28 does (its functions are imported), then per trial a certified
pair (fresh base noise, E0a row masks: three marks vs two, E28's amplitude rule) and a
random-direction control pair with the same support, amplitude and base -- the witness
entries permuted within their support with random signs, rejected unless K v != 0.
Every member goes through `internvl.interface_state` and its digest is recorded.

GPU part: VisualPRM-8B is loaded ONCE (`InternVLRewardModel`, bf16, remote code) and
scored the way its own `generate_steps_with_soft_score` scores a step: the question
and the candidate step are placed in ONE user turn of the `internvl2_5` template
(`### Question:\n<image>\n{q}\n\n### Solution Process:\n{step}`), the assistant turn
holds the '+' placeholder, `<image>` expands to <img> + <IMG_CONTEXT> x (256 x n_tiles)
+ </img>, and `forward(pixel_values, input_ids, attention_mask, image_flags)` is
called directly.  The decision position is the deployed one -- the last token of the
assistant role, whose next-token distribution over {'+','-'} IS the step score -- so
`logit_gap` is max |logits_a - logits_b| over the full vocabulary at that position;
`logit_gap_allpos` is the same maximum over every position.  Text is identical for
both members, so the only thing that can move a logit is the image tensor.

The remote code is checked before anything is scored: template name, forward
signature, num_image_token, placeholder machinery.  Any mismatch aborts with a clear
message rather than scoring the wrong thing.  `--no-model` runs the CPU part only and
prints the assembled prompt (the local self-test).
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "verify"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "constructibility"))

from aliasforge import internvl  # noqa: E402
from aliasforge.exact import make_exact_pair, pil_coeffs  # noqa: E402
from aliasforge.interface import per_field_report, state_digest  # noqa: E402
from e20_resamplers import bit_exact  # noqa: E402
from e28_visualprm import (  # noqa: E402
    default_resample_name, deployed_grid, gate_default, row_masks, sweep,
)

VERSION = "e28b.4"
SIDE = 2688
TILE = internvl.IMAGE_SIZE
WIDTHS = (32, 64)
STRIDE = 16
NOISE = 10                      # E28's base-noise half-range; amplitude follows from it
COLUMNS = ("seed", "trial", "arm", "side", "witness_maxabs", "amplitude", "pix_diff",
           "digests_collide", "fields_differ", "logit_gap", "argmax_a", "argmax_b",
           "p_plus_a", "p_plus_b", "wall_s", "digest16", "logit_gap_allpos")

# Shared question / step (e0a_exact, e0c_exact, e0_llava, e26_fields use these strings).
QUESTION = "How many horizontal marks appear in the image?"
STEP = "The image contains exactly three horizontal marks."

# The deployed prompt machinery, transcribed from the model repo's conversation.py
# (template 'internvl2_5', SeparatorStyle.MPT) and modeling_internvl_chat.py
# (InternVLRewardModel.generate_steps_with_soft_score).  Cross-checked against the
# loaded model's own template at run time.
TEMPLATE_NAME = "internvl2_5"
SYSTEM_TEMPLATE = "<|im_start|>system\n{system_message}"
SYSTEM_MESSAGE = ("你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位"
                  "联合开发的多模态大语言模型。")
ROLES = ("<|im_start|>user\n", "<|im_start|>assistant\n")
SEP = "<|im_end|>\n"
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
PLACEHOLDER = "+"
SCORE_TOKENS = ("+", "-")       # str2score = {'+': 1, '-': 0}


# --------------------------------------------------------------------------- witness
def stacked_witness(side, workers):
    """E28's per-side construction: gates, stacked operator, sweep, shortest witness."""
    rng = np.random.default_rng(0)
    (gx, gy), n_tiles, sizes_ok = deployed_grid(side)
    n_out = TILE * gx
    K1, K2 = pil_coeffs(side, n_out), pil_coeffs(side, TILE)
    g1 = gate_default(K1, side, n_out, rng) and bit_exact("bicubic", side, n_out, K1, rng)
    g2 = gate_default(K2, side, TILE, rng) and bit_exact("bicubic", side, TILE, K2, rng)
    if not (g1 and g2 and sizes_ok and n_tiles == gx * gy + 1):
        raise SystemExit(f"FATAL: deployed-path gate failed at side={side}: grid={gx}x{gy} "
                         f"n_tiles={n_tiles} tiles_448={sizes_ok} gate_primary={g1} gate_thumb={g2}")
    K = np.vstack([K1, K2])
    best = None
    for width in WIDTHS:
        t0 = time.time()
        n_win, lbs, b, n_unconv = sweep(K, side, width, STRIDE, workers)
        if not lbs:
            print(f"  width={width} windows={n_win} stacked kernel trivial", flush=True)
            continue
        c0, maxabs, wit, dim = b
        span = int(max(wit) - min(wit))
        print(f"  width={width} windows={n_win} nontrivial={len(lbs)} bound_min={min(lbs):.6g} "
              f"lll_unconverged={n_unconv} witness_maxabs={maxabs} span={span} c0={c0} dim={dim} "
              f"{time.time() - t0:.0f}s", flush=True)
        if best is None or maxabs < best["maxabs"]:
            best = {"c0": c0, "width": width, "maxabs": int(maxabs), "span": span,
                    "wit": [int(x) for x in wit], "bound_min": float(min(lbs))}
    if best is None or best["maxabs"] > 255 or best["span"] > 255:
        raise SystemExit(f"FATAL: no 8-bit witness at side={side}: {best}")
    full = np.zeros(side, dtype=np.int64)
    full[best["c0"]:best["c0"] + best["width"]] = best["wit"]
    if any(int(x) != 0 for x in (K.astype(object) @ full.astype(object))):
        raise SystemExit("FATAL: witness is not in the stacked kernel")
    best.update(grid=f"{gx}x{gy}", n_tiles=n_tiles, n_out=n_out, K=K, vec=full)
    return best


def control_vector(vec, K, rng, tries=50):
    """Same support and entry multiset as the witness, random direction; must NOT be in K's kernel."""
    nz = np.flatnonzero(vec)
    for _ in range(tries):
        v = np.zeros_like(vec)
        v[nz] = rng.permutation(vec[nz]) * rng.choice([-1, 1], size=nz.size)
        if np.array_equal(v, vec):
            continue
        if any(int(x) != 0 for x in (K.astype(object) @ v.astype(object))):
            return v
    raise SystemExit("FATAL: could not draw a non-kernel control vector")


def amplitude_rule(span, noise=NOISE):
    """E28's choice: the largest amplitude whose pedestal keeps the noisy base off the rails."""
    if span > 255 - 2 * noise:
        raise SystemExit(f"FATAL: witness span {span} does not fit an 8-bit image with noise {noise}")
    return max(1, min(60, (255 - 2 * noise) // span))


def build_pairs(side, wit, trial_rng):
    """(certified a, b, meta), (control a, b, meta) around one shared noisy base."""
    vec, K = wit["vec"], wit["K"]
    amp = amplitude_rule(wit["span"])
    w = amp * vec
    lo, hi = int(w.min()), int(w.max())
    ped = (-lo + (255 - hi)) // 2
    base = (ped + trial_rng.integers(-NOISE, NOISE + 1, (side, side, 3))).astype(np.int64)
    ma, mb = row_masks(side)
    xa, xb, meta = make_exact_pair(base, vec, amp, ma, mb)
    if xa is None:
        raise SystemExit(f"FATAL: certified pair saturates: {meta}")
    for _ in range(20):
        cv = control_vector(vec, K, trial_rng)
        ca, cb, cmeta = make_exact_pair(base, cv, amp, ma, mb)
        if ca is not None:
            break
    else:
        raise SystemExit("FATAL: every control draw saturated")
    return amp, (xa, xb, meta), (ca, cb, cmeta)


def digest_pair(xa, xb):
    sa, sb = internvl.interface_state(xa), internvl.interface_state(xb)
    rep = per_field_report(sa, sb)
    da, db = state_digest(sa), state_digest(sb)
    collide = int(da == db and rep["collides"])
    return sa, sb, da, db, collide, ";".join(rep["differ"] + rep["missing"])


# --------------------------------------------------------------------------- prompt
def build_query(n_tiles, num_image_token, system_message=SYSTEM_MESSAGE,
                question=QUESTION, step=STEP):
    """generate_steps_with_soft_score's query for ONE step, template internvl2_5 (MPT style)."""
    q = "<image>\n" + question
    step0 = f"### Question:\n{q}\n\n### Solution Process:\n{step}"
    ret = SYSTEM_TEMPLATE.format(system_message=system_message) + SEP
    ret += ROLES[0] + step0 + SEP
    ret += ROLES[1] + PLACEHOLDER + SEP
    image_tokens = IMG_START + IMG_CONTEXT * num_image_token * n_tiles + IMG_END
    return ret.replace("<image>", image_tokens, 1)


def placeholder_idx(ids_of, input_ids):
    """InternVLRewardModel.find_placeholder_idx with an injected `text -> ids` tokenizer."""
    bos = ids_of(ROLES[1])
    target = ids_of(ROLES[1] + PLACEHOLDER + SEP)
    idx = []
    for i in range(len(input_ids)):
        if input_ids[i:i + len(target)] == target:
            assert i + len(bos) - 1 >= 0
            idx.append(i + len(bos) - 1)
    return idx


def show_query(query):
    """Collapse the IMG_CONTEXT run so the prompt is printable."""
    n = query.count(IMG_CONTEXT)
    return query.replace(IMG_CONTEXT * n, f"<IMG_CONTEXT x{n}>").replace("\n", "\\n")


# --------------------------------------------------------------------------- model
class Scorer:
    """VisualPRM-8B loaded once; one forward per member, decision-position logits."""

    def __init__(self, model_dir, seed):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise SystemExit("FATAL: a GPU was allocated but torch.cuda.is_available() is False")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.torch = torch
        t0 = time.time()
        try:
            self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
            self.prompt_mode = "deployed"
        except Exception as exc:  # noqa: BLE001  (sentencepiece rejects this tokenizer.model)
            print(f"tokenizer_unavailable {type(exc).__name__}: {str(exc)[:120]} -> synthetic prompt "
                  f"(fixed token sequence, shared by both members; P(+) not defined)", flush=True)
            self.tok = None
            self.prompt_mode = "synthetic"
            with open(os.path.join(model_dir, "added_tokens.json")) as fh:
                added = json.load(fh)
            with open(os.path.join(model_dir, "tokenizer_config.json")) as fh:
                dec = json.load(fh).get("added_tokens_decoder", {})
            byname = {v["content"]: int(k) for k, v in dec.items()}
            self.special = {t: int(added[t]) for t in (IMG_START, IMG_END, IMG_CONTEXT)}
            self.special["im_start"] = byname["<|im_start|>"]
            self.special["im_end"] = byname["<|im_end|>"]
        # transformers >= 5.16 reads model.all_tied_weights_keys while loading; the remote-code
        # InternVLRewardModel never calls post_init(), so it lacks the attribute. A class-level
        # fallback (no tied weights) is the exact value post_init would compute for this model.
        from transformers import PreTrainedModel
        if getattr(PreTrainedModel, "all_tied_weights_keys", None) is None:
            PreTrainedModel.all_tied_weights_keys = {}
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True,
                                               low_cpu_mem_usage=True).cuda().eval()
        m = self.model
        # ---- the deployed machinery must be what this script transcribed
        problems = []
        if type(m).__name__ != "InternVLRewardModel":
            problems.append(f"model class is {type(m).__name__}, expected InternVLRewardModel")
        sig = list(inspect.signature(m.forward).parameters)
        for p in ("pixel_values", "input_ids", "attention_mask", "image_flags"):
            if p not in sig:
                problems.append(f"forward() lacks parameter {p!r}: {sig}")
        if getattr(m, "template", None) != TEMPLATE_NAME:
            problems.append(f"template is {getattr(m, 'template', None)!r}, expected {TEMPLATE_NAME!r}")
        if getattr(m, "system_message", None) != SYSTEM_MESSAGE:
            problems.append(f"system_message differs from the transcribed one: {m.system_message!r}")
        for attr in ("find_placeholder_idx", "generate_steps_with_soft_score", "num_image_token"):
            if not hasattr(m, attr):
                problems.append(f"model lacks {attr}")
        tpl = getattr(m, "conv_template", None)
        if tpl is None or tuple(tpl.roles) != ROLES or tpl.sep != SEP:
            problems.append(f"conversation template differs: roles={getattr(tpl, 'roles', None)} "
                            f"sep={getattr(tpl, 'sep', None)!r}")
        if self.tok is not None:
            self.ids = {t: self.tok.convert_tokens_to_ids(t) for t in (IMG_START, IMG_END, IMG_CONTEXT) + SCORE_TOKENS}
            for t, i in self.ids.items():
                if i is None or i == self.tok.unk_token_id:
                    problems.append(f"token {t!r} is unknown to the tokenizer")
        else:
            self.ids = {t: self.special[t] for t in (IMG_START, IMG_END, IMG_CONTEXT)}
            self.ids.update({t: None for t in SCORE_TOKENS})
        if problems:
            raise SystemExit("FATAL: remote code does not match the transcription:\n  " + "\n  ".join(problems))
        m.img_context_token_id = self.ids[IMG_CONTEXT]
        self.num_image_token = int(m.num_image_token)
        self.versions = f"torch={torch.__version__} transformers={transformers.__version__}"
        self.load_s = time.time() - t0
        self._input = None

    def prepare(self, n_tiles):
        """Tokenize the ONE fixed query and locate the decision position, the deployed way."""
        torch = self.torch
        m, tok = self.model, self.tok
        query = build_query(n_tiles, self.num_image_token, system_message=m.system_message)
        if tok is None:
            # The certificate is prompt-agnostic: any token sequence shared by both members
            # works. Keep the deployed structure (system, user with the image, assistant) with
            # a fixed ordinary piece id standing in for text, and score at the last position.
            T = 435
            sp = self.special
            ids = ([1, sp["im_start"]] + [T] * 8 + [sp["im_end"]] + [sp["im_start"]] + [T] * 4
                   + [sp[IMG_START]] + [sp[IMG_CONTEXT]] * (self.num_image_token * n_tiles) + [sp[IMG_END]]
                   + [T] * 16 + [sp["im_end"]] + [sp["im_start"]] + [T] * 2)
            input_ids = torch.tensor([ids], dtype=torch.long)
            self.score_pos = len(ids) - 1
            self._input = (input_ids.cuda(), torch.ones_like(input_ids).cuda())
            self.n_tokens = len(ids)
            self.query = f"<synthetic {len(ids)} tokens, {self.num_image_token * n_tiles} IMG_CONTEXT>"
            return self.query
        # the model's own template must produce the same string
        mod = sys.modules[type(m).__module__]
        tpl = mod.get_conv_template(m.template)
        tpl.system_message = m.system_message
        q = "<image>\n" + QUESTION
        tpl.append_message(tpl.roles[0], f"### Question:\n{q}\n\n### Solution Process:\n{STEP}")
        tpl.append_message(tpl.roles[1], PLACEHOLDER)
        ref = tpl.get_prompt().replace(
            "<image>", IMG_START + IMG_CONTEXT * self.num_image_token * n_tiles + IMG_END, 1)
        if ref != query:
            raise SystemExit(f"FATAL: transcribed prompt differs from the model's template:\n"
                             f"  model: {show_query(ref)}\n  mine:  {show_query(query)}")
        enc = tok(query, return_tensors="pt")
        input_ids = enc["input_ids"]
        n_ctx = int((input_ids[0] == self.ids[IMG_CONTEXT]).sum())
        if n_ctx != self.num_image_token * n_tiles:
            raise SystemExit(f"FATAL: {n_ctx} IMG_CONTEXT tokens in input_ids, expected "
                             f"{self.num_image_token}x{n_tiles}")
        idx = m.find_placeholder_idx(tok, input_ids, PLACEHOLDER=PLACEHOLDER)
        mine = placeholder_idx(lambda s: tok(s, add_special_tokens=False).input_ids, input_ids[0].tolist())
        if len(idx) != 1 or idx != mine:
            raise SystemExit(f"FATAL: placeholder index: model={idx} transcribed={mine} (expected one index)")
        self.score_pos = int(idx[0])
        self._input = (input_ids.cuda(), enc["attention_mask"].cuda())
        self.n_tokens = int(input_ids.shape[1])
        self.query = query
        return query

    def logits(self, pixel_values):
        """Full-vocab float32 logits (N, V) on the GPU for one member."""
        torch = self.torch
        input_ids, attention_mask = self._input
        pv = torch.from_numpy(np.ascontiguousarray(pixel_values)).to("cuda", getattr(self, "dtype", torch.bfloat16))
        flags = torch.ones(pv.shape[0], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = self.model(pixel_values=pv, input_ids=input_ids,
                             attention_mask=(None if getattr(self, "_mask_none", False) else attention_mask),
                             image_flags=flags)
        lg = out.logits[0].float()
        if lg.shape[0] != self.n_tokens:
            raise SystemExit(f"FATAL: logits have {lg.shape[0]} positions, input has {self.n_tokens}")
        return lg

    def sanity(self, pixel_values):
        """Diagnose the all-NaN logits of e28b.2/.3: (1) parameters: any NaN/inf/meta tensors;
        (2) the language model alone on a short text prompt, with and without an attention
        mask; (3) the vision tower alone; (4) the full forward with mask=None. Adopts the
        mask-free forward for the trials when that is the finite one."""
        torch = self.torch; m = self.model
        n_nan = n_meta = n_par = 0; big = 0.0
        for name, prm in m.named_parameters():
            n_par += 1
            if prm.is_meta: n_meta += 1; continue
            if not torch.isfinite(prm).all(): n_nan += 1
            big = max(big, float(prm.detach().abs().max()))
        emb = m.language_model.get_input_embeddings().weight
        print(f"diag params n={n_par} nonfinite={n_nan} meta={n_meta} absmax={big:.4g} embed_absmean={float(emb.detach().float().abs().mean()):.4g}", flush=True)
        ids = torch.tensor([[1, 435, 435, 435, 435]], device="cuda")
        for label, mask in (("lm_text_mask", torch.ones_like(ids)), ("lm_text_nomask", None)):
            with torch.no_grad():
                lg = m.language_model(input_ids=ids, attention_mask=mask).logits[0].float()
            print(f"diag {label} nan_frac={float(torch.isnan(lg).float().mean()):.4f} absmax={float(lg.nan_to_num().abs().max()):.4g}", flush=True)
        pv = torch.from_numpy(np.ascontiguousarray(pixel_values)).to("cuda", torch.bfloat16)
        with torch.no_grad():
            try:
                vf = m.extract_feature(pv).float()
                print(f"diag vision_feature shape={tuple(vf.shape)} nan_frac={float(torch.isnan(vf).float().mean()):.4f} absmax={float(vf.nan_to_num().abs().max()):.4g}", flush=True)
            except Exception as e:
                print(f"diag vision_feature error={type(e).__name__}: {e}", flush=True)
        input_ids, attention_mask = self._input
        flags = torch.ones(pv.shape[0], dtype=torch.long, device="cuda")
        for label, mask in (("full_mask", attention_mask), ("full_nomask", None)):
            with torch.no_grad():
                lg = m(pixel_values=pv, input_ids=input_ids, attention_mask=mask, image_flags=flags).logits[0].float()
            frac = float(torch.isnan(lg).float().mean())
            print(f"diag {label} nan_frac={frac:.4f} absmax={float(lg.nan_to_num().abs().max()):.4g} score_pos_nan={int(torch.isnan(lg[self.score_pos]).any())}", flush=True)
            if frac == 0.0:
                self.dtype = torch.bfloat16; self._mask_none = (mask is None); return torch.bfloat16
        self.dtype = torch.bfloat16; self._mask_none = False; return None

    def compare(self, la, lb):
        """(gap at the decision position, gap over all positions, argmax a/b, P(+) a/b)."""
        torch = self.torch
        p = self.score_pos
        gap = float((la[p] - lb[p]).abs().max())
        gap_all = float((la - lb).abs().max())
        cand = [self.ids[t] for t in SCORE_TOKENS]

        def pplus(l):
            if any(c is None for c in cand):
                return float("nan")
            return float(torch.softmax(l[p, cand], dim=-1)[0])

        return gap, gap_all, int(la[p].argmax()), int(lb[p].argmax()), pplus(la), pplus(lb)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--task", type=int, default=int(os.environ.get("TASK_ID", 0)))
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--no-model", action="store_true", help="CPU part only (local self-test)")
    a = ap.parse_args()
    if not a.model and not a.no_model:
        ap.error("--model is required unless --no-model")

    seed = a.task
    import PIL
    (gx, gy), n_tiles, _ = deployed_grid(SIDE)
    cfg = {}
    if a.model:
        try:
            with open(os.path.join(a.model, "config.json")) as fh:
                cfg = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FATAL: cannot read {a.model}/config.json: {exc}")
    tmpl = cfg.get("template", TEMPLATE_NAME)
    vc = cfg.get("vision_config", {})
    nit = int((cfg.get("force_image_size", TILE) // vc.get("patch_size", 14)) ** 2
              * cfg.get("downsample_ratio", 0.5) ** 2)
    versions = f"pillow={PIL.__version__} numpy={np.__version__}"
    try:
        import torch
        import transformers
        versions += f" torch={torch.__version__} transformers={transformers.__version__}"
    except Exception:  # noqa: BLE001
        versions += " torch=n/a transformers=n/a"
    print(f"exp=e28b_visualprm_logit version={VERSION} task={a.task}/{a.shards} seed={seed} "
          f"trials={a.trials} side={SIDE} grid={gx}x{gy} n_tiles={n_tiles} "
          f"model_dir={a.model or 'none'} arch={','.join(cfg.get('architectures', ['?']))} "
          f"template={tmpl} num_image_token={nit} deployed_resample={default_resample_name()} "
          f"{versions} question={QUESTION!r} step={STEP!r} score_pos=deployed_placeholder "
          f"no_model={int(a.no_model)}", flush=True)
    if tmpl != TEMPLATE_NAME or nit != 256:
        raise SystemExit(f"FATAL: config.json says template={tmpl} num_image_token={nit}; the "
                         f"transcription assumes {TEMPLATE_NAME}/256")

    # ---- CPU: the witness (identical to E28's) and the prompt
    t0 = time.time()
    wit = stacked_witness(SIDE, a.workers)
    print(f"witness side={SIDE} maxabs={wit['maxabs']} span={wit['span']} c0={wit['c0']} "
          f"width={wit['width']} bound_min={wit['bound_min']:.6g} amplitude={amplitude_rule(wit['span'])} "
          f"{time.time() - t0:.0f}s", flush=True)
    query = build_query(n_tiles, nit)
    print(f"prompt_transcribed n_img_context={query.count(IMG_CONTEXT)} text={show_query(query)}", flush=True)

    scorer = None
    if not a.no_model:
        t0 = time.time()
        scorer = Scorer(a.model, seed)
        scorer.prepare(n_tiles)
        print(f"model_loaded class={type(scorer.model).__name__} {scorer.versions} "
              f"num_image_token={scorer.num_image_token} n_tokens={scorer.n_tokens} "
              f"score_pos={scorer.score_pos} prompt_mode={scorer.prompt_mode} ids={scorer.ids} load_s={scorer.load_s:.0f} "
              f"{time.time() - t0:.0f}s", flush=True)
    else:
        ids_of = lambda s: [ord(c) for c in s]  # noqa: E731  stub: one id per character
        idx = placeholder_idx(ids_of, ids_of(query))
        print(f"stub_tokenizer placeholder_idx={idx} n_chars={len(query)} "
          f"token_at_idx={query[idx[0]]!r} next={query[idx[0] + 1]!r}", flush=True)

    rows = [COLUMNS]
    stats = {"certified": [], "control": [], "self": []}
    n_collide = {"certified": 0, "control": 0}
    t_start = time.time()
    for trial in range(a.trials):
        trial_rng = np.random.default_rng(seed * 1000 + trial)
        t0 = time.time()
        amp, (xa, xb, meta), (ca, cb, cmeta) = build_pairs(SIDE, wit, trial_rng)
        sa, sb, da, db, coll, differ = digest_pair(xa, xb)
        _, _, dca, dcb, ccoll, cdiffer = digest_pair(ca, cb)
        if np.array_equal(xa, xb) or np.array_equal(ca, cb):
            raise SystemExit("FATAL: degenerate pair (members pixel-identical)")
        n_collide["certified"] += coll
        n_collide["control"] += ccoll
        t_cpu = time.time() - t0
        arms = {
            "certified": (sa["pixel_values"], sb["pixel_values"], da, db, meta["pix_diff"], coll, differ),
            "control": (internvl.load_pixel_values(ca), internvl.load_pixel_values(cb),
                        dca, dcb, cmeta["pix_diff"], ccoll, cdiffer),
            "self": (sa["pixel_values"], sa["pixel_values"], da, da, 0, 1, ""),
        }
        line = [f"progress trial={trial + 1}/{a.trials} amp={amp} pix_diff={meta['pix_diff']} "
                f"cert_collide={coll} ctrl_collide={ccoll} cpu_s={t_cpu:.0f}"]
        for arm, (pa, pb, dga, dgb, pix, dcol, dif) in arms.items():
            t1 = time.time()
            gap = gap_all = arg_a = arg_b = pp_a = pp_b = ""
            if scorer is not None:
                if getattr(scorer, "dtype", None) is None:
                    scorer.sanity(pa)      # once: NaN check, float16 fallback
                la = scorer.logits(pa)
                lb = scorer.logits(pb)
                gap, gap_all, arg_a, arg_b, pp_a, pp_b = scorer.compare(la, lb)
                del la, lb
                stats[arm].append(gap)
                line.append(f"{arm}_gap={gap:.4g} allpos={gap_all:.4g} p_plus=({pp_a:.4f},{pp_b:.4f})")
            wall = round(time.time() - t1 + (t_cpu if arm == "certified" else 0.0), 2)
            for side_name, dg in (("a", dga), ("b", dgb)):
                rows.append((seed, trial, arm, side_name, wit["maxabs"], amp, pix, dcol, dif,
                             gap, arg_a, arg_b, pp_a, pp_b, wall, dg[:16], gap_all))
        line.append(f"elapsed_s={time.time() - t_start:.0f}")
        print(" ".join(line), flush=True)

    if a.out:
        with open(a.out, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

    n = a.trials
    print(f"RESULT task={a.task} seed={seed} trials={n} witness_maxabs={wit['maxabs']} "
          f"amplitude={amplitude_rule(wit['span'])} certified_collide={n_collide['certified']}/{n} "
          f"control_collide={n_collide['control']}/{n} rows={len(rows) - 1}")
    if scorer is not None:
        c, k, s = stats["certified"], stats["control"], stats["self"]
        floor = max(s) if s else 0.0
        at_floor = sum(1 for g in c if g <= floor)
        sep = sum(1 for g, sg in zip(k, s) if g > max(sg, 1e-12) * 10)
        print(f"RESULT cert_gap_max={max(c):.6g} cert_gap_zero={sum(1 for g in c if g == 0.0)}/{n} "
              f"cert_at_self_floor={at_floor}/{n} self_gap_max={floor:.6g} "
              f"ctrl_gap_min={min(k):.6g} controls_separated={sep}/{n} score_pos={scorer.score_pos} "
              f"n_tokens={scorer.n_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
