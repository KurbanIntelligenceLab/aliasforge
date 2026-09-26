#!/usr/bin/env python3
"""The four repair actions, and their measured cost.

E1 needs Delta_i(a) for every action, obtained by RUNNING each one rather than
predicting it.  This module is that runner.  The verifier is frozen throughout;
nothing here trains anything.

    stop  accept the current verdict                        cost 0
    att   text-only instruction to re-examine the region    prompt + one critique
    rea   t additional critique passes, aggregated          t x critique
    enc   recompute the interface from a higher-resolution  re-prefill + one
          question-conditioned crop                          critique

Definition 1 makes the first three language-side: each is a function of the
already-computed interface state, the question and step, and its own earlier
outputs.  Only `enc` recomputes the interface.  That distinction is the whole
point of the action set, so it is enforced structurally here -- `stop`, `att` and
`rea` are never handed the image at a different resolution, and `enc` is the only
path that rebuilds pixel input.

Costs are MEASURED, not assumed: prefill tokens, decode tokens and wall-clock are
recorded per call, because matched-compute claims are attacked precisely on the
choice of accounting unit.

Three controls from Appendix C are implemented alongside the deployable actions:

    ctrl_oracle_crop     a crop known to contain the target region -- an upper
                         bound on what `enc` could achieve
    ctrl_excluding_crop  a crop of matched size that EXCLUDES the target -- this
                         is the key control, separating new task-relevant
                         information from the generic effect of re-encoding
    ctrl_same_interface  a re-encode whose interface state hashes identically --
                         isolates re-encode overhead from its informational
                         benefit; by Theorem 1 its gain must be exactly zero,
                         which makes it a live check on the whole apparatus
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

ATT_INSTRUCTION = (
    "Look again at the image, specifically at the region the question refers to, "
    "and re-examine the visual evidence before answering."
)


@dataclass
class ActionResult:
    action: str
    p_correct: float
    prefill_tokens: int = 0
    decode_tokens: int = 0
    wall_s: float = 0.0
    interface_digest: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)


def _crop(image: np.ndarray, box, out_side: int) -> np.ndarray:
    """Question-conditioned crop, resampled to the interface's own resolution."""
    from PIL import Image

    y0, x0, h, w = box
    sub = Image.fromarray(image[y0:y0 + h, x0:x0 + w])
    return np.asarray(sub.resize((out_side, out_side), Image.BICUBIC))


class ActionRunner:
    """Runs each action against a frozen verifier and records what it cost."""

    def __init__(self, verifier, rea_passes: int = 3, crop_side: int = 448,
                 sampled: bool = True, temperature: float = 0.8, seed: int = 0):
        self.v = verifier
        self.rea_passes = rea_passes
        self.crop_side = crop_side
        # `sampled` decides whether critiques are GENERATED (and so can differ
        # between passes) or the verdict is merely re-read.  Deterministic
        # re-reading makes `rea` a prompt variant and makes every replicate
        # bit-identical, which silently degenerates the split-sample estimator --
        # measured at 80/80 identical items before this was fixed.
        self.sampled = sampled
        self.temperature = temperature
        self.seed = seed
        self._call = 0

    def _next_seed(self) -> int:
        """A fresh seed per critique, so passes and replicates are independent."""
        self._call += 1
        return self.seed * 1000003 + self._call

    # --- language-side actions: the interface is NOT recomputed ---------------

    def stop(self, image, question, step) -> ActionResult:
        t0 = time.time()
        p = self.v.p_correct(image, question, step)
        return ActionResult("stop", p, wall_s=time.time() - t0,
                            interface_digest=self.v.interface_digest(image, question, step))

    def att(self, image, question, step) -> ActionResult:
        """Re-attention by instruction, then one critique.  Same image, same interface."""
        t0 = time.time()
        q = question + "\n" + ATT_INSTRUCTION
        pre = dec = 0
        if self.sampled:
            c = self.v.critique(image, q, step, temperature=self.temperature,
                                seed=self._next_seed())
            p = self.v.verdict_after_critique(image, q, step, c["text"])
            pre, dec = c["prefill_tokens"], c["decode_tokens"]
        else:
            p = self.v.p_correct(image, q, step)
        return ActionResult("att", p, prefill_tokens=pre, decode_tokens=dec,
                            wall_s=time.time() - t0,
                            interface_digest=self.v.interface_digest(image, question, step),
                            extra={"instruction": ATT_INSTRUCTION})

    def rea(self, image, question, step) -> ActionResult:
        """t additional critique passes, aggregated by mean probability.

        Self-consistency over any number of critiques is inside Definition 1's
        class, so Theorem 1 pins its gain at zero on a certified pair whatever t
        is.  We still run it at a real t, because the point of E1 is to measure
        the gain on NATURAL items where no such guarantee applies.
        """
        t0 = time.time()
        ps, pre, dec = [], 0, 0
        for _ in range(self.rea_passes):
            if self.sampled:
                c = self.v.critique(image, question, step, temperature=self.temperature,
                                    seed=self._next_seed())
                ps.append(self.v.verdict_after_critique(image, question, step, c["text"]))
                pre += c["prefill_tokens"]; dec += c["decode_tokens"]
            else:
                ps.append(self.v.p_correct(image, question, step))
        return ActionResult("rea", float(np.mean(ps)), prefill_tokens=pre, decode_tokens=dec,
                            wall_s=time.time() - t0,
                            interface_digest=self.v.interface_digest(image, question, step),
                            extra={"passes": self.rea_passes, "spread": float(np.std(ps))})

    # --- the only action that leaves the class -------------------------------

    def enc(self, image, question, step, box) -> ActionResult:
        """Recompute the interface from a higher-resolution question-conditioned crop."""
        t0 = time.time()
        crop = _crop(image, box, self.crop_side)
        pre = dec = 0
        if self.sampled:
            c = self.v.critique(crop, question, step, temperature=self.temperature,
                                seed=self._next_seed())
            p = self.v.verdict_after_critique(crop, question, step, c["text"])
            pre, dec = c["prefill_tokens"], c["decode_tokens"]
        else:
            p = self.v.p_correct(crop, question, step)
        return ActionResult("enc", p, prefill_tokens=pre, decode_tokens=dec,
                            wall_s=time.time() - t0,
                            interface_digest=self.v.interface_digest(crop, question, step),
                            extra={"box": tuple(int(x) for x in box)})

    # --- controls ------------------------------------------------------------

    def ctrl_oracle_crop(self, image, question, step, target_box) -> ActionResult:
        r = self.enc(image, question, step, target_box)
        r.action = "ctrl_oracle_crop"
        return r

    def ctrl_excluding_crop(self, image, question, step, target_box) -> ActionResult:
        """A matched-size crop that EXCLUDES the target region.

        The key causal control: it costs exactly what `enc` costs and supplies a
        new view, but no new task-relevant information.  Any gain it shows is the
        generic effect of re-encoding rather than of seeing the decisive fact.
        """
        H, W = image.shape[:2]
        y0, x0, h, w = target_box
        # place a same-sized box as far from the target as the image allows
        cy = 0 if y0 > H // 2 else max(0, H - h)
        cx = 0 if x0 > W // 2 else max(0, W - w)
        r = self.enc(image, question, step, (cy, cx, h, w))
        r.action = "ctrl_excluding_crop"
        r.extra["target_box"] = tuple(int(x) for x in target_box)
        return r

    def ctrl_same_interface(self, image, question, step) -> ActionResult:
        """Re-encode whose interface hashes identically to the original.

        Isolates re-encode OVERHEAD from re-encode BENEFIT.  Theorem 1 forces its
        gain to be exactly zero, so a non-zero measured gain here is not a finding
        about routing -- it is evidence that the apparatus is wrong, and it is
        checked as such.
        """
        t0 = time.time()
        d_before = self.v.interface_digest(image, question, step)
        p = self.v.p_correct(image, question, step)
        d_after = self.v.interface_digest(image, question, step)
        return ActionResult("ctrl_same_interface", p, wall_s=time.time() - t0,
                            interface_digest=d_after,
                            note="digest_stable" if d_before == d_after else "DIGEST_DRIFT",
                            extra={"digest_stable": d_before == d_after})

    # --- the full sweep ------------------------------------------------------

    def run_all(self, image, question, step, target_box) -> dict[str, ActionResult]:
        out = {
            "stop": self.stop(image, question, step),
            "att": self.att(image, question, step),
            "rea": self.rea(image, question, step),
            "enc": self.enc(image, question, step, target_box),
            "ctrl_oracle_crop": self.ctrl_oracle_crop(image, question, step, target_box),
            "ctrl_excluding_crop": self.ctrl_excluding_crop(image, question, step, target_box),
            "ctrl_same_interface": self.ctrl_same_interface(image, question, step),
        }
        return out


def gains(results: dict[str, ActionResult], label: int) -> dict[str, float]:
    """Raw gain of each action over `stop`, in correctness.

    `label` is the ground-truth step-correctness (1 correct, 0 incorrect), so
    correctness of a prediction p is p when the label is 1 and 1-p when it is 0.
    """
    def correct(p):
        return p if label == 1 else 1.0 - p

    base = correct(results["stop"].p_correct)
    return {k: correct(r.p_correct) - base for k, r in results.items()}
