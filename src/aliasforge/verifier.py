#!/usr/bin/env python3
"""Run a frozen multimodal verifier and read its step-correctness decision.

Everything downstream needs this: E0c tests Theorem 1 on a real model, E1
measures Delta_i(a) by running each action, E2/E3 need the interface state the
probe reads.  The verifier is FROZEN throughout -- nothing here trains anything.

Design note on what to measure.  The manuscript's E0c asks for pair-balanced
accuracy within Monte Carlo error of 1/2 under stochastic decoding.  That is a
weak instrument: it needs many samples to resolve 1/2 to any precision, and it
converts an exact prediction into a noisy one.  Theorem 1 actually predicts
something far sharper.  If the interface states collide, the two members induce
the SAME output distribution, so the decision-position LOGIT VECTORS must be
identical -- not close, identical -- for a deterministic forward pass.  We
therefore measure max |logit(x0) - logit(x1)| directly.

That comparison is only meaningful against a noise floor, because GPU kernels are
not always bit-reproducible.  So every measurement is paired with a SELF test:
the same image pushed through twice.  The self difference is the floor; the pair
difference must not exceed it.  This separates "the theorem failed" from "the GPU
is nondeterministic", which a bare accuracy number cannot do.
"""

from __future__ import annotations

import numpy as np

from .interface import capture, state_digest

# Prompt used to turn a general instruction-tuned VLM into a step critic.  Kept
# verbatim here so the exact conditioning is part of the recorded artifact:
# Assumption A2 requires both members to share Q and R as identical token
# sequences, and that is only checkable if the template is pinned.
CRITIC_TEMPLATE = (
    "You are checking one step of visual reasoning.\n"
    "Question: {question}\n"
    "Candidate step: {step}\n"
    "Is this step correct given the image? Answer Yes or No."
)


class Verifier:
    """A frozen VLM prompted as a step critic, plus interface introspection."""

    def __init__(self, model_dir: str, device: str = "cuda", dtype: str = "bfloat16",
                 max_pixels: int | None = None, resample: int | None = None,
                 backend: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_dir = model_dir
        self.device = device
        self.torch = torch
        self.max_pixels = max_pixels
        self.resample = resample
        self.backend = backend

        # The resolution budget is part of the interface, so it is part of the
        # verifier's identity: a certificate is only valid for the budget it was
        # constructed against.  The resampling kernel is the other half of that
        # identity (E20: the certificate is a fact about the kernel), so it is
        # pinned the same way: `resample` is a PIL resampling int (BICUBIC=3,
        # LANCZOS=1) and `backend` selects the image-processor implementation
        # ("pil" | "torchvision").  Both are accepted image-processor kwargs in
        # transformers 5.x; None leaves the checkpoint's own setting in place.
        kw = {"trust_remote_code": True}
        if max_pixels is not None:
            kw["max_pixels"] = max_pixels
        if resample is not None:
            kw["resample"] = resample
        if backend is not None:
            kw["backend"] = backend
        self.processor = AutoProcessor.from_pretrained(model_dir, **kw)
        print(self.describe_image_processor())
        try:
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_dir, torch_dtype=getattr(torch, dtype), trust_remote_code=True
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=getattr(torch, dtype), trust_remote_code=True
            )
        self.model.to(device)
        self.model.eval()  # frozen: no dropout, no grad, no training anywhere

        tok = getattr(self.processor, "tokenizer", self.processor)
        self.yes_ids = self._token_ids(tok, ["Yes", " Yes", "yes", " yes"])
        self.no_ids = self._token_ids(tok, ["No", " No", "no", " no"])

    def describe_image_processor(self) -> str:
        """One parsable line naming the image-processor class and its resampling kernel."""
        ip = getattr(self.processor, "image_processor", None)
        if ip is None:
            return "verifier image_processor=none resample=none"
        rs = getattr(ip, "resample", None)
        try:
            rs = int(rs)
        except (TypeError, ValueError):
            pass
        return (f"verifier image_processor={type(ip).__name__} resample={rs} "
                f"max_pixels={self.max_pixels} backend={self.backend or 'default'}")

    @staticmethod
    def _token_ids(tok, variants: list[str]) -> list[int]:
        """Ids for each surface form of the answer, so scoring is not tokenizer-fragile."""
        ids = []
        for v in variants:
            enc = tok.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        return sorted(set(ids))

    def _inputs(self, image, question: str, step: str):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        text = CRITIC_TEMPLATE.format(question=question, step=step)
        try:  # chat-template path, for processors that expect one
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            prompt = text
        return self.processor(images=image, text=prompt, return_tensors="pt")

    def logits(self, image, question: str, step: str) -> np.ndarray:
        """Decision-position logit vector: a deterministic function of the interface."""
        enc = self._inputs(image, question, step)
        enc = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc)
        return out.logits[0, -1, :].float().cpu().numpy()

    def p_correct(self, image, question: str, step: str) -> float:
        """P(step correct) from the Yes/No logit contrast at the decision position.

        This reads the induced distribution EXACTLY -- it is a deterministic
        function of the interface, with no Monte-Carlo error.  That is the right
        estimator for E0c, and it is why the certificate test resolves to exactly
        zero rather than to within a sampling interval.

        It has a consequence that must not be overlooked when it is used in E1:
        repeated calls are bit-identical, so it supplies NO independent replicate.
        An estimator that needs two independent measurements per item (the
        split-sample routing value) degenerates to the plug-in one under this
        scoring -- measured directly: 80 of 80 items returned identical replicates.
        Use `p_correct_sampled` where genuine replicate variation is required.
        """
        lg = self.logits(image, question, step)
        y = float(np.logaddexp.reduce(lg[self.yes_ids])) if self.yes_ids else -np.inf
        n = float(np.logaddexp.reduce(lg[self.no_ids])) if self.no_ids else -np.inf
        return float(1.0 / (1.0 + np.exp(n - y)))

    def p_correct_sampled(self, image, question: str, step: str, n_samples: int = 8,
                          temperature: float = 1.0, seed: int | None = None) -> dict:
        """Empirical P(correct) from sampled verdicts, plus a Wilson interval.

        Needed wherever the ACTION is genuinely stochastic -- `rea` aggregates
        several critique passes, and self-consistency over critiques is only a
        real procedure if the critiques can differ.  Under deterministic scoring
        those passes collapse to one, and `rea` silently degenerates into a bare
        prompt variant.

        Sampling reintroduces measurement noise, which is exactly the condition
        under which the plug-in routing value is upward biased, so results from
        this path must go through the split-sample estimator.
        """
        lg = self.logits(image, question, step)          # one forward pass, reused
        y = float(np.logaddexp.reduce(lg[self.yes_ids])) if self.yes_ids else -np.inf
        n = float(np.logaddexp.reduce(lg[self.no_ids])) if self.no_ids else -np.inf
        p = 1.0 / (1.0 + np.exp((n - y) / max(temperature, 1e-6)))

        rng = np.random.default_rng(seed)
        draws = rng.random(n_samples) < p
        k = int(draws.sum())
        phat = k / n_samples
        # Wilson interval: the manuscript's stated way of reporting verifier
        # stochasticity, and honest at small n where the normal interval is not.
        z = 1.96
        denom = 1 + z * z / n_samples
        centre = (phat + z * z / (2 * n_samples)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n_samples
                           + z * z / (4 * n_samples ** 2)) / denom
        return {"p_hat": phat, "p_exact": float(p), "k": k, "n": n_samples,
                "wilson_lo": float(centre - half), "wilson_hi": float(centre + half)}

    def critique(self, image, question: str, step: str, max_new_tokens: int = 96,
                 temperature: float = 0.8, seed: int | None = None) -> dict:
        """Generate one critique of the step, by SAMPLING.

        This is what makes `att` and `rea` real actions rather than prompt
        variants.  A critique pass has to be able to come out differently on a
        second run, or aggregating several of them is arithmetic on one number.
        The generated text is then fed back as context for the verdict, so the
        critique actually conditions the judgement.
        """
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        prompt = (
            f"Question: {question}\nCandidate step: {step}\n"
            "Briefly critique this step against the image in one or two sentences."
        )
        enc = self._inputs(image, prompt, step)
        enc = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[-1]) if "input_ids" in enc else 0
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.95,
                pad_token_id=getattr(getattr(self.processor, "tokenizer", None),
                                     "pad_token_id", None),
            )
        gen = out[0][n_in:]
        tok = getattr(self.processor, "tokenizer", self.processor)
        text = tok.decode(gen, skip_special_tokens=True)
        return {"text": text, "prefill_tokens": n_in, "decode_tokens": int(gen.shape[-1])}

    def verdict_after_critique(self, image, question: str, step: str, critique_text: str) -> float:
        """P(correct) with the generated critique in context."""
        q = f"{question}\n\nCritique of the step:\n{critique_text}"
        return self.p_correct(image, q, step)

    def interface_digest(self, image, question: str, step: str) -> str:
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        text = CRITIC_TEMPLATE.format(question=question, step=step)
        return state_digest(capture(self.processor, image, text))

    # ------------------------------------------------------------------
    # E0c: the certificate, tested on a real model
    # ------------------------------------------------------------------

    def pair_test(self, x0: np.ndarray, x1: np.ndarray, question: str, step: str) -> dict:
        """Compare a certified pair's outputs against the hardware's own noise floor.

        Returns both the pair difference and the self difference.  Theorem 1
        predicts pair_max_abs_logit_diff == self_max_abs_logit_diff == 0 for a
        deterministic forward pass; a pair difference materially ABOVE the self
        floor is the signature of an unenumerated image-dependent field, which is
        exactly what experiment E0b is for.
        """
        d0 = self.interface_digest(x0, question, step)
        d1 = self.interface_digest(x1, question, step)

        l0a = self.logits(x0, question, step)
        l0b = self.logits(x0, question, step)   # same input twice -> the floor
        l1 = self.logits(x1, question, step)

        self_diff = float(np.abs(l0a - l0b).max())
        pair_diff = float(np.abs(l0a - l1).max())

        p0, p1 = self.p_correct(x0, question, step), self.p_correct(x1, question, step)
        return {
            "digest_a": d0[:16],
            "digest_b": d1[:16],
            "digests_collide": d0 == d1,
            "self_max_abs_logit_diff": self_diff,
            "pair_max_abs_logit_diff": pair_diff,
            # the honest verdict: identical up to the machine's own reproducibility
            "pair_within_noise_floor": pair_diff <= max(self_diff, 0.0),
            "p_correct_a": p0,
            "p_correct_b": p1,
            # member 0 carries label 1, member 1 carries label 0
            "pair_balanced_accuracy": 0.5 * (p0 + 1.0 - p1),
        }
