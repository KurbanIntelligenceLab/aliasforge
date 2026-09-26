#!/usr/bin/env python3
"""The item stream E1 runs on: (image, question, candidate step, correctness label).

Source is VisualProcessBench, the human-annotated step-correctness benchmark released
alongside VisualPRM, which the manuscript already commits to.  Each record carries a
question, a policy model's response split into `steps`, and `process_correctness` with
one label per step, so an item is one (record, step index) pair.

Two decisions worth stating, because they shape what E1 can conclude.

First, the label alphabet.  `process_correctness` uses 1 for a correct step, -1 for an
incorrect one, and 0 for neutral/unjudgeable.  Neutral steps are DROPPED rather than
folded into either class: a gain computed against an unjudgeable label is not a gain,
and keeping them would quietly inflate the denominator of every rate we report.

Second, the prefix.  A step is only meaningful given the steps before it, so each item
carries the preceding steps as context.  Scoring step k in isolation would ask the
verifier a different question from the one the benchmark labelled.
"""

from __future__ import annotations

import json
import pathlib
import zipfile
from dataclasses import dataclass

import numpy as np


@dataclass
class Item:
    key: str            # stable id: record index + step index
    image_path: str     # path inside the image archive
    question: str
    prefix: str         # steps before this one, joined
    step: str           # the candidate step being judged
    label: int          # 1 correct, 0 incorrect
    source: str         # data_source, used as the clustering unit
    policy_model: str

    @property
    def conditioned_question(self) -> str:
        """Question plus the reasoning prefix, which is what the label is relative to."""
        if not self.prefix:
            return self.question
        return f"{self.question}\n\nReasoning so far:\n{self.prefix}"


def load_items(root: str, limit: int | None = None, single_image_only: bool = True,
               seed: int = 0) -> list[Item]:
    """Parse test.jsonl into per-step items.

    `single_image_only` keeps records with exactly one image: multi-image records make
    "the interface" ambiguous (which image does a crop act on?), and mixing them in
    would confound the action definitions rather than broaden coverage.
    """
    p = pathlib.Path(root) / "test.jsonl"
    items: list[Item] = []
    with open(p) as fh:
        for ri, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            imgs = rec.get("image") or []
            if single_image_only and len(imgs) != 1:
                continue
            resp = rec.get("response") or {}
            steps = resp.get("steps") or []
            labels = resp.get("process_correctness") or []
            if len(steps) != len(labels):
                continue
            for si, (st, lb) in enumerate(zip(steps, labels)):
                if lb == 0:            # neutral / unjudgeable -> dropped, see docstring
                    continue
                items.append(Item(
                    key=f"{ri}:{si}",
                    image_path=imgs[0],
                    question=rec.get("question", ""),
                    prefix="\n".join(steps[:si]),
                    step=st,
                    label=1 if lb == 1 else 0,
                    source=rec.get("data_source", "unknown"),
                    policy_model=rec.get("policy_model", "unknown"),
                ))
    if limit is not None and len(items) > limit:
        # Sample without replacement so the subset is not the first N records, which
        # would be one data_source and would make the clustered bootstrap meaningless.
        idx = np.random.default_rng(seed).permutation(len(items))[:limit]
        items = [items[i] for i in sorted(idx)]
    return items


class ImageStore:
    """Reads images straight out of the shipped zip, so nothing is unpacked twice."""

    def __init__(self, root: str):
        self.zip_path = pathlib.Path(root) / "images.zip"
        self._zf: zipfile.ZipFile | None = None

    def _z(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def names(self) -> list[str]:
        return self._z().namelist()

    def load(self, path: str):
        from PIL import Image
        import io

        z = self._z()
        try:
            blob = z.read(path)
        except KeyError:
            # archives sometimes carry a leading directory component
            cand = [n for n in z.namelist() if n.endswith(path)]
            if not cand:
                raise
            blob = z.read(cand[0])
        return Image.open(io.BytesIO(blob)).convert("RGB")


def summarize(items: list[Item]) -> dict:
    from collections import Counter

    src = Counter(i.source for i in items)
    return {
        "n_items": len(items),
        "n_correct": sum(i.label == 1 for i in items),
        "n_incorrect": sum(i.label == 0 for i in items),
        "n_sources": len(src),
        "top_sources": src.most_common(6),
        "n_records": len({i.key.split(":")[0] for i in items}),
    }
