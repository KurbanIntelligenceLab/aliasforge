#!/usr/bin/env python3
"""Second verifier: does the certificate transfer to a different architecture family?

Generality has been the paper's weakest structural point -- one model, and the other
candidate provably excluded. This closes it using the prediction the dyadic result
makes: a FIXED-resolution tower resizes every input to one side S, so feeding 2S is an
exact 2x downscale, which is dyadic and therefore constructible. LLaVA-1.5 uses
CLIP-L/336, so 672 -> 336.

Nothing about the construction changes -- same integer kernel, same verification. That
is the point: if the theory is right, a different model family should need no new ideas.

Reports the same three arms as E0c on the first verifier: certified pair, non-kernel
control, and the same-image self difference that is the hardware noise floor.
"""
from __future__ import annotations
import argparse, csv, os, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from aliasforge.exact import pil_coeffs, find_short_vector, make_exact_pair  # noqa: E402
from aliasforge.interface import capture, per_field_report, state_digest  # noqa: E402

SIDE, TARGET = 672, 336
Q = "How many horizontal marks appear in the image?"
STEP = "The image contains exactly three horizontal marks."

def masks(n):
    m = np.zeros(SIDE, bool)
    for i, y0 in enumerate((int(.13*SIDE), int(.38*SIDE), int(.63*SIDE))):
        if i < n: m[y0:y0+int(.12*SIDE)] = True
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT",""))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image

    proc = AutoProcessor.from_pretrained(a.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.float16).to("cuda").eval()
    print(f"exp=e0_llava side={SIDE} target={TARGET} factor={SIDE/TARGET:.2f} seed={a.seed}")

    K = pil_coeffs(SIDE, TARGET)
    vec, meta = find_short_vector(K, width=64, stride=16, max_abs=8)
    if vec is None:
        print("RESULT no kernel vector"); return 1
    print(f"kernel_vector max_abs={int(np.abs(vec).max())} nonzeros={int((vec!=0).sum())}")

    def logits(img):
        prompt = f"USER: <image>\n{Q}\nCandidate step: {STEP}\nIs this step correct? ASSISTANT:"
        enc = proc(images=Image.fromarray(img), text=prompt, return_tensors="pt")
        enc = {k: (v.to("cuda") if hasattr(v,"to") else v) for k,v in enc.items()}
        with torch.no_grad(): out = model(**enc)
        return out.logits[0,-1,:].float().cpu().numpy()

    m3, m2 = masks(3), masks(2)
    rows = []
    for t in range(a.trials):
        rng = np.random.default_rng(a.seed*100+t)
        base = np.clip(128+18*np.sin(np.mgrid[0:SIDE,0:SIDE][1]/11.)[...,None]
                       + rng.integers(-6,7,(SIDE,SIDE,3)),0,255).astype(np.uint8)
        for arm in ("certified","control"):
            if arm == "certified":
                x0,x1,mm = make_exact_pair(base, vec, 20, m3, m2)
            else:
                off = np.zeros_like(vec); nz = np.flatnonzero(vec)
                off[nz] = rng.integers(1,4,size=nz.size)
                x0,x1,mm = make_exact_pair(base, off, 20, m3, m2)
            if x0 is None or mm.get("pix_diff",0)==0: continue
            sa = capture(proc, Image.fromarray(x0), Q); sb = capture(proc, Image.fromarray(x1), Q)
            rep = per_field_report(sa, sb)
            collide = state_digest(sa)==state_digest(sb)
            l0a, l0b, l1 = logits(x0), logits(x0), logits(x1)
            self_d = float(np.abs(l0a-l0b).max()); pair_d = float(np.abs(l0a-l1).max())
            rows.append(dict(model="llava-1.5-7b", trial=t, arm=arm,
                             digests_collide=int(collide), self_diff=self_d,
                             pair_diff=pair_d, at_floor=int(pair_d<=max(self_d,0.0)),
                             pix_diff=mm["pix_diff"], fields_differ=",".join(rep["differ"]) or "none"))
            print(f"trial={t} arm={arm} collide={int(collide)} self={self_d:.3e} "
                  f"pair={pair_d:.3e} at_floor={int(pair_d<=max(self_d,0.))} "
                  f"fields_differ={','.join(rep['differ']) or 'none'}")
    if a.out and rows:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out,"w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    cert=[r for r in rows if r["arm"]=="certified"]; ctrl=[r for r in rows if r["arm"]=="control"]
    print(f"RESULT model=llava-1.5-7b certified={sum(r['digests_collide'] for r in cert)}/{len(cert)} "
          f"at_floor={sum(r['at_floor'] for r in cert)}/{len(cert)} "
          f"controls_separated={sum(1 for r in ctrl if r['pair_diff']>max(r['self_diff'],1e-12)*10)}/{len(ctrl)}")
    return 0

if __name__ == "__main__": sys.exit(main())
