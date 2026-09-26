#!/usr/bin/env python3
"""Emit the processor-breadth table body from the E26 records.

One row per deployed pipeline whose processor bundle could be loaded: the image-processor
class the library instantiates, the resize policy recovered by probing (input->output side,
ratio), and the outcome. `certified k/n` counts exact-kernel pairs whose full processor
output digested identically (self-certifying). Everything else is `not decided`, with the
reason: a tiled or patch-geometry policy the single-resize construction does not cover, a
float resize with no fixed-point kernel, a near-identity default policy, a driver that could
not identify the image processor, or a bundle that could not be loaded. The job's own bound
column is NOT used (it predates the sound lattice). Writes generated/tab_breadth_body.tex.
"""
import csv, glob, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
rows = [r for f in sorted(glob.glob(str(root / "results/constructibility/e26-breadth/sh_*.csv"))) for r in csv.DictReader(open(f)) if r.get("processor") and not r["processor"].startswith("#")]
by = {}
for r in rows: by.setdefault(r["processor"], []).append(r)
def outcome(rs):
    cert = [r for r in rs if r["arm"].startswith("exact_kernel")]
    k = sum(int(r["certified"] or 0) for r in cert)
    if cert and k: return f"certified ${k}/{len(cert)}$", "yes"
    e = (rs[0].get("error") or "")
    pol = rs[0].get("policy") or ""
    if "dir missing" in e or "could be loaded" in e: return "not decided: bundle not loadable", "--"
    if (rs[0].get("image_processor_class") or "") == "CLIPTokenizer": return "not decided: processor not identified", "--"
    if "geometry not recognized" in e or "unrecogn" in pol: return "not decided: patch geometry", "--"
    if "no exact fixed-point kernel" in e: return "not decided: float resize", "--"
    if "til" in pol: return "not decided: tiled policy", "--"
    if rs[0].get("ratio") and 1.0 < float(rs[0]["ratio"]) < 1.1: return "not decided: near-identity policy", "--"
    return "not decided", "--"
order = ["google/siglip-base-patch16-384","google/siglip-so400m-patch14-384","Salesforce/blip2-opt-2.7b","mistral-community/pixtral-12b",
         "HuggingFaceTB/SmolVLM-Instruct","HuggingFaceM4/idefics2-8b","Qwen/Qwen2-VL-7B-Instruct","microsoft/Phi-3.5-vision-instruct",
         "adept/fuyu-8b","openai/clip-vit-large-patch14-336","allenai/Molmo-7B-D-0924","google/paligemma-3b-pt-224","openbmb/MiniCPM-V-2_6",
         "llava-hf/llava-v1.6-mistral-7b-hf","OpenGVLab/InternVL2-8B"]
out = ["\\begin{tabular}{@{}llll@{}}", "\\toprule",
       "Pipeline & Image processor & Resize found & Outcome \\\\", "\\midrule"]
n_cert = 0
unloaded = []          # bundles the driver never got a processor out of: one line, not a row
REASON = {"google/paligemma-3b-pt-224": "gated", "openbmb/MiniCPM-V-2_6": "gated",
          "llava-hf/llava-v1.6-mistral-7b-hf": "bundle not fetched", "OpenGVLab/InternVL2-8B": "bundle not fetched",
          "allenai/Molmo-7B-D-0924": "remote code failed to import",
          "openai/clip-vit-large-patch14-336": "the driver did not identify its image processor"}
for p in order + [q for q in by if q not in order]:
    if p not in by or p == "llava-hf/llava-1.5-7b-hf":      # certified in Table 1 already
        continue
    rs = by[p]; r0 = rs[0]; o, c = outcome(rs)
    cls = (r0.get("image_processor_class") or "").replace("_", "\\_")
    if cls in ("", "--", "CLIPTokenizer") or "not loadable" in o:
        unloaded.append((p, REASON.get(p, "not loadable")))
        continue
    n_cert += (c == "yes")
    res = f"${r0['n_in']}$ to ${r0['n_out']}$ (${float(r0['ratio']):.2f}\\times$)" if r0.get("n_in") and r0.get("ratio") else "--"
    # The org prefix and the ImageProcessor suffix are constant boilerplate that pushed the
    # table to 510pt against a 397pt text width, shrinking it to about 7pt. The caption says
    # both are elided.
    short_p = p.split("/", 1)[1] if "/" in p else p
    short_cls = cls[:-len("ImageProcessor")] if cls.endswith("ImageProcessor") else cls
    out.append(f"\\texttt{{{short_p.replace('_', chr(92)+'_')}}} & {short_cls} & {res} & {o} \\\\")
# Bundles the driver never got a processor from are not rows: the text before the table says
# they were not attempted, and a list of names under a rule is not a result.
out += ["\\bottomrule", "\\end{tabular}"]
(root / "analysis" / "output").mkdir(exist_ok=True)
(root / "analysis" / "output" / "tab_breadth_body.tex").write_text("\n".join(out) + "\n")
print("wrote analysis/output/tab_breadth_body.tex; certified pipelines:", n_cert)
