# Records

The per-item records behind every number in the paper, grouped like `experiments/`. Shards are CSV
files, one per task. The two router-feature experiments, `routing/e34-strong-probe` and
`routing/e35-signal-zqr`, store their features as NumPy archives (`.npz`). Table, theorem and
appendix numbers refer to the arXiv version of the paper.

| Records | Script | Paper |
|---|---|---|
| [`certification/e0a-exact`](certification/e0a-exact) | [`e0a_exact.sh`](../experiments/certification/e0a_exact.sh) | Table 1, Qwen2.5-VL-7B row; Appendix E |
| [`certification/e0-llava`](certification/e0-llava) | [`e0_llava.sh`](../experiments/certification/e0_llava.sh) | Table 1, LLaVA-1.5-7B row; Appendix E |
| [`certification/e0c-exact`](certification/e0c-exact) | [`e0c_exact.sh`](../experiments/certification/e0c_exact.sh) | Table 1, logit gap and pair-balanced accuracy |
| [`certification/e28-visualprm`](certification/e28-visualprm) | [`e28_visualprm.sh`](../experiments/certification/e28_visualprm.sh) | Table 1, VisualPRM-8B row: the stacked-kernel pair on the tiling path |
| [`certification/e28b-visualprm-logit`](certification/e28b-visualprm-logit) | [`e28b_visualprm_logit.sh`](../experiments/certification/e28b_visualprm_logit.sh) | Appendix E, VisualPRM-8B logit read |
| [`certification/e28c-visualprm-pairs`](certification/e28c-visualprm-pairs) | [`e28c_visualprm_pairs.sh`](../experiments/certification/e28c_visualprm_pairs.sh) | Table 1, VisualPRM-8B row: twenty pairs and twenty controls |
| [`certification/e26-fields`](certification/e26-fields) | [`e26_fields.sh`](../experiments/certification/e26_fields.sh) | Table A2, the hashed interface fields |
| [`constructibility/e20b-bounds`](constructibility/e20b-bounds) | [`e20b_bounds.sh`](../experiments/constructibility/e20b_bounds.sh) | Table A3, constructibility by resampling filter and ratio |
| [`constructibility/e10b-screen`](constructibility/e10b-screen) | [`e10b_screen.sh`](../experiments/constructibility/e10b_screen.sh) | Table A4, the screen of 20 declared configurations |
| [`constructibility/e26-breadth`](constructibility/e26-breadth) | [`e26_breadth.sh`](../experiments/constructibility/e26_breadth.sh) | Table A5, breadth across deployed processors |
| [`constructibility/e33-nearkernel`](constructibility/e33-nearkernel) | [`e33_nearkernel.sh`](../experiments/constructibility/e33_nearkernel.sh) | Appendix A, remark on output quantization: supporting counts of near-kernel collisions |
| [`constructibility/e27-kernelswap`](constructibility/e27-kernelswap) | [`e27_kernelswap.sh`](../experiments/constructibility/e27_kernelswap.sh) | Ethics statement: supporting records for the resampling mitigation (Lanczos kernel swap) |
| [`natural_images/e16-natural`](natural_images/e16-natural) | [`e16_natural.sh`](../experiments/natural_images/e16_natural.sh) | Appendix D, solid-band construction on natural images |
| [`natural_images/e18-freeplace`](natural_images/e18-freeplace) | [`e18_freeplace.sh`](../experiments/natural_images/e18_freeplace.sh) | Appendix D, free placement on natural images |
| [`natural_images/e23-legible`](natural_images/e23-legible) | [`e23_legible.sh`](../experiments/natural_images/e23_legible.sh) | Figure 2 and Appendix D, legibility-constrained placement |
| [`natural_images/e24-native`](natural_images/e24-native) | [`e24_native.sh`](../experiments/natural_images/e24_native.sh) | Appendix D, native-size eligibility |
| [`natural_images/e25-padded`](natural_images/e25-padded) | [`e25_padded.sh`](../experiments/natural_images/e25_padded.sh) | Figure 2 and Appendix D, unresampled placement on the canvas |
| [`routing/e21-pathmatched`](routing/e21-pathmatched) | [`e21_pathmatched.sh`](../experiments/routing/e21_pathmatched.sh) | Section 7 and Tables 2, A8, A10, A11: per-action gains against the path-matched baseline |
| [`routing/e1-full`](routing/e1-full) | [`e1_full.sh`](../experiments/routing/e1_full.sh) | Appendix J, gains against the unprompted baseline |
| [`routing/e19-placebo`](routing/e19-placebo) | [`e19_placebo.sh`](../experiments/routing/e19_placebo.sh) | Appendix J, cost of the prompt path |
| [`routing/e12-cost`](routing/e12-cost) | [`e12_cost.sh`](../experiments/routing/e12_cost.sh) | Table A6 and Appendix F, the measured cost vector under three cost measures |
| [`routing/e14-crop2`](routing/e14-crop2) | [`e14_crop2.sh`](../experiments/routing/e14_crop2.sh) | Section 7 and Appendix J, the ten-crop oracle |
| [`routing/e2-probe`](routing/e2-probe) | [`e2_probe.sh`](../experiments/routing/e2_probe.sh) | Section 7, probe controls and the certified anchor |
| [`routing/e3-pathmatched`](routing/e3-pathmatched) | [`e3_ias.sh`](../experiments/routing/e3_ias.sh) | Table 2, label-free score rows |
| [`routing/e31b-screen-actions`](routing/e31b-screen-actions) | [`e31b_screen_actions.sh`](../experiments/routing/e31b_screen_actions.sh) | Table 2 dominated-action column; Appendix E |
| [`routing/e32-fill-table2`](routing/e32-fill-table2) | [`e32_fill_table2.sh`](../experiments/routing/e32_fill_table2.sh) | Table 2, raw-distance row and pixel-feature routers |
| [`routing/e34-strong-probe`](routing/e34-strong-probe) | [`e34_strong_probe.sh`](../experiments/routing/e34_strong_probe.sh) | Table A9, vision-tower features of every natural item |
| [`routing/e34b-probe-pairs`](routing/e34b-probe-pairs) | [`e34b_probe_pairs.sh`](../experiments/routing/e34b_probe_pairs.sh) | Table A9, router actions on the certified pairs |
| [`routing/e35-signal-zqr`](routing/e35-signal-zqr) | [`e35_signal_zqr.sh`](../experiments/routing/e35_signal_zqr.sh) | Tables 2 and A9, the verifier's decision-position hidden state on every natural item and certified pair |
| [`routing/power-e1`](routing/power-e1) | [`power_e1.sh`](../experiments/routing/power_e1.sh) | Table A7, power of the routing gate |
