# Device-A true-edge scan — corrected beam placement (2026-08-02)

## What this is

Full 7-position beam scan across the **true digitized flake boundary** of Device A,
both polarizations (E∥a on GPU 4, E∥b on GPU 3), with every correction from the
position audit applied:

- Scan positions defined by distance `d` from the actual polygon boundary crossing
  along the scan direction (pre-run check: all 7 positions PASS against the real
  boundary; `scan_true_positions.json`).
- Beam: Gaussian, scenario waist 8.75 µm (source waist parameter 12 µm), source
  span 32 µm (truncated-Gaussian capture 0.9995, clears the PML gate at d=−2 µm).
- Production optical contract: palik-lossy SiO2, paper-b-closure ε_c, PML 24,
  flake dz 10 nm, auto shutoff 1e-5, per-polarization empty-domain incident
  reference.
- Thermal/PTE: expanded conservative Cartesian FVM, 60 µm domain, Si depth 20 µm,
  material-overlap Q remap, metal `isolated-lower-bound`, 285 µW incident
  normalization.

Optical/thermal artifacts: `/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end/edgetrue_*_20260802/`.
Figures: `edgetrue_paper_figures_20260802/` (copied here).

## Results

| d (µm) | I_a (nA) | I_b (nA) | \|I_a\|/\|I_b\| |
|---:|---:|---:|---:|
| −2 | +18.73 | +15.41 | 1.215 |
| −1 | +21.12 | +17.59 | 1.201 |
|  0 | +22.60 | +19.00 | **1.190** |
| +1 | +22.29 | +18.79 | 1.187 |
| +2 | +20.10 | +16.72 | 1.202 |
| +3 | +16.30 | +13.18 | 1.238 |
| +5 |  +6.72 |  +3.76 | 1.787 |

(+ = into flake; paper edge reference ratio 0.8366.)

### Findings

1. **Edge-peaked profile reproduced.** Both polarizations peak at the true edge
   (d≈0) and fall to ~30% (a) / ~20% (b) of peak 5 µm inside — qualitatively the
   paper Fig. 3I shape.
2. **Sign is internally consistent.** All 14 currents share one sign under our
   fixed weighting-potential polarity (top contact ψ=1). The paper's negative
   values correspond to the opposite electrode/amplifier polarity convention; the
   a/b relative sign (same edge → same sign) matches, which is the
   convention-invariant check.
3. **Ratio stays >1 (a stronger), paper says <1 at the edge.** With corrected
   positions the edge ratio drops from the earlier audit's 1.62 to **1.19**, but
   the ordering |I_a|>|I_b| persists at every position. Minimum ≈1.19 at d=0…+1,
   rising to 1.79 at d=+5 (approaching the bulk-like value seen in the earlier
   deep-inside audit). Consistent with the prior audit conclusion: not a pipeline
   bug; remaining candidates are the E∥a edge-hotspot physics of the digitized
   (approximate) geometry and possible metric/position definition differences in
   the paper.
4. **Beam realization verified post-hoc.** 2-D Gaussian fit of the empty-run
   |E|² at the flake plane: center error 0.003 µm (a) / 0.016 µm (b) vs intent;
   realized waist 9.0 µm (⊥ pol.) / 10.3 µm (∥ pol.) vs nominal 8.75 µm —
   vector-focusing broadening, symmetric between polarizations (fit residual
   <0.4%).

## Operational notes

Runs were serialized behind two gates after repeated failures: a license gate
(each GPU engine takes 9 of the shared 54 `lum_fdtd_solve` seats; the
"could not match resource name" error is the licence-starvation disguise) and a
root-disk gate (root fs repeatedly hit 0 B free from concurrent campaigns; ≥3 GB
required before launch). Per-polarization isolated `XDG_CONFIG_HOME` avoids the
Lumerical shared-settings race.

## Figures

- `fig2G_style_psi_streamlines.png` — weighting potential ψ contours and −∇ψ
  streamlines over the digitized device outline.
- `fig3F_style_dT_maps.png` — flake-average ΔT maps, beam at the true edge, both
  polarizations.
- `fig3I_style_current_and_ratio.png` — I(d) for both polarizations and the
  ratio profile vs the paper edge reference.
- `scan_results_table.json` — numeric table incl. beam readback.
