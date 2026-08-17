# Final exact-binary eight-case PTE matrix

Status: **COMPLETED_EXACT_BINARY_EIGHT_CASE_PHYSICS_MATRIX_WITH_OBJECTIVE_PRESERVATION_DIAGNOSTICS**

## Outcome

Runs 044, 045, 046, 048, 055, 056, 057, and 058 form the complete 2×2×2 matrix: top/bottom versus left/right electrodes, thermally-grown versus evaporated TaIrTe4/SiO2 interface, and `Ea` versus `Eb` illumination. Every promoted structure is exactly 0/1 and has **zero** bad nodes in the requested discrete 500 nm solid-and-void opening audit.

The table reports the fresh physical evaluation of the forced exact structure, even when its current is lower than the continuous checkpoint. The old 1% objective-preservation gate is not rewritten: it fails for Run 044, Run 046, Run 055, Run 056, Run 057, Run 058. This does not invalidate their Maxwell/thermal/electrical solution; it records the performance cost of enforcing the final geometry.

| Run | electrodes | interface | pol. | 500 nm bad | P_Q @285 µW (µW) | Tmax rise (K) | continuous I (nA) | exact I (nA) | change | physical gates |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 044 | top-bottom | thermally_grown | Ea | 0 | 59.3168 | 0.38982 | 91.7591 | 90.0131 | -1.903% | True |
| 045 | top-bottom | thermally_grown | Eb | 0 | 60.1323 | 0.19087 | 56.3687 | 56.3588 | -0.018% | True |
| 046 | left-right | thermally_grown | Ea | 0 | 43.4844 | 0.25232 | 20.9228 | 19.6915 | -5.885% | True |
| 048 | left-right | thermally_grown | Eb | 0 | 54.4750 | 0.19746 | 22.5407 | 22.6064 | 0.291% | True |
| 055 | top-bottom | evaporated | Ea | 0 | 51.3149 | 2.9234 | 807.0132 | 774.5461 | -4.023% | True |
| 056 | top-bottom | evaporated | Eb | 0 | 56.3333 | 3.1119 | 925.2011 | 893.6255 | -3.413% | True |
| 057 | left-right | evaporated | Ea | 0 | 56.5988 | 2.814 | 314.8472 | 297.5015 | -5.509% | True |
| 058 | left-right | evaporated | Eb | 0 | 52.8068 | 2.8034 | 322.1210 | 312.0541 | -3.125% | True |

## What was calculated

- Optical: each exact candidate has its own fresh v261 GPU Maxwell forward solution. `P_Q`, six-face power, closure, and the conservative volumetric `Q(x,y,z)` mapped to the explicit 3-D thermal grid are retained.
- Thermal: the stored CUDA solution uses explicit air, 285 nm SiO2, and 20 µm Si, anisotropic TaIrTe4 `k=(3.8,14.4,1.0) W/(m K)` in Lumerical `(x=b,y=a,z=c)`, finite TaIrTe4/SiO2 G, and SiO2/Si `G=1.1e9 W/(m² K)`.
- Electrical: terminal current is the full triangular-FEM integral over the 100 nm TaIrTe4 sheet, not a single point and not a strict-gradient proxy. The plotted strict-centered gradients are diagnostic maps only; a node is NaN unless all `±x` and `±y` solid neighbours exist.
- Current decomposition: `b/x`, `a/y`, positive spatial, and negative spatial contributions are independently reintegrated from the stored temperature and weighting fields. Their total agrees with the certified terminal current to roundoff.

## Interface provenance

Runs 055–058 embed the evaporated interface contract (`G=7.37e4 W/(m² K)`) directly in their raw JSON. Runs 044–048 are legacy artifacts created before this metadata field was added: their thermally-grown scenario (`G=7.37e6 W/(m² K)`) follows the then-default `thermal.py` execution contract and launch environment, but is **not** represented as newly embedded raw metadata. The legacy raw JSON files are unchanged.

## Scaling and integrity

Raw FDTD excitation powers differ slightly by exact geometry. Values labelled “at 285 µW” apply the linear factor `285e-6/source_power_W`, exactly as in the original objective certificates. This is physical linear-response normalization, not an empirical fit. No Q clipping, smoothing, gain, global rescaling, tiling, or source deletion is used. Raw NPZ/FSP/H5 files remain outside Git; their paths, sizes, and SHA-256 values are in `RAW_ARTIFACT_MANIFEST.json`.

## Figure guide

- `final_exact_binary_structures.png`: all eight exact structures on common Lumerical axes.
- `final_exact_binary_physics_metrics.png`: exact versus continuous current, performance cost, absorbed power, and peak temperature.
- `final_exact_binary_current_components.png`: axis and signed-spatial current decomposition.
- `final_exact_binary_pair_comparisons.png`: paired Ea/Eb optimized-run outcomes. Because each bar uses a different optimized structure, this is not a same-device polarization-selectivity measurement.
- `run*_fields.png`: per-case exact density, Q, temperature, strict gradients, weighting potential, and total/component current maps.
- `run*_profiles.png`: central Q, temperature/gradient, and current profiles.
- `run*_decomposition.png`: absorption versus depth and integrated current terms.

This report does not claim that the forced repair is the optimum of a new binary combinatorial optimization. It is the requested physically re-evaluated, manufacturability-clean final structure for each completed run.
