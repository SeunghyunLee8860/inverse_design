# W12 explicit-3D edge-gradient robustness audit

Status: `COMPLETED_OFFLINE_W12_EXPLICIT3D_EDGE_GRADIENT_ROBUSTNESS_AUDIT`

This audit used only the stored explicit-3D temperature artifact.  It did not
run FDTD or a thermal solve.

## Provenance correction

The published `0.879613` is reproduced from the same explicit-3D artifact by
selecting the original 142-cell staircase edge and taking the ratio of the
separate a/b raw maxima.  The separate `0.798934` audit value used a broader
0.5-µm inside-edge band.  They are different spatial comparators, not
different optical checkpoints.

## Maxwell thickness-average, original staircase edge

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
| mask_aware_finite_difference | 0.914024 | 0.913987 | 0.915278 | 0.912302 |
| least_squares_radius_0.2um | 0.700540 | 0.700296 | 0.701057 | 0.709755 |
| least_squares_radius_0.3um | 0.578599 | 0.578458 | 0.577563 | 0.588843 |
| least_squares_radius_0.4um | 0.490422 | 0.490251 | 0.489693 | 0.502914 |

## Maxwell thickness-average, fixed 0.1–0.3 µm inside band

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
| mask_aware_finite_difference | 0.613217 | 0.609877 | 0.450773 | 0.445356 |
| least_squares_radius_0.2um | 0.565487 | 0.559117 | 0.448121 | 0.450121 |
| least_squares_radius_0.3um | 0.476411 | 0.476141 | 0.407116 | 0.417550 |
| least_squares_radius_0.4um | 0.406869 | 0.405883 | 0.361910 | 0.375104 |

## Analytic thickness-average, original staircase edge

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
| mask_aware_finite_difference | 1.472461 | 1.472397 | 1.470335 | 1.469263 |
| least_squares_radius_0.2um | 1.476797 | 1.476810 | 1.476244 | 1.474340 |
| least_squares_radius_0.3um | 1.478518 | 1.478485 | 1.478698 | 1.476932 |
| least_squares_radius_0.4um | 1.479630 | 1.479523 | 1.480244 | 1.478717 |

The JSON additionally retains surface and midplane results, individual fixed
normal bands at 0.1, 0.2, and 0.3 µm, least-squares radius sensitivity,
neighbour counts, fit condition numbers, field NRMSE, and correlations.

All new a/b and finite-difference/least-squares plots use one common color
scale.  No raw single-cell maximum is promoted without p99, RMS, and mean.

## Provenance

- input: `/home/seunghyun/tairte4_artifacts/paper_ir_w12_explicit3d_thermal50_20260731/w12_50nm_maxwell_analytic_explicit3d_fields.npz`
- SHA-256: `399914ae58a3051b5405ccb24d44ee073c810e85302695b29f82633a5c882c2e`
- JSON: `w12_edge_gradient_robustness_summary.json`
- CSV: `w12_edge_gradient_robustness_cases.csv`
- command: `/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/validation/paper_ir_sanity/audit_w12_explicit3d_edge_gradient_robustness.py --input-npz /home/seunghyun/tairte4_artifacts/paper_ir_w12_explicit3d_thermal50_20260731/w12_50nm_maxwell_analytic_explicit3d_fields.npz --report-dir photothermal_pte/reports/paper_ir_w12_explicit3d_thermal50_gradient_robustness`
