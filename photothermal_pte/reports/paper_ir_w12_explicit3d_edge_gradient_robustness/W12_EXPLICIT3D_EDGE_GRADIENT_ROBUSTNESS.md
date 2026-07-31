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
| mask_aware_finite_difference | 0.879613 | 0.879578 | 0.886038 | 0.890100 |
| least_squares_radius_0.2um | 0.783776 | 0.783960 | 0.786059 | 0.789529 |
| least_squares_radius_0.3um | 0.662673 | 0.662635 | 0.664870 | 0.667748 |
| least_squares_radius_0.4um | 0.582121 | 0.579637 | 0.581913 | 0.584554 |

## Maxwell thickness-average, fixed 0.1–0.3 µm inside band

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
| mask_aware_finite_difference | 0.602675 | 0.602577 | 0.493581 | 0.478654 |
| least_squares_radius_0.2um | 0.631621 | 0.628491 | 0.507391 | 0.494561 |
| least_squares_radius_0.3um | 0.532583 | 0.525785 | 0.480659 | 0.478552 |
| least_squares_radius_0.4um | 0.487761 | 0.484392 | 0.450075 | 0.450362 |

## Analytic thickness-average, original staircase edge

| reconstruction | raw-max b/a | p99 b/a | RMS b/a | mean b/a |
|---|---:|---:|---:|---:|
| mask_aware_finite_difference | 1.475105 | 1.475100 | 1.473830 | 1.471925 |
| least_squares_radius_0.2um | 1.476839 | 1.476843 | 1.476289 | 1.474289 |
| least_squares_radius_0.3um | 1.478385 | 1.478659 | 1.478698 | 1.476983 |
| least_squares_radius_0.4um | 1.479485 | 1.479482 | 1.480159 | 1.478463 |

The JSON additionally retains surface and midplane results, individual fixed
normal bands at 0.1, 0.2, and 0.3 µm, least-squares radius sensitivity,
neighbour counts, fit condition numbers, field NRMSE, and correlations.

All new a/b and finite-difference/least-squares plots use one common color
scale.  No raw single-cell maximum is promoted without p99, RMS, and mean.

## Provenance

- input: `/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_50nm_maxwell_analytic_explicit3d_20260731/w12_50nm_maxwell_analytic_explicit3d_fields.npz`
- SHA-256: `9b2287b5b18eb9c4d9c164ddd45d750ae05ff846d6a0f0e3936465e413be47ac`
- JSON: `w12_edge_gradient_robustness_summary.json`
- CSV: `w12_edge_gradient_robustness_cases.csv`
- command: `/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/validation/paper_ir_sanity/audit_w12_explicit3d_edge_gradient_robustness.py --input-npz /data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_50nm_maxwell_analytic_explicit3d_20260731/w12_50nm_maxwell_analytic_explicit3d_fields.npz --report-dir photothermal_pte/reports/paper_ir_w12_explicit3d_edge_gradient_robustness`
