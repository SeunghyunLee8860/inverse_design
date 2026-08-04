# Straight 45° edge optical/thermal control

Status: `FAILED_STRAIGHT_45_EDGE_PAPER_GRADIENT_TREND`

## Outcome

The polygon corner was removed. The calculation used a single corner-free
TaIrTe4/air boundary `lab y=x`, with TaIrTe4 in `y<=x`, normal-incidence
11 µm Gaussian illumination centred on the edge, and independent `E||a` and
`E||b` v261 GPU FDTD runs. The existing conservative remap and explicit
anisotropic/multimaterial thermal FVM were then applied.

The optical and numerical-conservation gates pass. The requested paper trend
does **not**: `E||b` absorbs more total power and has a larger flake-average
temperature, but `E||a` has the larger peak temperature and edge-normal
temperature gradient on both meshes. Therefore no weighting field or PTE
current was evaluated.

## Geometry and illumination contract

- TaIrTe4: 130 nm, half-plane `y<=x`; no corner in the physical domain.
- Lab `x=b`, lab `y=a`; edge outward normal `(-x+y)/sqrt(2)`.
- 285 nm SiO2 on Si; optical electrodes omitted for this isolated edge check.
- Wavelength 11 µm, `w0=6.5 µm`, beam centre `(0,0)`, 285 µW incident power.
- `w0=6.5 µm` is a named scenario, not a paper-extracted exact beam radius.
- Optical domain: 48×48 µm, six PML faces, 24 PML layers, 10 nm flake mesh.
- Thermal domain: 48×48 µm, 20 µm Si depth; 200 and 100 nm lateral meshes.

## GPU optical result

| metric | E||a | E||b | b/a |
|---|---:|---:|---:|
| `P_Q` at central 1 W/m² | 1.174688593e-11 W | 1.454936809e-11 W | 1.238572 |
| `P_six` | 1.177632646e-11 W | 1.448968121e-11 W | — |
| six-face closure | 0.249998% | 0.411927% | — |
| `P_abs` at 285 µW | 30.063514 µW | 37.235833 µW | 1.238572 |

Both closures are below 0.5%, both raw Q fields have zero negative voxels,
and the analytic exact-flake mask exactly equals `y<=x` over the 130-nm
thickness. No clipping, smoothing, gain, global rescaling, tiling, or source
deletion was used.

The external optical case JSON records the pre-checkpoint HEAD because these
runs were launched from the working tree. The identical straight-edge
execution-source content is frozen by report generation commit `828207e257f69b9f2e9a3306ce5b7f9abdf16efa`;
the raw metadata was preserved rather than rewritten.

## Thermal result

| 100 nm metric | E||a | E||b | b/a |
|---|---:|---:|---:|
| flake `Tmax` | 0.239720817 K | 0.228104561 K | 0.951543 |
| flake-average ΔT | 0.034194044 K | 0.042289870 K | 1.236761 |
| max `|∂T/∂n|` | 3.597186e+04 K/m | 2.902218e+04 K/m | 0.806802 |
| p99 `|∂T/∂n|` | 3.590310e+04 K/m | 2.900695e+04 K/m | 0.807923 |

The 200-nm mesh gives the same qualitative reversal: max-gradient
`b/a=0.753767` and p99-gradient
`b/a=0.753090`. Peak gradient
magnitudes are not mesh converged, so they are not promoted as quantitative
experimental predictions; the polarization ordering is nevertheless
unchanged by refinement.

All four thermal cases have mapping error below 0.5%, energy-balance error
below 1%, and linear residual below 1e-8.

## Interpretation and next gate

Removing the approximate polygon corner did not restore the Figure-3F trend.
The present failure is driven by source localization: the `E||a` Q field is
more concentrated near the edge, while `E||b` deposits more total but broader
power. This result rules out the old concave corner as the sole explanation,
but it does not identify a unique remaining cause.

Before PTE, the discriminating follow-ups are optical-Q profile comparison
against the paper's exact simulation contract (especially the unreported
beam radius/spot definition and material-axis convention) and a converged
edge-gradient estimator. Weighting-field changes cannot fix this pre-weighting
thermal trend and were intentionally not used.

## Files

- `STRAIGHT_45_EDGE_OPTICAL_THERMAL_CONTROL.png`
- `STRAIGHT_45_EDGE_MESH_TREND.png`
- `straight_45_edge_summary.json`
- `straight_45_edge_cases.csv`
- `RAW_ARTIFACT_MANIFEST.json`
