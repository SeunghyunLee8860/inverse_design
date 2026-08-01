# Corrected-substrate corner-free 45° edge control

Status: `COMPLETED_CORRECTED_SUBSTRATE_STRAIGHT_45_EDGE_CONTROL`

This is a simple `y=x` half-plane control, not the Device-A polygon and not a
paper reproduction.  It uses the explicitly assumed scalar-Gaussian
`w0=8.75 um` scenario.  There are no electrodes, corners, weighting field,
PTE current, adjoint, or optimization.

## Optical/material contract

- Six PML faces; no periodic/Bloch boundaries.
- TaIrTe4: 130 nm; `epsilon_x=epsilon_b`, `epsilon_y=epsilon_a`,
  `epsilon_z=epsilon_b`.
- Substrate: 285 nm `SiO2 (Glass) - Palik` on `Si (Silicon) - Palik`.
- Full raw control-volume Q is preserved.  Thermal uses only the separately
  saved `Q_TaIrTe4_only_W_m3`; lossy substrate Q is not projected into the
  flake.
- No clipping, smoothing, gain, global rescaling, tiling, or polarization
  matching was used.

## Results

| metric | E||a | E||b | b/a |
|---|---:|---:|---:|
| TaIrTe4 P_Q at 1 W/m2 | 1.175214597e-11 | 1.466901088e-11 | 1.248198 |
| matched-volume closure | 0.312757% | 0.341076% | — |
| Tmax | 2.338654064e-01 K | 2.204494141e-01 K | 0.942634 |
| TaIrTe4 average T | 6.933575839e-03 K | 8.656135243e-03 K | 1.248437 |
| strict-centred P99 abs(dT/dn) | 5.918470401e+04 K/m | 3.113052893e+04 K/m | 0.525989 |

The gradient is not evaluated on a staircase-edge cell.  A value is retained
only when all four `+x,-x,+y,-y` TaIrTe4 neighbours exist; otherwise it is
masked.  The comparator uses the nearest fully centred interior band.
The pixelwise coordinate-rotation identity has maximum relative errors
`7.512e-16` and
`7.236e-16` for
`E||a/b`, and there are no finite gradient samples outside the flake.

## Interpretation

Numerical pass: `True`.  Paper-like b/a gradient trend: `False`.
This 100-nm lateral run is a baseline.  It is not promoted as a mesh-converged
production optical Q without a targeted refinement comparison.
