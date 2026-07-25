# Finite 2 um optical Q report

Status: **VALIDATED** by source controls, six-face energy closure, and
domain/PML/mesh convergence.

## Geometry and normalization

The calculation contains one finite 2 µm × 2 µm × 100 nm anisotropic TaIrTe4
flake on 285 nm SiO2/Si, with air above and PML on all six FDTD boundaries.
The fixed design is one centered 1.5 µm-radius, 600 nm-high disk made from the
same optical SiO2 material as the bottom spacer. It is not repeated.

The source is a finite scalar Gaussian beam, not a plane wave: 2 µm waist,
focus at the flake center, 6.8 µm source aperture, 3–6 µm broadband pulse, and
4 µm single-point analysis. The actual v261 GPU solver rejected TFSF because
GPU FDTD does not support it, so TFSF results are not substituted or claimed.

Each polarization uses its matching empty-layered-stack E/H reference. Q is
normalized to a measured central downward incident intensity of 1 W/m². For
the final 2 µm-waist source, the native central intensity is
6.52185261e-4 W/m², the total incident power at the unit-central-intensity
response is 1.13892e-11 W, and the minimum/peak intensity over the flake is
0.60028. No flux gain, Q rescaling, clipping, periodic crop, or periodic
tiling is used.

## Polarization controls

All powers below are for the 1 W/m² central-intensity response.

| case | P_Qx (W) | P_Qy (W) | P_Qz (W) | P_Q (W) | P_six (W) | closure | sigma_abs (m²) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat x | 1.55021e-12 | 1.54686e-14 | 0 | 1.56568e-12 | 1.57092e-12 | 0.3339% | 1.56568e-12 |
| flat y | 4.96909e-14 | 1.73092e-12 | 0 | 1.78061e-12 | 1.78637e-12 | 0.3223% | 1.78061e-12 |
| flat 45° | 7.99954e-13 | 8.73199e-13 | 0 | 1.67315e-12 | 1.67865e-12 | 0.3277% | 1.67315e-12 |
| fixed x, 8 µm domain | 2.53015e-12 | 2.85071e-14 | 0 | 2.55865e-12 | 2.56276e-12 | 0.1604% | 2.55865e-12 |

The existing periodic flat/TMM result remains only a large-area,
infinite-film reference. It is not imposed as the finite-flake answer.

## Convergence

The pass gate is less than 1% successive change in absorbed power and less
than 5% successive L1/L2 change in the Q distribution. Hotspot maxima and
component fractions are reported as diagnostics rather than silently folded
into the absorbed-power gate.

| sweep | settings | final successive ΔP_Q | final ΔP_six | final spatial L1 | final spatial L2 | result |
|---|---|---:|---:|---:|---:|---|
| lateral domain | 8/12/16 µm | 0.01996% | 0.01956% | 0.01992% | 0.02121% | pass |
| PML layers | 16/24 | 0.000210% | 0.001633% | 0.000584% | 0.000639% | pass |
| TaIrTe4 dz | 10/5/2.5 nm | 0.12778% | 0.10913% | 0.23601% | 0.28295% | pass |

For the mesh sweep, the point-sampled Q hotspot changes by 2.34% from 5 to
2.5 nm while integrated power and the full spatial distribution pass. The
component fractions remain approximately 98.9% Qx, 1.1% Qy, and 0% Qz.

The Gaussian waist was separately characterized at 1.5, 1.75, and 2 µm using
a fresh measured empty-stack intensity reference for each waist:

| waist (µm) | P_Q (W) | P_six (W) | closure | Q hotspot (W/m³) |
|---:|---:|---:|---:|---:|
| 1.50 | 2.53097e-12 | 2.53492e-12 | 0.1557% | 3.13389e7 |
| 1.75 | 2.54279e-12 | 2.54690e-12 | 0.1613% | 3.22511e7 |
| 2.00 | 2.55865e-12 | 2.56276e-12 | 0.1604% | 3.31904e7 |

The 1.75→2 µm P_Q change is 0.6199%; the spatial L1/L2 changes are
1.4346%/1.8700%.

## Final unit-response artifact

The selected final case is the converged 16 µm domain, 24 PML layers, 5 nm
TaIrTe4 dz, 2 µm-waist fixed-design x-polarized calculation:

- P_Q = 2.56071371e-12 W
- P_six = 2.56486066e-12 W
- six-face closure = 0.161683%
- sigma_abs = 2.56071371e-12 m²
- sigma_abs/A_geo = 0.64017843
- Q hotspot = 3.32156091e7 W/m³
- Qx/Q = 0.98886049, Qy/Q = 0.01113951, Qz/Q = 0

The artifact grid retains 50 nm zero padding around the pabs volume, while
the exact physical flake bounds `x,y=[-1,1] µm`, `z=[-100,0] nm` and an exact
flake mask are stored in metadata. The raw NPZ is not committed. Its server
path, 8,692,646-byte size, SHA-256
`7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`,
generation commit, and exact reproduction command are in
`RAW_ARTIFACT_MANIFEST.json`.

The outer total-field box records 5.71111e-12 W of outward power incident on
the PML for the final case. No unsupported direct PML-loss getter is claimed,
and Gaussian total-field power is not mislabeled as pure scattered power.

HEAT, adjoint, gradients, optimization, and PTE were not run. HEAT Draft PR
#2 was not modified.
