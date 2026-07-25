# Finite 2 um optical Q report

Status: polarization checkpoint; final convergence is pending.

## Geometry and normalization

The calculation contains one finite 2 µm × 2 µm × 100 nm anisotropic TaIrTe4
flake on 285 nm SiO2/Si, with air above and PML on all six FDTD boundaries.
The source is a finite scalar Gaussian beam, not a plane wave: 2 µm waist,
focus at the flake center, 6.8 µm source aperture, 3–6 µm broadband pulse, and
4 µm single-point analysis.

Each polarization uses its matching empty-layered-stack E/H reference. Q is
normalized to a measured central downward incident intensity of 1 W/m². No
flux gain, Q rescaling, clipping, periodic crop, or periodic tiling is used.

## Flat polarization controls

All powers below are for the 1 W/m² central-intensity response.

| polarization | P_Qx (W) | P_Qy (W) | P_Qz (W) | P_Q (W) | P_six (W) | closure | sigma_abs (m²) |
|---|---:|---:|---:|---:|---:|---:|---:|
| x | 1.55021e-12 | 1.54686e-14 | 0 | 1.56568e-12 | 1.57092e-12 | 0.3339% | 1.56568e-12 |
| y | 4.96909e-14 | 1.73092e-12 | 0 | 1.78061e-12 | 1.78637e-12 | 0.3223% | 1.78061e-12 |
| 45 deg | 7.99954e-13 | 8.73199e-13 | 0 | 1.67315e-12 | 1.67865e-12 | 0.3277% | 1.67315e-12 |

The finite-beam footprint is not uniform: the accepted empty controls record
minimum/peak intensity near 0.60 over the flake. Results are therefore labelled
Gaussian-beam cross sections, and beam-waist convergence remains mandatory.

## Current gates

- source-off control: pass;
- empty layered-stack controls: pass;
- flat x/y/45-degree volume-Q versus six-face closure: pass;
- fixed-design x closure: pending;
- 8/12/16 µm lateral-domain convergence: pending;
- PML-layer convergence: pending;
- 10/5/2.5 nm flake-dz convergence: pending;
- beam-waist convergence: pending;
- final finite Q artifact: not yet validated.

HEAT, adjoint, gradients, optimization, and PTE are not run here.
