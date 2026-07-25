# Finite 2 um optical Q report

Status: source, polarization, and fixed-design closure checkpoints pass;
final convergence is pending.

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

## Fixed-design x control

The single centered 1.5 µm-radius, 600 nm-high design disk uses exactly the
same optical SiO2 material as the 285 nm bottom spacer and is not repeated.

| case | P_Qx (W) | P_Qy (W) | P_Qz (W) | P_Q (W) | P_six (W) | closure | sigma_abs (m²) | sigma/A_geo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed x | 2.53015e-12 | 2.85071e-14 | 0 | 2.55865e-12 | 2.56276e-12 | 0.1604% | 2.55865e-12 | 0.639663 |

The raw unit-response artifact grid is 80 × 80 × 41 with 50 nm lossless
padding around the pabs monitor. Its coordinate ranges are approximately
`x,y=[-1.05335, 1.05335] µm` and `z=[-150, 50] nm`; the exact physical flake
bounds `x,y=[-1,1] µm`, `z=[-100,0] nm` are stored separately in metadata so
the padding is never interpreted as TaIrTe4.

The finite-beam footprint is not uniform: the accepted empty controls record
minimum/peak intensity near 0.60 over the flake. Results are therefore labelled
Gaussian-beam cross sections, and beam-waist convergence remains mandatory.

## Current gates

- source-off control: pass;
- empty layered-stack controls: pass;
- flat x/y/45-degree volume-Q versus six-face closure: pass;
- fixed-design x closure: pass;
- 8/12/16 µm lateral-domain convergence: pending;
- PML-layer convergence: pending;
- 10/5/2.5 nm flake-dz convergence: pending;
- beam-waist convergence: pending;
- final finite Q artifact: not yet validated.

HEAT, adjoint, gradients, optimization, and PTE are not run here.
