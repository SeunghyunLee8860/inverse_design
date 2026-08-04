# Matched rho=0.5 CPU-TFSF forward gate

Status: `VALIDATED_MATCHED_RHO05_CPU_TFSF_FORWARD`

## Fixed physical contract

- central design/PTE ROI: exact `x,y=[-1,1] µm`
- design: `2 µm × 2 µm × 600 nm`, uniform physical `rho=0.5`
- design interpolation:
  `epsilon(rho)=1+rho*(1.38^2-1)`
- TaIrTe4 optical background: `100 nm`, laterally extended through PML
- TFSF: `x,y=[-1.3,1.3] µm`, `z=[-1.8,1.5] µm`
- Q and closed-six-face volume:
  `x,y=[-1.15,1.15] µm`, `z=[-0.15,0.75] µm`
- boundaries: six PML faces; no periodic/Bloch boundary
- PML: 32 layers, stabilized x/y and standard z
- flake mesh: `dz=2.5 nm`
- central incident intensity target: `1 W/m²`

The v261 PML profile matrix readback was
`[2,2,2,2,1,1]`, where `1=standard` and `2=stabilized`.

## Matched PML geometry

An initial fail-closed setup retained the PML-24 outer bounds while changing
to PML-32. That moved the realized z-top PML inner face to `1.41333 µm`,
inside the TFSF z-max at `1.5 µm`; the solver was therefore not run.

The promoted case adds the eight extra `50 nm` layers outside the original
physical interior. Its outer bounds are `x,y=±3.6 µm` and
`z=[-3.6,3.4] µm`. The realized PML inner bounds are:

- x: `[-2.0000000000000016, 1.9999999999999953] µm`
- y: `[-2.0000000000000016, 1.9999999999999953] µm`
- z: `[-2.0114285714285744, 1.8117647058823516] µm`

Thus the ROI, design, TFSF, Q volume, material stack, source, and monitor
positions were unchanged; only the outer padding needed for the additional
PML layers was added.

## Result

- source intensity: `0.9990759709401064 W/m²`
- source-intensity relative error: `9.240290598936385e-4`
- `P_Q`: `1.6887880194040323e-12 W`
- `P_six`: `1.6893345559747856e-12 W`
- six-face closure: `3.2352180852533735e-4` (`0.0323522%`)
- `Qx`: `1.6885593488584841e-12 W`
- `Qy`: `2.286705455481133e-16 W`
- `Qz`: `0 W`
- dominant Qx hotspot:
  `(0.025, ~0, -0.0025) µm`,
  `6.625457662100932e6 W/m³`
- ROI x/y reflection NRMSE:
  `4.5605734e-7 / 4.4570181e-7`

The `0.5%` source-normalization and six-face-closure gates both passed.
No clipping, smoothing, gain, global rescaling, tiling, or old-artifact crop
was used.

## Timing boundary

Native solve / complete-session wall times were
`583.9737408906221 / 590.5969020240009 s`. A separate pre-existing GPU FDTD
optimization was active during this run. These times are therefore retained
only as contended reference values and are not promoted as clean performance
benchmarks.

This checkpoint ran no thermal solve, adjoint, finite difference, or
optimization.
