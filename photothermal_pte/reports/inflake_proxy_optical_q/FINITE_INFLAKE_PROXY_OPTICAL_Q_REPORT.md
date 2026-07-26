# Finite in-flake SiO2 proxy optical-Q validation

Status: **VALIDATED_FINITE_INFLAKE_PROXY_OPTICAL_Q**

This is a fresh v261 GPU FDTD validation. The radius-1.5-µm PR #3 artifact was neither reused nor cropped.

## Geometry and optical contract

- TaIrTe4: 2 µm × 2 µm × 100 nm
- centered SiO2 disk: radius 0.8 µm, height 600 nm, fully inside the flake
- outside the disk: air; no support annulus, overhang support, or oxide pillar
- finite Gaussian waist 2 µm, aperture 6.8 µm, source 3–6 µm, analysis 4 µm
- six PML boundaries, auto nonuniform mesh, conformal variant 1, accuracy 5
- central incident intensity: 1 W/m²

All six outer FDTD faces use PML. The disk–TaIrTe4, TaIrTe4–SiO2, SiO2–Si, and solid–air optical interfaces are solved through their material permittivities and conformal Maxwell mesh; thermal interface conductance is not part of this optical solve.

## Promoted fresh result

- P_Q = `2.036108860469182e-12 W`
- P_six = `2.040668004695463e-12 W`
- six-face closure = `0.223414304%`
- Qx/Q, Qy/Q, Qz/Q = `0.99332407`, `0.00667593037`, `0`
- hotspot (x,y,z) = `(-1.333333333e-08, -9.733333333e-07, -5.000000000e-09) m`

## Convergence

- domain 12→16 µm: ΔP_Q `0.0240581%`, ΔP_six `0.0232486%`, spatial-Q L2 `0.025513%`
- PML 16→24 layers: ΔP_Q `0.000270435%`, ΔP_six `0.00134641%`, spatial-Q L2 `0.000594892%`
- flake dz 5→2.5 nm: ΔP_Q `0.0769457%`, ΔP_six `0.0503751%`, spatial-Q L2 `0.608514%`

All source-off, empty-stack x/y/45°, finite-flat x/y/45°, proxy, six-face closure, domain, PML, and flake-dz gates passed.

The promoted NPZ independently passed finite-value, component-sum, coordinate-order, geometry, prohibited-operation, and P_Q reintegration audits.

No clipping, smoothing, gain, global rescaling, tiling, source deletion, thermal solve, PTE, adjoint, gradient, or optimization was used.
