# Device-A fast finite optical mesh validation

Status: `FAILED_FAST_DEVICE_A_MATERIAL_OVERLAP_SPATIAL_CONVERGENCE`

This is a one-polarization (`E || a`) finite Device-A optical mesh diagnostic.
It is not a promoted production heat source and no thermal, PTE, adjoint, or
optimization calculation was run.

## Corrected material-overlap thermal-source comparison

The earlier raw-control-volume comparison is retained below as an optical
diagnostic, but it is **not** the TaIrTe4-only thermal source.  Both optical
cases were independently mapped onto the same explicit thermal grid using
optical-cell/thermal-material overlap, without nearest-cell relocation.

- mapped power change: 0.001165%
- mapped lateral Q NRMSE: 1.000888%
- mapped full-3D Q NRMSE: 3.330164%
- mapped depth Q NRMSE: 0.129946%
- mapped correlation: 0.999433853
- mapping power error: zero in both cases

The spatial gate still fails, but its localization is now explicit:

- within |x|,|y| <= 9 um: 0.156511% full-3D NRMSE
- 9--12 um transition: 81.1628% of squared error
- within 0.25 um of the binary FVM flake boundary: 99.5750% of squared error
- worst z layer: -5.0 nm, carrying 81.8665% of squared error

This identifies the failed candidate as a mesh-layout error: the 50-to-100 nm
transition crossed illuminated Device-A material boundaries.  It is not a
power-conservation failure in the remap.

Important limitation:
Omega_TaIrTe4 is the union of binary FVM cells selected by the thermal cell-center polygon mask; it is not an analytic polygon cut-cell volume.
Consequently this report does not claim an analytic polygon cut-cell thermal
geometry.


## Fixed contract

- Scalar Gaussian at 11 um, explicitly assumed `w0=8.75 um`.
- 60 x 60 um lateral domain, 50 x 50 um source aperture, six PML boundaries.
- Palik SiO2/Si and paper-derived anisotropic TaIrTe4 with `epsilon_z=epsilon_b`.
- Conformal variant 1, mesh accuracy 3, TaIrTe4 `dz=10 nm`.
- Fine optical x/y mesh is 50 nm; the full Device-A/Q outer region is 200 nm.
- No clipping, smoothing, gain, global rescaling, tiling, or source deletion.

`half-span` means the half-width of the square 50-nm refinement window around
the registered beam centre.  It is not a convection coefficient.

## Raw optical control-volume diagnostic

All five GPU calculations completed with auto-shutoff <= 1e-5 and six-face
closure below 0.5%.  Total absorbed power appears converged much earlier than
the spatial source.  The most useful fast candidate used 50 nm within +/-9 um,
100 nm from 9--12 um, and 200 nm outside.  Relative to the +/-12 um all-50-nm
reference it achieved:

- total-power change: `0.0560%`;
- depth-profile NRMSE: `0.3588%`;
- lateral-Q NRMSE: `1.9577%`;
- full-3D-Q NRMSE: `9.2816%`;
- conservative-remap power error: `0.000e+00`;
- runtime: `489.6 s`
  versus `615.0 s`.

The candidate saves about 20% solver time, but its lateral and full-3D Q
metrics exceed the 0.5% gate.  It is therefore not promoted.  The 5, 7, and
9 um direct 50-to-200-nm transitions also fail spatial convergence even when
their total powers are close.  Total power alone would have produced a false
pass.

## Interpretation and next minimal test

The current data do not isolate `dz=10 nm` versus `dz=5 nm`; every case in this
checkpoint uses 10 nm.  The observed failure is localized at real Device-A
material boundaries that the candidate accidentally placed in its 100/200-nm
region.  The next economical optical test must follow those illuminated
flake/electrode boundaries with narrow 50-nm mesh strips while leaving only
homogeneous remote air/SiO2/Si coarse.  Blindly changing the whole outer region
to 100 nm is not the diagnosed fix.  It should not proceed to thermal/PTE until
the mapped thermal-source spatial gate is resolved or the user explicitly
approves a downstream-observable replacement gate.
