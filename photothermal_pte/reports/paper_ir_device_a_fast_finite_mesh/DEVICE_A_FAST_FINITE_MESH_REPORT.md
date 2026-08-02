# Device-A fast finite optical mesh validation

Status: `FAILED_FAST_DEVICE_A_SPATIAL_Q_CONVERGENCE`

This is a one-polarization (`E || a`) finite Device-A optical mesh diagnostic.
It is not a promoted production heat source and no thermal, PTE, adjoint, or
optimization calculation was run.

## Fixed contract

- Scalar Gaussian at 11 um, explicitly assumed `w0=8.75 um`.
- 60 x 60 um lateral domain, 50 x 50 um source aperture, six PML boundaries.
- Palik SiO2/Si and paper-derived anisotropic TaIrTe4 with `epsilon_z=epsilon_b`.
- Conformal variant 1, mesh accuracy 3, TaIrTe4 `dz=10 nm`.
- Fine optical x/y mesh is 50 nm; the full Device-A/Q outer region is 200 nm.
- No clipping, smoothing, gain, global rescaling, tiling, or source deletion.

`half-span` means the half-width of the square 50-nm refinement window around
the registered beam centre.  It is not a convection coefficient.

## Result

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
checkpoint uses 10 nm.  The observed failure is the x/y refinement-window
sensitivity.  A next test should keep the 50-nm illuminated region but compare
a 100-nm outer Device-A mesh against the current 200-nm outer mesh on one
polarization.  It should not proceed to thermal/PTE unless the spatial optical
gate is resolved or an explicitly approved downstream-observable gate replaces
the strict raw-Q gate.
