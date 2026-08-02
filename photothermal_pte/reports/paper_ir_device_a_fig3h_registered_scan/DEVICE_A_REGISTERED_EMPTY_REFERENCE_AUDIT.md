# Registered Device-A empty-reference audit

Status: `BLOCKED_REGISTERED_EMPTY_B_INNER_LATERAL_FLUX_GATE_REQUIRES_PHYSICAL_REVIEW`

Both polarization-matched empty SiO2/Si GPU calculations completed and reached
the requested auto-shutoff.  The existing `E||a` acceptance passed.  `E||b`
failed only the local inner-box lateral-flux gate:

| metric | E parallel a | E parallel b |
|---|---:|---:|
| auto-shutoff | 8.356340e-06 | 8.345160e-06 |
| inner max lateral flux / incident | 1.093490e-05 | 1.038434e-04 |
| outer max lateral flux / incident | 5.365899e-07 | 8.571098e-07 |
| source-aperture edge / central intensity | 1.517607e-06 | 1.595661e-06 |

The registered beam centre is outside the flake, while the inner six-face box
is local to the flake.  Lateral Poynting flux through that box is therefore a
physical part of the off-flake Gaussian illumination, not a direct PML leakage
measurement.  The outer lateral fractions are below `1e-6` for both
polarizations.

No gate was relaxed and no finite Device-A solve was started.  The proposed
correction is to retain the inner signed flux as a diagnostic and use the outer
box lateral flux for the `<1e-4` truncation gate, while retaining the existing
matched-volume closure, auto-shutoff, source-aperture, and material-readback
gates.  This contract change requires explicit approval.
