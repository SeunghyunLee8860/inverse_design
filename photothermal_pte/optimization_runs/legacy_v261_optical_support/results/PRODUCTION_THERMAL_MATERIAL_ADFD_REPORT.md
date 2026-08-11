# Production fixed-Q thermal-material AD-FD

Status: `VALIDATED_PRODUCTION_FIXED_Q_THERMAL_MATERIAL_ADFD_ENDPOINTS`

This checkpoint freezes the validated Maxwell volumetric Q and differentiates
only the thermal design-material branch: isotropic gray design kappa and the
TaIrTe4/design interface conductance. The 201x201 nodal density on the coarse
20 um canvas maps to 200x200 thermal cells by four-node cell averaging; its
exact transpose is used for the nodal gradient.

| scenario | selected h | AD-FD relative error | mapping dot error | mapping FD error | energy balance |
|---|---:|---:|---:|---:|---:|
| grown/grown | 0.0025 | 2.229173e-06 | 1.811426e-15 | 2.621206e-11 | 1.655433e-12 |
| evaporated/evaporated | 0.005 | 1.119103e-05 | 1.811426e-15 | 5.943742e-14 | 1.096998e-11 |

The grown/grown raw result remains immutable with its original failed status.
That status came solely from combining a `1e-12` transpose gate with a
roundoff-sensitive `h=1e-5` mapping FD. Its physical AD-FD error converges from
`3.563e-5` to `2.229e-6` as `h` goes from `0.01` to `0.0025`. The published
gate separates mapping transpose (`1e-12`) from mapping FD (`1e-8`); no
empirical normalization or gradient rescaling was used.

This is not the combined optical/thermal gradient. Maxwell adjoint,
coarse-gradient window selection, latent filter/projection AD-FD, and
optimization remain blocked.
