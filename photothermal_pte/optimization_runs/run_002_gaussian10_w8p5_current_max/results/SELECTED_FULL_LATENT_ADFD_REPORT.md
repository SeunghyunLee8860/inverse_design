# Selected full latent/filter/projection AD–FD

Status: `VALIDATED_SELECTED_FULL_LATENT_ADFD`

This is the final numerical chain used by the continuous optimizer: latent 373×373 density, finite nonperiodic 500 nm conic filter, beta=2 tanh projection, complex SiO2 optical interpolation, GPU Maxwell Q, conservative 3D remap, CUDA anisotropic/finite-G thermal solve, and uniform-45° PTE objective.

| latent direction | AD (A) | FD (A) | relative error | normalized error |
|---|---:|---:|---:|---:|
| adjoint_aligned | 1.329719392101e-19 | 1.329724490372e-19 | 0.000383% | 0.000383% |
| fixed_seed_random | -1.657484510156e-21 | -1.656941748864e-21 | 0.032746% | 0.000726% |

Worst relative error is `0.032746%`. No clipping, empirical normalization, or gradient rescaling was used. The continuous optimization may start. Final fabrication promotion still requires explicit binary DRC and robust physical-interface scenario checks.
