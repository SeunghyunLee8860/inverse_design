# Full latent/filter/projection FDTDX PTE AD--FD

Status: **VALIDATED_FULL_LATENT_FILTER_PROJECTION_FDTDX_PTE_ADFD**

At beta `2`, an interior latent baseline is solved such that the
finite-filtered/projected physical density reconstructs the already certified
physical baseline to `9.992e-16` maximum absolute error.
Thus the validated unscaled physical gradient can be pulled back exactly
through the filter/projection transpose without a changed-state approximation.

| direction | strength / ||g_latent|| | AD (A) | full FD (A) | relative error | ||g_latent|| error |
|---|---:|---:|---:|---:|---:|
| latent_adjoint_aligned | 1.000000 | 8.239196135902e-18 | 8.239379933904e-18 | 0.002231% | 0.002231% |
| latent_smooth_asymmetric | 0.177430 | -1.461877868407e-18 | -1.462348901725e-18 | 0.032211% | 0.005717% |
| latent_fixed_seed_random | 0.019563 | 1.611850953844e-19 | 1.604815207007e-19 | 0.436501% | 0.008539% |


Worst strong-direction error is `0.032211%`; worst near-null-safe
latent-gradient-norm error is `0.008539%`. No Q, latent,
density, objective, or gradient clipping/rescaling is used.

This validates the full differentiable chain for the stated numerical filter
scenario. The 750 nm filter radius is not yet a fabrication confidence
interval, and no Au optimization has been run.
