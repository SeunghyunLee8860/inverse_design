# Full combined FDTDX--thermal--weighting PTE multi-direction AD--FD

Status: **VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_MULTIDIRECTION_ADFD**

The previously validated adjoint-aligned central difference is reused by exact
SHA. Four new independent directions recompute the complete forward chain at
`rho +/- 0.01 d`. All perturbations remain inside `(0,1)` without clipping.

| direction | strength / ||g|| | combined AD (A) | full FD (A) | relative error | ||g||-normalized error |
|---|---:|---:|---:|---:|---:|
| combined_adjoint_aligned | 1.000000 | 6.453035058204e-18 | 6.452153139305e-18 | 0.013667% | 0.013667% |
| smooth_asymmetric | 0.173971 | -1.122639469311e-18 | -1.123824752034e-18 | 0.105469% | 0.018368% |
| central_localized | 0.086274 | 5.567302224687e-19 | 5.560443394113e-19 | 0.123198% | 0.010629% |
| design_edge_localized | 0.000845 | -5.454661076185e-21 | -5.344112513596e-21 | 2.026681% | 0.001713% |
| fixed_seed_random | 0.120694 | 7.788452349584e-19 | 7.779366782068e-19 | 0.116654% | 0.014080% |


The worst strong-direction relative error is `0.123198%`;
the worst near-null-safe gradient-norm error is `0.018368%`.
The worst linear residual is `9.866e-10` and the worst thermal and
terminal balances are `0.000000%` and `0.000000%`.

No Q, density, objective, or gradient clipping/rescaling is used. This closes
the physical-density multi-direction gate only; it does not yet certify the
latent/filter/projection chain and does not authorize optimization.
