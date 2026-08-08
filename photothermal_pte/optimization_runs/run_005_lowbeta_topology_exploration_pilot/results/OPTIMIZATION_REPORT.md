# Run 005 fully binary PTE optimization

Status: `COMPLETED_FULLY_BINARIZED_EXACT_500NM_CONSTRAINED_PTE_OPTIMIZATION`

## Final design

- Final beta: `2048`; global iteration: `46`
- Gray fraction (0.01 < rho < 0.99): `2.803154e-04`
- Mean `4*rho*(1-rho)`: `5.693115e-05`
- Exact 500 nm solid/void bad cells: `0` / `0`
- Final evaluated density contains exactly `0` and `1`
- No post-hoc binary repair was used

## Fresh binary physics validation

- Pre-threshold continuous FOM: `8.688574877725e-07 A/W`
- Thresholded-binary FOM: `8.679256315189e-07 A/W`
- Binary FOM change: `-0.107251%`
- P_Q / P_six: `5.476087197869e-14` / `5.474744186646e-14 W`
- Six-face closure: `2.452502e-04`
- Thermal residual: `8.946627e-11`
- Thermal energy balance: `6.401750e-12`
- Solver path: fresh GPU Maxwell plus CUDA thermal/PTE; no CPU solver fallback

The small binary loss above is measured by a fresh solver evaluation, not inferred from a linearized gradient.
