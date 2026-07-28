# Corrected combined five-direction diagnostic

- Status: `FAILED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD`
- Full certificate: **not validated**
- Worst strong error: `2.4273861e-05`
- Worst multidirection normalized error: `3.156058e-05`
- Worst subspace angle: `0.0011922823 deg`
- Optical closure: `0.00021654606`
- Q mapping: `3.5800079e-16`
- Thermal energy balance: `3.1763952e-12`
- Linear residual: `1.0239768e-11`
- Mapping transpose: `2.9832592e-15`

## Unresolved subgate

- 4um central_localized: plateau `0.0013950939` > `0.001`
- 4um fixed_seed_random: plateau `0.0050941101` > `0.001`
- 6um fixed_seed_random: plateau `0.0024250789` > `0.001`

The small directional derivatives reach the current FDTD finite-difference noise floor. No normalization or gradient rescaling was applied. Gray-law sensitivity, full latent AD-FD, and optimization remain blocked.
