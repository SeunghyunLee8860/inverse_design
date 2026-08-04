# Corrected combined strong-direction diagnostic

- Status: `DIAGNOSTIC_PASSED_CORRECTED_COMBINED_STRONG_DIRECTION_ADFD`
- Scope: one adjoint-aligned direction only; this is not the final five-direction certificate.
- Empirical normalization: absent
- Gradient rescaling: absent
- Old Stage 10 and pre-plateau diagnostic raw results: preserved

## Directional results

- 4um: `h=0.01: 1.6466209e-05`, `h=0.005: 1.2176752e-05`, `h=0.0025: 2.1406717e-05`; plateau `9.2300773e-06`; strict monotone `false`
- 6um: `h=0.01: 1.9693371e-05`, `h=0.005: 1.1941257e-05`, `h=0.0025: 2.4273861e-05`; plateau `1.2332751e-05`; strict monotone `false`

The centered-FD derivatives form a solver-noise-limited plateau well below `0.1%`. Strict monotone reduction is reported separately and is false; it is not hidden or rewritten. The independent strong AD-FD gate remains `1%`.

## Worst auxiliary gates

- strong relative error: `2.4273861e-05`
- optical closure: `0.00021654606`
- Q mapping: `2.386256e-16`
- thermal energy balance: `3.1763952e-12`
- linear residual: `1.0213522e-11`
- mapping transpose: `2.9832592e-15`

The next gate is central-localized, design-edge-localized, smooth/asymmetric, and fixed-seed-random combined AD-FD. Gray-law sensitivity, full latent AD-FD, and optimization remain blocked.
