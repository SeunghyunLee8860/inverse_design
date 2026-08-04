# Final full-latent combined PTE AD–FD

**Status: `VALIDATED_FULL_LATENT_COMBINED_PTE_ADFD_WITH_USER_ACCEPTED_FD_NOISE`**

This is the end-to-end finite, nonperiodic certificate:

`81x81 latent -> 500 nm finite conic filter -> beta=8 tanh projection ->
component-specific Yee material Jacobian -> Maxwell Q -> conservative thermal
remap -> explicit anisotropic/material/interface thermal solve -> uniform-45°
PTE objective`.

No clipping, periodic wrapping, empirical normalization, gradient rescaling,
gain, Q smoothing, or Q rescaling was used. Optimization was not run.

## User-approved exception

The earlier physical-rho near-null strict plateau failure remains immutable.
The user explicitly accepted that solver-noise-level miss and authorized the
next stages. The present certificate therefore does not relabel the earlier
checkpoint; it uses the core 1% parity/angle/conservation gates without making
strict h-to-h/2 plateau convergence a gate.

## Full latent result

| metric | result | gate |
|---|---:|---:|
| worst strong-direction error | 0.011677% | <1% |
| global five-direction normalized error | 0.014397% | <1% |
| global directional gradient angle | 0.002225° | <1° |
| finite filter/projection transpose error | 6.052e-16 | <1e-12 |
| component-Yee transpose error | 2.983e-15 | <1e-12 |
| worst optical six-face closure | 0.014093% | <0.5% |
| worst Q mapping error | 2.372e-16 | <0.5% |
| worst thermal energy-balance error | 2.843e-12 | <1% |
| worst linear residual | 1.023e-11 | <1e-8 |

At `h=0.005`, all five directions (adjoint-aligned, central-localized,
design-edge-localized, smooth-asymmetric, and fixed-seed random) pass for both
4 and 6 µm named thermal flake scenarios. The maximum individual selected
error is 0.172688%.

## Gray-law sensitivity

`phi_p(rho)=rho^p`, for `p=1,2,3`, was applied consistently to optical
permittivity, bulk thermal conductivity, and interface conductance via the
same effective fraction. These are numerical scenarios, not a confidence
interval. Relative to `p=1`, `p=2` changes P_Q by
10.238% and
the PTE gradients by about 7.6–7.9°. `p=3` changes P_Q by
15.044% and
the gradients by about 14.4–14.8°. Gray-law choice is therefore a material
model uncertainty and must remain explicit during optimization.

## Figures

- [Full latent parity](figures/18_full_latent_adfd_parity.png)
- [Latent and gradient maps](figures/19_full_latent_gradient_maps.png)
- [Gray-law sensitivity](figures/20_gray_law_sensitivity.png)

Raw NPZ/FSP files remain outside Git and are SHA-pinned by the manifest.
