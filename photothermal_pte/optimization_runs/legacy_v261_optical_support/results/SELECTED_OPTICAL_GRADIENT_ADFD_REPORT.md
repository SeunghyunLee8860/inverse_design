# Selected production optical-gradient AD–FD

Status: `VALIDATED_SELECTED_OPTICAL_GRADIENT_ADFD`

The selected 373×373 physical-density optical gradient now passes. No new Maxwell solve, thermal solve, or optimization iteration was used for the final correction.

## Result

| quantity | AD | FD | relative error | gate |
|---|---:|---:|---:|---:|
| scalar $P_Q$ control | -7.743390204767e-15 W | -7.743411774764e-15 W | 2.785593e-06 | <1% |
| spatially weighted optical PTE | 8.178678869451e-20 A | 8.178920529520e-20 A | 2.954669e-05 | <1% |
| corrected one-direction combined smoke | 8.545357319273e-20 A | 8.545598286818e-20 A | 2.819786e-05 | <1% |

The optical terms are indirect `2.402571698596e-20 A` and direct material loss `5.776107170854e-20 A`.

## Root cause and correction

The forward Gaussian must remain active with zero amplitude to preserve the exact forward auto-nonuniform mesh. That left two active sources in the adjoint project, while default `cwnorm(1)` normalized monitor fields to the zero-amplitude Gaussian source spectrum instead of the FieldRegion spectrum. The FieldRegion-only CW field is reconstructed from the same raw monitor data under official `cwnorm(1)` and `cwnorm(2)` states. The two-state spatial residual is `2.525865e-16`. No FD-derived scale, empirical normalization, or gradient rescaling is used.

## Scope

This validates one selected-grid physical-density direction at `h=0.005`. It does not yet validate broader combined directions, optical gray-law sensitivity, exact-binary DRC, full latent AD–FD, or optimization.
