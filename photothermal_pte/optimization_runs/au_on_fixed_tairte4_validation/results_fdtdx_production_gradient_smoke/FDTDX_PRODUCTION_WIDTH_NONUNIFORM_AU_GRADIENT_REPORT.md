# FDTDX production-width nonuniform-Au gradient smoke

Status: **VALIDATED_FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_SMOKE**

This checkpoint validates the optical total-absorbed-power gradient for a
nonuniform 20×20 Au-density field on the production-width 48 µm × 48 µm
FDTDX domain. It does not validate thermal, electrical, PTE, or optimization
gradients.

## Contract

- wavelength: 10 µm; scalar Gaussian waist: 8.5 µm
- fixed TaIrTe4: 20 µm × 20 µm × 100 nm
- design Au: 10 µm × 10 µm × 50 nm
- latent density: 20×20 at 500 nm; Yee sampling: 100×100 at 100 nm
- passive material relaxation: Drude coupling strength `s(rho)=rho^3`
- FDTDX source commit: `f26f84b70a8cceec9b889553955a868624736bf1`
- checkpointed AD: 16 checkpoints
- no clipping, smoothing, gain, or result/gradient rescaling

## Optical checks

- P_Q: 4.782529577e-13 W
- empty-subtracted six-face closure: 0.001937%
- late-window change: 0.171052%
- gradient L2 norm: 2.314620292e-14 W

## AD–FD interpretation

A direction is called strong only if `max(|AD|,|FD|) >= 0.05 ||gradient||_2`.
Near-null directions retain their raw local relative error, but are gated with
`|AD-FD|/||gradient||_2`; this avoids division by a small directional
derivative. No empirical gradient rescaling is used.

- finest strong-direction relative error: 0.095844%
- finest all-direction gradient-L2-normalized error: 0.023846%

## Runtime

- XLA compile: 16.321 s
- one value+gradient: 506.603 s
- four FD forwards: 181.671 s

The aborted four-checkpoint attempt was a performance-contract failure, not a
physics or gradient failure. Sixteen checkpoints completed the same production
AD in 8.44 minutes while staying within GPU memory.
