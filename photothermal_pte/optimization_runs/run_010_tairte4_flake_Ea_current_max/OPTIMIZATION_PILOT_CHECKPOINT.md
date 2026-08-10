# Run 010 E parallel a optimization pilot checkpoint

Status: `RUNNING_TAIRTE4_FLAKE_EA_OBJECTIVE_LED_CONTINUATION`

The optimization starts from exact uniform latent, filtered, and physical
density 0.5. It has no symmetry constraint, half-domain forcing, S-shaped
seed, or inherited Run 009 density.

## First accepted update

- beta: `2`
- morphology weight: `0` (diagnostic only at this stage)
- maximum latent trust step: `0.02`
- initial signed full-flake current: `1.68397820e-23 A`
- accepted current: `7.01704971e-19 A`
- accepted physical-density range: `0.473753` to `0.526247`
- optical closure: `5.92276e-6`
- thermal residual: `9.85321e-11`
- thermal energy balance: `1.09586e-12`
- evaluation wall time: `169.4 s`

The exact 500 nm morphology audit reports 101 void-phase bad cells. This is a
low-beta diagnostic and did not veto the objective-led topology step. The
outside of the central design window is correctly treated as the fixed-solid
TaIrTe4 frame in this audit.

## Current-density Yee derivative correction

A fixed Jacobian built at uniform rho=0.5 is accurate for uniform and smooth
gray layouts but has a worst 2.638% global mapping error on a deliberately
rough near-binary random layout. It is therefore not used as a global
production Jacobian. Each optimization evaluation builds a layout-only local
component-Yee Jacobian at the current density. On the same near-binary control,
that local operator passed centered mapping FD with worst relative error
`3.63994e-8`. This construction performs 0 Maxwell solves.

The two failed global-chain diagnostics remain external provenance and were
not promoted. No empirical gradient normalization or rescaling was introduced.
