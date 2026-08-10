# NLopt LD_MMA restart contract

Status: `READY_FOR_FRESH_NLOPT_LD_MMA_RUN018_RUN019`

Run016 is preserved as a diagnostic custom-MMA checkpoint.  It is not an
input or resume state for the fresh optimization.

## Update algorithm

- Library: NLopt 2.11.0
- Algorithm: `LD_MMA`
- User-defined move limit: none
- Custom `mma_step`: not imported by the production NLopt driver
- Adam moments: none
- Gradient direction normalization: none
- Post-update clipping: none
- Variable bounds: latent density in `[0,1]`
- Stopping tolerances: `ftol_rel=1e-3`, `xtol_rel=1e-3`
- Fail-closed stage ceiling: 40 full-physics evaluations

NLopt receives the minimization form of the signed-current objective and its
latent gradient.  The positive constant used to convert the raw unit-source
current to the fixed 285-uW current changes conditioning only, not direction.

## Gradient chain

Each distinct NLopt point performs the existing GPU-Maxwell/CUDA-thermal/
electrical forward and adjoint evaluation.  The returned physical-density
gradient is

`gradient_optical + gradient_thermal + gradient_electrical`.

The optimizer receives the latent gradient produced by the certified
filter/projection VJP.  Finite differences are not run during optimization.
Existing Ea and Eb combined AD-FD certificates remain the preflight evidence.

## Constraints and continuation

- Minimum terminal conductance is a strict NLopt inequality from beta 1.
- Differentiable 500-nm solid and void inequalities begin at beta 8.
- NLopt constraint tolerance is `1e-6`.
- Beta schedule is `1,2,4,8,16,32,64,128`.
- No symmetry or material-volume constraint is imposed.
- Final completion requires zero exact 500-nm bad nodes, less than 1% gray
  fraction, and a fresh exact-binary full-physics solve.

Every saved PNG/JSON is labelled as an NLopt full-physics evaluation.  NLopt
does not expose an internal accepted-iterate flag, so trial evaluations are
not falsely described as accepted updates.

## Offline checks

- NLopt scalar inequality analytic control passes.
- NLopt vector inequality callback/Jacobian layout control passes.
- Production source audit confirms no `MOVE_LIMIT`, custom `mma_step`, Adam,
  normalized direction, or hard-clipped update.
- Contact-anchored suite: 62 passed, 2 GPU-dependent tests skipped.
- Default fixed-frame geometry compatibility tests also pass.
