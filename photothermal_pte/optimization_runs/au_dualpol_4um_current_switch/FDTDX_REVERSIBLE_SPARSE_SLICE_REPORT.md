# FDTDX reversible sparse-slice report

Date: 2026-08-25 (Asia/Seoul)

Status: `PASS_SMALL_SCENE_SPARSE_SLICE_VJP_AND_EXACT_PAYLOAD_AUDIT`;
exact-grid runtime and latent AD-FD remain open.

## Change

`fdtdx_parity_reversible_sparse_sliced_vjp.py` retains the standard pinned
FDTDX forward step and the already proven ADE/CPML reverse equations. The
forward slice scan carries one full current state, but emits P-current and
P-previous only on explicitly supplied disjoint material regions. At a reverse
slice boundary, regional P is expanded into an otherwise zero full-domain
array before exact-primal reset.

E/H and CPML psi remain full-domain because they are not confined to the
material support. The final phasor state and the time-step schedule are retained
once and are not emitted per slice.

## Fail-closed support contract

The existing concrete coefficient-support audit must run before entering
`jax.value_and_grad`. Its PASS result contains the exact certified region
bounds and proves `c1/c2/c3/c4` are zero outside them with nonzero `c3` inside.
The custom VJP requires this result and rejects it if the requested region list
does not exactly match the audited bounds.

This host audit is intentionally outside the differentiated function. Running
it on a traced `c3` would require host concretization and fail during a real
latent gradient. The gradient itself remains connected: the full `c3` parameter
tuple is a differentiable input to every pinned FDTDX step and its step
cotangents are accumulated across all slices.

## Small-scene gradient parity

On the real 24-step six-face CPML, Lorentz slab, point-source, late-phasor
scene, one regional P support covers the slab. A six-step full-P sliced VJP and
the corresponding sparse-P sliced VJP produce the same phasor-power objective
and complete regional `c3` gradient at `rtol=5e-4`, with a nonzero gradient.

The comparison differentiates only the regional coefficient parameter. That is
the correct production contract: the 81 x 81 latent density changes Au
coefficients only inside the certified Au slice; TaIrTe4 coefficients are fixed.
Arbitrary hypothetical dispersion perturbations outside the support are not
silently represented.

## Exact-grid payload audit

An actual CPU placement of the `186 x 186 x 286` physical Ea model was built
without any Maxwell time step. Applying the new slice audit to its placed field
state and the actual `fixed_tairte4`/`au_design` slices returned:

| item | bytes |
|---|---:|
| full-domain E/H and CPML psi | 273,559,872 |
| full-domain P-current/P-previous | 712,400,832 |
| regional P-current/P-previous | 82,944,000 |
| full slice checkpoint | 985,960,704 |
| sparse slice checkpoint | 356,503,872 |
| removed per slice | 629,456,832 |

The audit status is `PASS`; detector and time-step state are not emitted per
slice. The exact-grid process performed allocation/placement only and then
exited. It did not run FDTD, use a GPU, or call Lumerical/HEAT/CHARGE.

The complete target-folder CPU suite passes `230 passed` after fast-forwarding
the disjoint Lumerical DFM commit `3d353bda`.

## Current boundary

This proves the sparse reset representation and its local gradient, not a
production solve. Open gates are:

- determine a stable slice length on the exact material/PML coefficients;
- measure compile, forward, reverse, and peak-device-memory costs on a short
  exact-grid run;
- run bounded Ea/Eb latent-density AD-FD only if the runtime projection remains
  within the user feasibility limit;
- then validate material Q, thermal/electrical residuals, signed current, and
  the complete objective gradient.

The next GPU action must be a short bounded probe on separately verified-idle
GPU UUIDs. It must not be a full 256,163-step gradient or an optimizer run.
