# FDTDX increment-state production-path integration

## Scope and boundary

This checkpoint integrates the cancellation-resistant Lorentz/Drude
`(P, delta-P)` ADE into an isolated FDTDX fork. It is opt-in; the upstream
second-order polarization recurrence remains the default. No Lumerical file or
run was edited, launched, or reinterpreted. No GPU was used for this checkpoint.

This closes one small-domain Lorentz forward/checkpointed-adjoint software
gate; a corresponding full-FDTD Drude parameter control remains required.
It does **not** certify the 4-um full geometry, any mesh, a gray Au material law,
the thermal/electrical maps, or optimization readiness.

## Reproducible fork state

- pinned upstream base: `f26f84b70a8cceec9b889553955a868624736bf1`
- isolated kernel commit: `24d0cb2374bf03b6bfdc528b189c69685b74dfee`
- opt-in production integration commit:
  `fc09ce54dc32ea13e27d2af799cdb3771801bf65`
- fork branch: `codex/increment-state-ade`
- exported integration patch:
  `fdtdx_patches/0002-feat-dispersion-integrate-opt-in-increment-state-ADE.patch`
- patch SHA-256:
  `1532e032fbe3656b4397f6c8d94314339f4bd94b0e2583c162c317c106b901cb`
- integrated `src/fdtdx/increment_state.py` SHA-256:
  `bd2d11a3a5b10d49d3a9d13c997134f4cf9d7bb451b42a8536d673711890faf3`

A clean checkout reproduces the fork by applying patch `0001`, then `0002`,
to the pinned base. The project-side
`test_fdtdx_increment_state_integration_patch.py` pins the second patch bytes,
commit header, exact 16-file scope, required production markers, and the rule
that the patch cannot touch Lumerical.

## What changed in FDTDX

`SimulationConfig(dispersive_state_representation="increment")` now selects
the new path. Existing arrays are reused without another full-grid allocation:

- `dispersive_P_curr` stores `P`;
- `dispersive_P_prev` stores `delta-P`;
- `dispersive_c1/c2/c3` store `A/C/B`;
- `dispersive_c4` must be absent.

The same representation is used by uniform/static multi-material placement,
device parameter application, diagonal `update_E`, point/plane/mode sources,
mode-overlap detectors, carrier-frequency effective permittivity, and the TFSF
broadband impedance spectrum. CCPR, oriented poles, and a dispersive
full-tensor material path fail closed in increment mode.

## CPU validation evidence

All commands forced the CPU backend. The dedicated environment was
`/home/seunghyun200/.venvs/fdtdx-fresh-py312`.

- increment kernel/spectrum/full-FDTD gates: `11 passed in 8.35 s`;
- dispersion/initialization/source/mode plus new gates:
  `166 passed in 30.27 s` after formatting;
- complete FDTDX unit suite:
  `2605 passed, 2 skipped, 1 xfailed in 238.63 s`;
- project-side FDTDX audit suite including the patch gate:
  `161 passed in 11.91 s`;
- Ruff 0.15.22 on every changed Python file: no remaining errors.

The small driven lossy Lorentz scene ran through actual coefficient placement,
source correction, `checkpointed_fdtd`, and reverse-mode differentiation. For
one active `B` voxel:

```text
AD = 2.240709215e-01
FD = 2.239563912e-01
symmetric relative error = 2.556324618e-04
absolute difference = 1.145303249e-04
```

The test is a full-FDTD checkpointed AD-FD gate, not merely differentiation of
the isolated one-cell kernel.

## Remaining optimizer blocker: continuous dispersive mixing

The current generic FDTDX `Device` path linearly interpolates every stored
coefficient between its two materials. In increment mode that means a gray
air/Au voxel interpolates `A` and `C` as well as the coupling `B`. This changes
the oscillator damping/resonance with density; it is not automatically the
same as a fixed-pole susceptibility whose strength alone follows a chosen Au
occupancy law. The endpoints are exact, but intermediate gray voxels are not
yet a literature-backed material model.

Therefore this integration does not authorize a continuous-density inverse
design. Before that, the papers and the existing optical/thermal/electrical
maps must be audited together and one shared physical occupancy must be
defined. For a fixed-pole optical relaxation, the likely implementation is to
hold the physical pole dynamics fixed and interpolate only the susceptibility
strength, but that choice must be derived and AD-FD tested rather than assumed.
Exact-binary controls do not depend on this gray-law decision and are the next
allowed field-level timing/mesh probe.

## Runtime and GPU launch boundary

The old z16/t32 FDTDX model took about `19.1 min` for one polarization. Two
independent polarization solves therefore gave a measured Maxwell lower bound
near `38 min` per forward-plus-adjoint design iteration, or more than `63 h`
for 100 iterations before thermal/electrical work. That old validation grid is
not an optimizer grid.

No full-size timing exists yet for the patched increment-state fork, so it is
incorrect to promise a faster number. The next launch must be a short,
coarse, exact-binary timing/closure control with explicit compile and steady
execution times. Immediately before launch, inspect compute-process ownership;
run Ea and Eb concurrently on two distinct verified-idle GPUs, and never use a
GPU carrying another user's process. One FDTDX solve remains single-GPU.

Promotion remains blocked until those controls establish a practical runtime,
then source/material closure and z convergence pass on newly generated
increment-state artifacts. Old second-order source pairs and results cannot be
mixed into that comparison.
