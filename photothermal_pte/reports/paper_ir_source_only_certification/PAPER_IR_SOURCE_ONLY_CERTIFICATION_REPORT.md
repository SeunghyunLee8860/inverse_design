# Paper-IR source-only beam certification

Status: `BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE_BEFORE_SOURCE_ONLY`

The prior `VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT` result remains
unchanged.  The paper/SI audit completed, but the new source-only case did not
open a Lumerical FDTD session.  It is therefore neither a failed beam nor a
certified beam: its optical observables were never produced.

## Paper contract

The paper reports a 7–13 µm Block LaserTune QCL, a 40x reflective objective
with NA=0.4, an approximately 9–16 µm diffraction-limited spot, and 11 µm /
285 µW for Figure 3.  It does not publish whether the spot is a radius,
diameter, FWHM, or 1/e^2 width, nor the exact 11-µm waist plane or pupil fill.
The detailed `PAPER_REPORTED`, `PAPER_INFERRED`, and `EXPLICIT_ASSUMPTION`
records are in `paper_ir_beam_contract_summary.json`.

The first source-only candidate is an explicit assumption:

- wavelength: 11 µm
- Gaussian 1/e^2 intensity radius: 12.0 µm
- eta=lambda/(pi*w0): 0.291784
- Rayleigh range: 41.126304 µm
- backward-source distance from waist: -5.065000 µm
- source span/domain: 50/60 µm
- analytic square capture: 99.99291408%
- analytic source-boundary maximum/mean: 1.93378964e-04 / 5.86048981e-05

The scalar model is not production-approved.  A matched NA=0.4 vector
thin-lens comparison is still required.  The old nominal `w0=2 µm` artifacts
remain `DIAGNOSTIC_ONLY_INVALID_FOR_PAPER_LIKE_BEAM` and are forbidden for
thermal, PTE, or Figure-3 reproduction.

## Startup probes

Two contract-only attempts failed before session creation.  Both report:
`ANSYSLI exited or could not read server port`.  Neither attempt completed
`runsetup`, started a GPU solve, or invoked CPU fallback.  No TaIrTe4,
substrate, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization calculation ran.

## Source-only gates

| Gate | Result |
|---|---|
| requested_vs_realized_width_relative_error_below_0p5pct | NOT EVALUATED |
| beam_center_error_below_one_cell | NOT EVALUATED |
| square_capture_at_least_99p9pct | NOT EVALUATED |
| realized_source_boundary_max_below_1e_minus_3 | NOT EVALUATED |
| incident_power_closure_below_0p5pct | NOT EVALUATED |
| field_time_convergence_below_0p5pct | NOT EVALUATED |
| no_NaN_or_Inf | NOT EVALUATED |
| GPU_only_no_CPU_fallback | NOT EVALUATED |
| auto_shutoff_at_most_1e_minus_5 | NOT EVALUATED |

These are `NOT EVALUATED`, not failures and not passes.

## Grid, memory, and runtime

Actual contract-only grid/memory readback is unavailable because the session
did not open.  For context only, a historical 48-µm high-index material case
had 343,657,881 grid points and a 15.169-GiB
precise GPU-memory estimate.  Blind lateral-area scaling to 60 µm would give
about 536,965,439 points and
23.702 GiB, but this is **not** a
homogeneous-air source-only estimate and cannot certify feasibility.

The old 12-µm nominal-w0=2-µm 4-ps cases averaged 820.933 s.  Five
times that old-grid mean is 68.41 min, but it
is only historical arithmetic, not a prediction for the 60-µm contract.
Total expected GPU time remains
`UNRESOLVED_UNTIL_LICENSE_AND_RUNSETUP`.

## Decision

The four planar/finite-edge optical cases are not worth executing now.  First
restore license/session startup, obtain the actual source-only grid/memory
readback, execute one GPU-only homogeneous-air case, and pass the realized
beam gates.  Then perform the matched scalar/vectorial comparison before any
material case.  No production-Q promotion is made by this checkpoint.
