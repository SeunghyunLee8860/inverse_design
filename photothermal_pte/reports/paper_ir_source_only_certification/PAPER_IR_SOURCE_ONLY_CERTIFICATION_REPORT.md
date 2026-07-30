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

The fixed source-only scenario is an explicit assumption:

- wavelength: 11 µm
- Gaussian 1/e^2 intensity radius: 12.0 µm
- eta=lambda/(pi*w0): 0.291784
- Rayleigh range: 41.126304 µm
- backward-source distance from waist: -5.065000 µm
- source span/domain: 50/60 µm
- analytic square capture: 99.99291408%
- analytic source-boundary maximum/mean: 1.93378964e-04 / 5.86048981e-05

Its required label is **paper-like scalar-Gaussian scenario with an
explicitly assumed waist**.  It is not an experimentally reproduced beam or
a paper-certified beam.  A thin-lens comparison is an optional future
diagnostic and is not a gate or blocker.  The old nominal `w0=2 µm`
artifacts remain `DIAGNOSTIC_ONLY_INVALID_FOR_PAPER_LIKE_BEAM` and are
forbidden for thermal, PTE, or Figure-3 reproduction.

## Startup probes

4 contract-only attempts failed before session creation.  They report:
`ANSYSLI exited or could not read server port`.  None of the attempts completed
`runsetup`, started a GPU solve, or invoked CPU fallback.  No TaIrTe4,
substrate, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization calculation ran.

The newest probe uses the fixed scalar contract at commit
`f3fc01614590200ca5217e5139ebe2b1b314bccc`.  TCP/port reachability by
itself is not a license certificate: the minimum fix is to restore the v261
Ansys Licensing Client Proxy/server-port handshake for this user session.
No CPU fallback is an acceptable workaround.

## Source-only gates

| Gate | Result |
|---|---|
| requested_vs_realized_x_waist_error_below_0p5pct | NOT EVALUATED |
| requested_vs_realized_y_waist_error_below_0p5pct | NOT EVALUATED |
| Gaussian_fit_NRMSE_below_0p5pct | NOT EVALUATED |
| beam_center_error_below_one_cell | NOT EVALUATED |
| xy_ellipticity_below_0p5pct | NOT EVALUATED |
| square_capture_at_least_99p9pct | NOT EVALUATED |
| realized_source_boundary_max_below_1e_minus_3 | NOT EVALUATED |
| incident_power_closure_below_0p5pct | NOT EVALUATED |
| actual_mesh_readback_available | NOT EVALUATED |
| GPU_memory_readback_available | NOT EVALUATED |
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
beam gates.  If it passes, proceed directly to planar a/b and
straight-45-degree finite-edge a/b with the identical scalar source geometry
and incident-power normalization.  No production-Q promotion is made by this
checkpoint.
