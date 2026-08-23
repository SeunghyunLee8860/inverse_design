# Time/absorption closure findings

Status: `BLOCKED_LONG_TIME_FDTD_INSTABILITY`

This diagnostic used the historical factor-8 partial material-z grid only to
isolate time and absorption behavior. It is not a mesh certificate and uses
no downstream thermal/electrical current.

## Findings

- The closed-surface phasor detector initially reported four times the
  time-domain flux because the pinned FDTDX implementation omits Tukey weights
  while retaining coherent-gain normalization. The repository now uses a
  rectangular switch-only window for this detector. At 24 periods the
  phasor/time-domain flux difference fell from about 75% to 0.3039%.
- The 24-period field is not stationary even though integrated Q changes only
  0.1436%: the volume-weighted spatial Q NRMSE is 1.6273%, above the 0.5%
  gate.
- The coupled solve becomes unstable at longer time. For 32/4, 40/4, and
  40/8 (total/window periods), spatial Q NRMSE is 45.61%, 97.13%, and 98.89%.
  The time-domain closed flux becomes negative and reaches `-9.47e-10 W` in
  the 40/4 case, while late target-Q grows to `6.75e-12 W`.
- Q computed from continuous target permittivity differs from Q computed from
  the realized float32 discrete-ADE susceptibility by 1.11--1.14% in every
  case. Production heat generation must use the realized discrete-ADE loss,
  followed by renewed AD-FD validation.

## Consequence

Do not run or interpret optical mesh convergence until the long-time
instability is isolated and eliminated. The next diagnostic must separate
substrate-only, Au-only, TaIrTe4-only, and full dispersive cases and test a
smaller Courant factor. After a stable time contract is selected, recalibrate
the source on every candidate mesh and start full-domain z convergence.

Per-case Q arrays are restart caches under the configured raw root and are not
tracked by Git. Their paths and SHA-256 values are recorded in the small JSON
manifests in this directory.
