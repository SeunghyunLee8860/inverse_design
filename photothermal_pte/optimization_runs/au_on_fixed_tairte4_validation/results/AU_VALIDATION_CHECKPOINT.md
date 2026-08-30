# Au-on-fixed-TaIrTe4 validation checkpoint

Historical material checkpoint status:
`VALIDATED_AU_MATERIAL_READBACK_DENSITY_PATH_NOT_YET_CERTIFIED`

Current promoted status:
`BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_NO_STABLE_GPU_DIFFERENTIABLE_AU_PATH`

## What is complete

- Frozen wavelength: `10 um`.
- Ordal Au endpoint: `n+ik = 12.1 + 69.2i`.
- Relative permittivity: `-4642.23 + 1674.64i`.
- Bulk-reference electrical conductivity: `41152263.4 S/m`.
- Bulk-reference thermal conductivity: `317 W/(m K)`.
- Candidate density law: interpolate complex `n`, then use `epsilon=n^2`.
- Offline endpoint, analytic derivative, passivity, and JVP/VJP unit tests pass.

The Au transport values are reference scenarios, not certified thin-film or
Au/TaIrTe4 contact properties. The first electrical control will use
`S_Au=0`; nonzero Au thermopower remains a sensitivity case.

## Lumerical readback

- Status: `VALIDATED_LUMERICAL_AU_MATERIAL_READBACK`.
- FDTD solve executed: `False`.
- GPU engine acquired: `False`.
- Installation in the retained result: `/home/seunghyun/lumerical_r12/opt/lumerical/v261`.

The normal-host v261 session opened successfully.  The exact single-frequency
`(n,k)` endpoint has relative complex-permittivity fit error
`0.479801%` and passes the
0.5% gate.  A global fit of the complete 0.667--286 um Ordal table has
`0.981684%` error at 10 um and
is retained as a failed diagnostic rather than the production endpoint.

## What is deliberately not claimed

- No binary Au/air Maxwell control has run.
- No Au density optical AD-FD has run.
- No Au/TaIrTe4 thermal or electrical contact has been selected or validated.
- No electrode weighting-field gradient has been certified.
- No Au topology optimization has started.

The approved fallback to sharp-interface level-set/shape optimization is used
only if the density route fails material readback, binary endpoint equivalence,
or AD-FD after the license/API gate is restored.

## Later fail-closed optical-gradient result

The density route failed because the uniform `rho=1` Au `importnk2` endpoint
diverged, while the matched exact-scalar Au control remained stable. The
sharp-interface fallback was then tested with independent central finite
differences and one actual GPU FieldRegion adjoint solve.

- 25 nm exact-binary Au forward closure: `0.138667%`.
- Strong central FD (`h=0.05 um`): `-2.9041234736757855e-17 W/um`.
- Candidate sharp-interface AD: `+4.079992905639351e-12 W/um`.
- AD/FD magnitude ratio: approximately `1.405e5`; sign mismatch.
- Final surface-quadrature refinement change: `0.434578%`.

The source round trip, coordinate pairing, and quadrature checks pass, but
the continuous moving-domain Au-loss trace is incompatible with the discrete
conformal-Yee `P_Q` objective. It is not fitted or rescaled. The numerical
details and raw-artifact hashes are published in
`AU_SHARP_INTERFACE_PQ_ADJOINT_REPORT.md` and
`AU_SHARP_INTERFACE_PQ_ADJOINT_RAW_ARTIFACT_MANIFEST.json`.

No Au/TaIrTe4 thermal, electrical, PTE, adjoint, or optimization result is
promoted from this failed gradient checkpoint.

## Final GPU-path diagnosis

The documented temperature-grid coupling was tested as a numerical optical
carrier.  It is not a physical temperature and is never exported to the
thermal solver.  Conformal variant 1 exactly reproduces a moderate complex
endpoint (`epsilon=3.75+2i`) on the component grids, with `0.015910%`
six-face closure and `6.23581e-8` auto-shutoff.  The exact 10-um Au endpoint
diverges for all tested 50-nm-film variants: forward/reverse base direction,
linear/nonlinear table, and 1-K/1000-K numerical carrier spans.  Reducing the
FDTD stability factor from `0.99` to `0.5` still causes divergence near the
same physical time (`2.49e-13 s`), so a large Courant step is not the root
cause.  PVA does not apply this material coupling.

Two independent smooth-3-D checks also fail:

- all field traces from 100 nm inside Au through 100 nm outside Au retain the
  wrong sign relative to central FD;
- the component-wise solver-discrete conformal-epsilon Jacobian changes by
  `68.114%` between geometry steps and also has the wrong sign.

The exact scalar-Au forward model and fixed-geometry material derivative are
preserved, but no tested v261 GPU path is simultaneously exact-Au stable and
differentiable.  See
`AU_TEMPERATURE_CARRIER_AND_SMOOTH3D_DIAGNOSIS_REPORT.md` and its raw-artifact
manifest.  No empirical sign flip, normalization or gradient rescaling is
used.
