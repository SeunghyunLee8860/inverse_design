# Au-on-fixed-TaIrTe4 validation checkpoint

Historical material checkpoint status:
`VALIDATED_AU_MATERIAL_READBACK_DENSITY_PATH_NOT_YET_CERTIFIED`

Current promoted status:
`BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_UNVALIDATED`

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
