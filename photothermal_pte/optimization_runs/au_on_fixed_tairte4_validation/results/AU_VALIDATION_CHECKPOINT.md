# Au-on-fixed-TaIrTe4 validation checkpoint

Status: `VALIDATED_AU_MATERIAL_READBACK_DENSITY_PATH_NOT_YET_CERTIFIED`

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
