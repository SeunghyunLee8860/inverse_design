# Au-on-fixed-TaIrTe4 validation checkpoint

Status: `BLOCKED_LUMERICAL_LICENSE_SESSION_STARTUP`

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

- Status: `BLOCKED_LUMERICAL_LICENSE_SESSION_STARTUP`.
- FDTD solve executed: `False`.
- GPU engine acquired: `False`.
- Installation in the retained result: `/opt/lumerical/v261`.

Both the `/opt/lumerical/v261` installation and the user-owned v261
installation were attempted. Session startup stopped before material import
because ANSYSLI did not create/read its license-sharing port file. This is not
a material-fit failure and it is not an optical validation pass.

## What is deliberately not claimed

- No binary Au/air Maxwell control has run.
- No Au density optical AD-FD has run.
- No Au/TaIrTe4 thermal or electrical contact has been selected or validated.
- No electrode weighting-field gradient has been certified.
- No Au topology optimization has started.

The approved fallback to sharp-interface level-set/shape optimization is used
only if the density route fails material readback, binary endpoint equivalence,
or AD-FD after the license/API gate is restored.
