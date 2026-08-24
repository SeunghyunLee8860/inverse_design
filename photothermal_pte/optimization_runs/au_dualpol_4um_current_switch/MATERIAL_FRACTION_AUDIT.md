# Historical shared-gray Au material-fraction audit

Status: `SUPERSEDED_BY_EXACT_BINARY_AU_GEOMETRY_CONTRACT`

This file records why O3/TE1 was inconsistent. The later shared-linear law was
a useful software diagnostic, but it is not an authorized representation of
physical Au for the Lumerical inverse-design route. See
`LUMERICAL_MAXWELL_GPU_PDE_ROUTE.md`.

## Finding

The historical optimization did not use one relaxed Au fraction.  Maxwell
used `rho**3` for Au oscillator strength while the thermal conductivity,
Au/Ta thermal contact, Au sheet conductivity, and Au/Ta electrical contact
used `rho`.  The committed gray-law factorial shows that this changes the
predicted currents materially and can change the requested sign.

## Historical code correction

`material_fraction.py` became the centralized definition in the legacy gray
code path:
fraction:

```text
f_Au(rho) = rho
```

Maxwell ADE strength and absorption, thermal conductivity and contact,
electrical sheet conductivity and contact, and every corresponding analytic
gradient call this same function.  The derivative is also centralized.  O3 is
retained only inside scripts that explicitly reproduce the historical
diagnostic.

That unification removes the O3/TE1 software mismatch, but it does not turn a
gray cell into exact Au. The legacy path is now production-blocked. New work
must use exact binary dispersive-Au geometry in every physical evaluation;
continuous variables may move a boundary but may not interpolate Au material.

## Fail-closed consequences

- Historical O3/TE1 forward, AD-FD, optimization, and z-mesh outputs remain
  evidence about the historical code only.
- Legacy resume rejects a manifest that lacks its shared-law contract, so an
  O3/TE1 history cannot be silently continued as O1/TE1.
- A shared-law gradient or mesh certificate cannot unblock the new exact-Au
  Lumerical optimization.
- The electrical operator retains tiny conductivity/contact floors to avoid a
  singular floating-void block.  Thus `rho=0` is a numerically regularized
  electrical void, not a mathematically exact disconnected endpoint.  This
  floor sensitivity must be quantified before final promotion.
