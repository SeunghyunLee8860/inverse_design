# Shared Au material-fraction audit

Status: `CODE_PATH_UNIFIED_REVALIDATION_REQUIRED`

## Finding

The historical optimization did not use one relaxed Au fraction.  Maxwell
used `rho**3` for Au oscillator strength while the thermal conductivity,
Au/Ta thermal contact, Au sheet conductivity, and Au/Ta electrical contact
used `rho`.  The committed gray-law factorial shows that this changes the
predicted currents materially and can change the requested sign.

## Code correction

`material_fraction.py` is now the only production definition of relaxed Au
fraction:

```text
f_Au(rho) = rho
```

Maxwell ADE strength and absorption, thermal conductivity and contact,
electrical sheet conductivity and contact, and every corresponding analytic
gradient call this same function.  The derivative is also centralized.  O3 is
retained only inside scripts that explicitly reproduce the historical
diagnostic.

Gray `rho` is not claimed to be a fabricated homogeneous material.  It is a
continuous topology relaxation, and promotion still requires an exact-binary
Au/void geometry and a separate 500 nm solid/void audit.

## Fail-closed consequences

- Historical O3/TE1 forward, AD-FD, optimization, and z-mesh outputs remain
  evidence about the historical code only.
- Production resume now rejects a manifest that lacks the shared-law contract,
  so an O3/TE1 history cannot be silently continued as O1/TE1.
- The shared-law combined gradient and mesh convergence still require new
  certificates before optimization can restart.
- The electrical operator retains tiny conductivity/contact floors to avoid a
  singular floating-void block.  Thus `rho=0` is a numerically regularized
  electrical void, not a mathematically exact disconnected endpoint.  This
  floor sensitivity must be quantified before final promotion.
