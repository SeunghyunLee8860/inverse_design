# Production inverse-design readiness

Status: `BLOCKED_LUMERICAL_DISPERSIVE_DENSITY_ROUTE_NOT_VALIDATED`

All legacy production entry points (`10`, `12`, and `13`) call
`require_production_readiness()` before creating output directories or
compiling a Maxwell runner. There is no environment-variable bypass. The
readiness audit contains an unconditional
`lumerical_dispersive_density_route_validated=false` gate because those
entry points still run the historical FDTDX/shared-linear path.

The selected Lumerical optical constitutive law is now implemented in
`au_density_relaxation.py`:

```text
rho_bar -> n(rho_bar)+i k(rho_bar) -> epsilon=[n+i k]^2
```

It has exact background and frozen Ordal-Au endpoints at 4 um, an analytic
complex derivative, no `rho**3`, and solver-free passivity/FD tests. This is
implementation progress, not a production certificate.

Production remains blocked until a new hash-linked certificate chain proves:

1. the target device geometry, contacts, crystal axes, stack, illumination,
   Au role, and uncertain interface scenarios;
2. on the actual B200, empty/imported-full/ordinary dispersive-Au endpoint and
   source-band parity for both polarizations;
3. a uniform-density field/Q sweep without an optimizer-exploitable gray
   resonance;
4. a nonuniform Lumerical component-Yee material Jacobian with centered-FD and
   transpose tests;
5. full optical/thermal/electrical latent AD-FD for `Ea` and `Eb`;
6. optical x/y/z/PML, source/time/Q closure, thermal mesh, electrical mesh,
   contact, and void-floor convergence;
7. an independent final 500-nm exact-binary reevaluation using ordinary
   sampled-data dispersive Au.

The historical partial-z, O3/TE1, shared-linear, and FDTDX artifacts cannot
satisfy these gates. They remain diagnostics only.
