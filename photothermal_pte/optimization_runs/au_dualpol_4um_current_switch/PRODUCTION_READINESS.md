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

The unified Lumerical GPU runner now supports source-only, exact
empty/full/simple-L, and imported-density controls for both crystal
polarizations. Its imported-density path now accepts and hash-binds a
nonuniform 81x81 projected nodal checkpoint; previously it could only run a
uniform scalar and therefore could not evaluate an optimizer topology. On the
current RTX host, the explicitly non-promotable Ea/Eb
source-only gate passed, while exact-full Ea on the 20-nm stack-z baseline
failed Q/six-face closure by 30.43%. The linked 5-nm/50-nm z source gate passed,
but its material successor was blocked by temporary shared-license exhaustion.
None of this satisfies the required B200 inventory gate. Every material run
requires a passed, hash-, accelerator-, GPU-UUID-, and solver-matching 4-um
source-only waist/power record. The dynamic preflight also requires all nine
`lum_fdtd_solve` tasks needed by the installed GPU path.

Production remains blocked until a new hash-linked certificate chain proves:

1. the target device geometry, contacts, crystal axes, stack, illumination,
   Au role, and uncertain interface scenarios;
2. on the actual B200, source-only plus empty/imported-full/ordinary
   dispersive-Au 4-um endpoint field/absorption/Q parity for both
   polarizations, with actual full-domain mesh readback and the
   single-frequency carrier's finite-source-band error reported separately;
3. a uniform-density field/Q sweep without an optimizer-exploitable gray
   resonance;
4. the 81x81 nodal-density to 80x80 custom-PDE cell map and exact transpose,
   plus a nonuniform Lumerical component-Yee material Jacobian with centered-FD
   and transpose tests;
5. full optical/thermal/electrical latent AD-FD for `Ea` and `Eb`;
6. optical x/y/full-domain-z/PML, source/time/Q closure, thermal mesh,
   electrical mesh, contact, and void-floor convergence; full-domain z must
   link thin-stack refinement to Si-bulk/air/PML refinement;
7. an independent final 500-nm exact-binary reevaluation using ordinary
   sampled-data dispersive Au.

The historical partial-z, O3/TE1, shared-linear, and FDTDX artifacts cannot
satisfy these gates. They remain diagnostics only.

The Au-specific nonperiodic colored sparse-Jacobian implementation and its
solver-free synthetic FD/transpose tests now exist. Gate 4 is still open
because no completed nonuniform density FSP has yet supplied the actual
Lumerical `index_detail` mesh for that certificate.
