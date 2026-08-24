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
and the later exact-full retry completed but still failed closure by 29.239%.
Exact-empty passed with 0.01636%; CV0/CV1/staircase, 2.5-nm z, and strict
time/decay checks did not fix full Au. The root cause was the Au sampled-data
MCM20 overfit branch. MCM4/6/8/12/16 form a common stable plateau and MCM6
passes Q/flux closure at 0.08935%, so Au now defaults to six maximum
coefficients. The rho=1 imported carrier passes its own closure but still has
1.849% complex-field endpoint error versus exact MCM6. Exact/import endpoint
parity and production spatial convergence therefore remain open; see
`AU_MCM_FIT_FINDINGS.md`.
The subsequent MCM6 z study passed the isolated bulk/air/PML 50-to-25-nm
refinement but failed the thin-stack 5-to-2.5-nm refinement: normalized Q,
complex endpoint field, and E2 changed 1.3298%, 0.9850%, and 1.1618%,
respectively, against 0.5% gates. The linked 5/50-to-2.5/25-nm pair also
failed, so 2.5/25 nm is not a converged mesh. The earlier strict 2-ps run used
rejected MCM20 and cannot close the MCM6 time axis. See
`LUMERICAL_Z_MESH_FINDINGS.md`.
The correct MCM6 source-only/empty/full duration pair has now been run. For
exact-full, 1 ps/1e-7 versus 2 ps/1e-9 changed normalized Q by 0.00456%, flux
by 0.01086%, complex endpoint field by 0.00184%, and E2 by 0.00135%; exact
empty also passed. The RTX Ea time axis is therefore closed, while z remains
failed. See `LUMERICAL_TIME_CONVERGENCE_FINDINGS.md`.
The subsequent fixed 5/50-nm interface triage found that CV0 and staircase
agree below 0.15% in all tested source-normalized Maxwell metrics, but CV0's
official exact-index material filter leaves 11.8313%/7.7844% of empty/full
absorption unassigned. CV1 both differs optically and leaves 6.2624%/11.2663%
unassigned. Staircase reduces omission to 0.001012%/0.195399% and is therefore
the next linked-z development candidate. This is not a convergence or
production certificate; see `LUMERICAL_INTERFACE_METHOD_FINDINGS.md`.
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
