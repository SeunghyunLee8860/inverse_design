# FDTDX gray-Au law quarantine audit

## Outcome

The historical FDTDX continuous-Au routes are closed, not repaired or
selected.  The final machine-readable audit is valid, but it explicitly keeps
the FDTDX optimizer disabled:

- historical optical `rho^3` / thermal-electrical `rho` (`O3/TE1`): forbidden;
- shared-linear gray FDTDX material: forbidden for production;
- patched FDTDX generic continuous `Device` material: forbidden for Au
  production;
- exact integer/bool air-or-ordinary-Au endpoint placement: allowed only as a
  reference calculation;
- strict FDTDX optical mesh: not selected;
- FDTDX optimizer start: forbidden.

This audit did not run or modify Lumerical.  The separate Lumerical session
owns its Maxwell implementation and validation.

## Certificate and provenance

The audit generator and tests were committed and pushed at
`d27cfc82e42c5d0a4a8a8336735415c813b9e847`.  It ran from a clean repository,
used no GPU and no Maxwell/Lumerical solve, and completed in `0.30 s`.

- certificate:
  `/home/seunghyun200/fdtdx_results/fdtdx_gray_law_final_audit_d27cfc82/FDTDX_GRAY_LAW_FINAL_AUDIT.json`
- certificate SHA-256:
  `1d3909f126210a87b5aea182b5993a801b48939343981de436453458df2d8cb9`
- generator SHA-256:
  `7da93e63e1cd643363049b97f9642fa4e3f496dd564c802da75a9e0643d14201`
- status: `VALIDATED_FDTDX_GRAY_LAW_QUARANTINE_AUDIT`
- failed audit checks: none
- full regression suite: `464 passed, 7 subtests passed in 27.41 s`

The status means the quarantine decision and artifact bindings are valid.  It
does **not** mean that an optimization material law or production mesh is
ready.

## What was checked in code

An AST scan of the active FDTDX runtime files found no executable literal
power of three in `material_fraction.py`, `combined_4um.py`,
`multiphysics_4um.py`, or optimizer entrypoints 10/12/13.  This does not
rehabilitate the historical shared-linear gray law: that law remains marked
as a numerical consistency baseline whose gray state is not physical
geometry.

All three optimizer entrypoints call `require_production_readiness()` before
creating output directories or compiling a Maxwell runner:

| entrypoint | readiness line | first output mutation | first Maxwell compile |
|---|---:|---:|---:|
| `10_optimize_4um_dualpol_au_ld_mma.py` | 731 | 733 | 851 |
| `12_optimize_exact_binary_au_topology.py` | 318 | 320 | 331 |
| `13_optimize_robust_binary_au_ld_mma.py` | 377 | 378 | 388 |

Production readiness is false.  In particular, the dispersive density route,
mesh certificate, coupled-gradient certificate, actual-device confirmations,
and selected numerical contract are absent.  Calling any of these optimizers
therefore fails before an output mutation or Maxwell compilation.

The exact-binary FDTDX reference path accepts only an `80 x 80` bool/integer
0/1 mask, rejects even float arrays containing only 0/1, rejects integer
non-endpoints, and maps the mask by integer piecewise-constant replication.
It records `gray_density_allowed=false` and `rho_power=null`.

## One geometry state, different physical properties

The corrected modeling rule is not “use the same exponent in every PDE.”  It
is:

1. define one filtered/projected topology occupancy on the `81 x 81` design
   nodes;
2. map that same state exactly to the `80 x 80` physical cells;
3. derive optical, thermal, electrical, and interface properties from that one
   occupancy with explicit endpoint-consistent constitutive laws;
4. treat intermediate occupancy only as an optimization carrier;
5. promote only after independent exact-binary ordinary-Au reevaluation.

The four-corner nodal-to-cell map passes its transpose test at
`2.8585e-15`; centered finite differences are below `1.43e-8`.  The
solver-free future optical interpolation has no `rho` exponent and passes its
analytic complex derivative check below `1.25e-11`.  These tests verify the
array calculus only.  They do not validate a Maxwell/Yee binding.

It is physically reasonable that the same occupancy produces different
quantities such as complex permittivity, thermal conductivity, electrical
conductivity, and interface conductance.  It is not reasonable to let each
PDE see a different effective geometry through arbitrary `rho` powers.  The
present thermal/electrical conductivity and contact maps are still
provisional because their physical bounds and zero-floor sensitivities have
not been validated on the actual device.

## Paper-backed decision

The five local PDFs were rebound byte-for-byte in the certificate.  The two
Nature Communications detector papers use explicit binary Au resonator
geometries and geometry/orientation sweeps, not a fabricated `rho^3` metal.
The TaIrTe4 device paper uses local thermoelectric current plus a
Shockley-Ramo weighting field, making real electrodes, crystal axes, thermal
gradients, and collection geometry part of the sign prediction.

The topology-optimization literature can use intermediate density, but as a
numerical design carrier with a declared dispersive interpolation,
filter/projection and exact-material promotion.  Christiansen et al. motivate
the future `n,k`-then-square optical interpolation; Zeng and Xu specifically
warn that an unsuitable interpolation can introduce nonphysical field
amplification.  Neither supports the historical `O3/TE1` split.

- local 2022 paper SHA-256:
  `f0f3cef83c5a4dd98302f4f74f265d39b90f2a5fc95cd97bf0d1b1ebaa9fb4d0`
- local 2022 SI SHA-256:
  `927d41f1d6f62916ba15cdf0eb0ec9a37edf457e0cb7b8133365d1d0f13b342b`
- local 2024 paper SHA-256:
  `91f80b23e4ea0b962fb6780df9c511dbc75264ee9677d3317dbaa1a7d1770749`
- local 2024 SI SHA-256:
  `72c2c1264c8d53e4fc22b356fbfd1cf99c229a5b7cabd3f8069f516d156cc2fb`
- local 2026 TaIrTe4 paper SHA-256:
  `7a573dd775483e5c5af3ac95e07554027bad5cd45fb3d72074eddf157ad930ff`

Primary sources:
[Nature Communications 2022](https://doi.org/10.1038/s41467-022-32309-w),
[Nature Communications 2024](https://doi.org/10.1038/s41467-024-51599-w),
[Advanced Functional Materials 2026](https://doi.org/10.1002/adfm.75986),
[Christiansen et al.](https://doi.org/10.1016/j.cma.2018.08.034), and
[Zeng and Xu](https://doi.org/10.1021/acsphotonics.1c00260).

## Mesh and objective consequence

This audit revalidates, by exact certificate hashes, both the blocked strict
optical z-tail and the downstream frozen-Q PTE tail.  No strict FDTDX mesh is
selected.  The user-balanced z2 thin-stack case uses `2.5 nm`, not `5 nm`;
it is only a downstream diagnostic because the z2-to-z4 strict optical
comparison remains blocked.

The exact-L500 diagnostic also gives same-sign currents for both
polarizations (`+6.11455 nA` for Ea and `+6.36632 nA` for Eb at z2), so it
does not meet the requested sign-switch objective.  Those values use virtual
full-flake-edge terminals, not the actual device electrodes, and therefore
must not be reported as a fabricated-device prediction.

## Remaining blockers and next action

No additional FDTDX Maxwell refinement, z64 run, gray optimization, adjoint,
or optimizer timing is justified.  The remaining work is primarily a device
definition and coupled-model problem:

1. obtain and encode the actual flake outline/thickness, a/b-axis angle,
   electrode/pad polygons, signed output contacts, Au electrical role,
   SiO2/Si stack, and beam profile;
2. bind defensible ranges for TaIrTe4-SiO2 and Au-TaIrTe4 thermal contact,
   Au-TaIrTe4 electrical contact, and the present artificial void floors;
3. after the independently owned Maxwell route is validated, rerun thermal
   and electrical mesh/contact/void-floor convergence on the actual geometry;
4. validate the complete coupled latent-variable AD-FD chain and finally
   reevaluate promoted designs with exact ordinary Au.

Until those inputs exist, further mesh refinement would converge the wrong
boundary-value problem more precisely and cannot establish the desired
opposite-current device.
