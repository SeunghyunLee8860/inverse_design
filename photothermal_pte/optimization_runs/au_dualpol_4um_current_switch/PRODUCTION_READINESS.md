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
production certificate. Its first 5/50-to-2.5/25-nm linked pair has now
failed every empty/full Maxwell metric (0.6656--1.3954% against a 0.5% gate),
and the next 2.5/25-to-1.25/12.5-nm pair also fails (empty E2 0.6013%;
full metrics 0.5268--0.6884%). A still finer staircase pair is required
before any downstream or x/y gate; see
`LUMERICAL_INTERFACE_METHOD_FINDINGS.md`.
The staircase 0.625/6.25-nm source/empty/MCM6-full set is now complete. Its
pair with 1.25/12.5 nm passes every empty/full Maxwell scalar and endpoint
field gate (maximum change 0.3436%). The official-Pabs custom-CUDA downstream
diagnostic revealed an x-direction material-mask bias because `index_x` is
x-staggered. The replacement component-Yee map pairs `Qx/Qy/Qz` with
collocated fitted `epsilon_x/epsilon_y/epsilon_z`. It conserves native Q below
`3e-15`, leaves at most `1.9e-16` relative absorption unassigned, and restores
the symmetric empty/full current controls to `1.12e-10`/`1.48e-8`, both far
below one ppm. Temperature and Tmax convergence also pass. Only the empty/full
remapped-source L2 NRMSE remains failed at 1.5580%/1.2458%; the three-mesh
source error is nearly first-order. Production z convergence therefore
remains blocked only by this strict volumetric interface-source gate and the
still-untested cases, not by a zero-current or material-identity defect. A
0.3125-nm source-only grid passed, but its material run projected about nine
hours per case and was intentionally stopped.
The same nonuniform Ea forward has now passed the relaxed-density Q/PDE gate.
All three native Yee Q components were deposited into thermal-cell power by
exact overlap without an exact-material equality filter, then differentiated
through the custom CUDA thermal/electrical forward and adjoint systems. Total
Q conservation and the remap transpose were zero-error at reported precision;
the native-Q/thermal-adjoint contraction error was `1.59e-16`. The invocation
took 20.79 s, ran no new Maxwell solve, and used no Lumerical HEAT/CHARGE
license. A subsequent R1.2 FieldRegion adjoint and independent centered
projected-density pair now pass complete Ea AD-FD with relative error
`2.207e-5` and equal sign, without empirical scaling. This is one direction
on the RTX development mesh, not mesh or B200 production evidence.

The old optimizer/DFM carrier was also found to be 80x80 cell-centered and
therefore incompatible with the 81x81 nodal Lumerical state. The new
`lumerical_4um_design_mapping.py` keeps latent/filter/projected arrays nodal
and derives 80x80 PDE/DFM cells only through the exact four-node average and
transpose. Its solver-free filter/projection/cell/DFM directional-FD audit
passes, including removal of the former ReLU nondifferentiability. The
subsequent complete beta-4 latent Ea chain also passes centered AD-FD:
AD `-2.766595495e-8 A`, FD `-2.766380278e-8 A`, equal sign, and relative
error `7.779e-5` (0.00778%). Its filter/projection JVP and VJP contractions
agree to `1.20e-16`. The four Maxwell solves used about 237 s of solver time,
not a multi-hour convergence run. This covers one Ea direction on the RTX
development mesh.
The same complete beta-4 latent gate now also passes for one Eb direction:
AD `-5.529878050e-8 A`, FD `-5.529062519e-8 A`, equal sign, and relative
error `1.4748e-4` (0.01475%). Its filter/projection transpose error is
`1.20e-16`; the plus/minus signal is 1.758% of current magnitude. The four
Eb Maxwell solves used about 269 s of solver time and the three custom-CUDA
evaluations used about 55 s. The material Jacobian was reused only after the
projected density, component epsilon hashes/shapes, Yee coordinates, and
frequency matched; polarization-dependent E/Q arrays were correctly allowed
to differ. Neither polarization used Lumerical HEAT/CHARGE or an FDTDX Maxwell
solve. These initial derivative certificates are not a demonstration
of the required signed switching objective: the unoptimized baseline currents
are both negative (`Ea=-8.334 nA`, `Eb=-15.591 nA`). The objective-level and
second-direction extensions follow below.
The hash-bound Ea/Eb artifacts were then combined by script 39 without any
new solve. The exact epigraph is `t-I_Ea<=0`, `t+I_Eb<=0`; its latent
constraint gradients are `-dI_Ea/drho` and `+dI_Eb/drho`. At the common
baseline, the balanced utility is `-8.334 nA` and Ea is active. The combined
balanced-objective directional AD-FD error is `7.779e-5`; the two epigraph
constraint errors are `7.779e-5` and `1.4748e-4`.
A four-member deterministic smooth direction family was then added while
preserving the original direction-0 hash. Direction 1 is nearly orthogonal to
direction 0 and also passes: Ea AD `1.966482804e-8 A`, FD
`1.966344926e-8 A`, error `7.011e-5`; Eb AD `3.871015880e-8 A`, FD
`3.870857434e-8 A`, error `4.093e-5`. Its signed balanced-objective and both
constraint errors pass as well. The four new perturbed Maxwell forwards
totaled 225.7 s and the four custom-CUDA evaluations 71.7 s.
Direction 2 also passes: Ea error `9.981e-5`, Eb error `1.890e-4`, with
matching signs and a passed signed-objective/constraint audit. Its four
Maxwell forwards totaled 229.2 s and its four custom-CUDA evaluations 72.7 s.
Direction 3 also passes: Ea AD/FD
`-7.435606117e-9`/`-7.434900185e-9 A` (error `9.494e-5`) and Eb AD/FD
`-1.759945920e-8`/`-1.759713187e-8 A` (error `1.322e-4`). Its signed
balanced-objective and both constraint checks pass. The four forwards totaled
232.3 s and the four custom-CUDA evaluations 75.4 s. Ea and Eb therefore each
have all four planned independent directions on this development mesh. The
fail-closed Lumerical evaluation driver, mesh selection, and B200 repetition
remain open; no optimizer was run and switching has not yet been achieved.
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
   electrical mesh, contact, and void-floor convergence; the remaining
   thin-interface volumetric-Q gate must be bounded without committing to
   nine-hour-per-control brute-force runs;
7. an independent final 500-nm exact-binary reevaluation using ordinary
   sampled-data dispersive Au.

The historical partial-z, O3/TE1, shared-linear, and FDTDX artifacts cannot
satisfy these gates. They remain diagnostics only.

The Au-specific nonperiodic colored sparse-Jacobian implementation and its
solver-free synthetic FD/transpose tests now exist. A completed hash-linked
nonuniform density FSP has also supplied an actual 5/50-nm staircase
Lumerical `index_detail` development mesh: the material map passed with worst
FD/transpose errors `3.54e-11`/`3.96e-16` and zero Maxwell solves during the
Jacobian build. Gate 4 remains open for the ultimately selected mesh and B200,
and this result does not close the Maxwell field or combined AD-FD gates.
