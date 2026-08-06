# Run 002 — 10 µm Gaussian current maximization

Status: `VALIDATED_SELECTED_OPTICAL_GRADIENT_ADFD`. A homogeneous-air
source-only Maxwell gate, small CUDA thermal forward/adjoint controls,
uniform rho=0/0.5/1 scalar-vs-`importnk2` complex-material controls, and a
matched-volume rho=0.5 production-candidate GPU forward have run. The exact
material-attributed Q has also passed the full production 3D CUDA thermal/PTE
forward and implicit-adjoint controls for four interface-G scenarios. No
optimization solve has started. The selected `373×373` physical-density
optical gradient now passes one directional AD-FD smoke at `h=0.005`; its
corrected one-direction combined Maxwell/CUDA-thermal derivative also passes.
This is not yet the full latent AD-FD certificate.

This is a new physical contract, not a continuation that silently reuses the
4 µm CPU-TFSF certificate.  The requested source is a scalar Gaussian at
10 µm with a target realized waist radius of 8.5 µm.  The optical TaIrTe4
background extends through the transverse PML so no artificial flake edge is
introduced.  The finite thermal flake and substrate remain explicit.

## Frozen planning choices

- candidate source span/domain: 40/48 µm; domain audits: 56 and 64 µm;
- six PML boundaries, 24 layers, no periodic/Bloch boundary;
- complex Kitamura-2007 SiO2 closure at 10 µm, not lossless `n=1.38`;
- 1.0 µm design height baseline; 0.6 and 1.5 µm are pre-optimization
  sensitivity cases;
- 50 nm production design nodes, 500 nm final solid/void DRC, and 525 nm
  differentiable steering target;
- the immutable combined gradient on the coarse 20×20 µm canvas selected and
  froze a centered 18.6×18.6 µm production window before optimization;
- four named bottom/design TaIrTe4-SiO2 interface-G combinations;
- material-resolved TaIrTe4 and SiO2 optical loss must both reach the thermal
  RHS without clipping, gain, or rescaling;
- thermal forward and implicit-adjoint linear solves are CUDA-only in the
  production path.

The initial-FOM strategy uses two separately optimized signed objectives,
fixed nondimensional objective scaling, low-beta asymmetric seeds, a nominal
stage before robust morphology, and multiple starts.  It never dynamically
rescales a gradient to match finite differences.

## Allowed commands now

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python run_optimization.py --setup-audit
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python run_optimization.py --preflight
```

Both commands are solver-free. `--execute` does not exist until the Gaussian
source gate, component-Yee mapping audit, material-resolved Q remap, and CUDA
thermal-adjoint parity are complete.

The first licensed checkpoint is a homogeneous-air, GPU-only source audit:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python audit_source_only_gpu.py \
  --output-dir /absolute/raw/path/run002_source_only \
  --gpu-device "GPU 4" \
  --contract-only
```

Remove `--contract-only` only after the runsetup readback is accepted. The
target-plane waist is measured; an 8.5 um source-object input is not silently
assumed to realize an 8.5 um target-plane waist after discretization.

The next completed licensed checkpoint compares a uniform complex material in
scalar `(n,k) Material` and `importnk2` form at rho=0, 0.5, and 1.  The
rho=0.5 and rho=1 component-grid spatial-Q NRMSE values are at roundoff level,
and their matched-volume six-face closure errors are below 0.016%.  See
`results/COMPLEX_MATERIAL_EQUIVALENCE_REPORT.md`.  This does not certify the
nonuniform density-to-component-Yee Jacobian; optimization remains disabled.

A subsequent isolated-control smoke test constructed explicit sparse
component operators for a nonuniform 101×101 complex density.  Its worst
mapping-only centered-FD error is `1.34e-9`, its worst JVP/VJP dot error is
`7.32e-15`, and its E/index coordinate mismatch is `8.48e-22 m`.  This proves
the construction method but is deliberately not promoted as the final
production-geometry Jacobian.

The matched-volume coarse production candidate uses a 48×48 µm six-PML FDTD
domain, a 20×20×1 µm rho=0.5 design canvas, and a long TaIrTe4 optical
background.  Its GPU forward produced `P_Q=7.296954820427281e-14 W` and
`P_six=7.296652586385535e-14 W`, with `0.004142%` closure and
`7.81123e-8` final auto-shutoff.  The immutable FSP and native component-Q NPZ
are SHA-pinned in the raw-artifact manifest.  This validates only the forward
gate; it does not authorize optimization.

The same completed FSP was then switched to layout and used to construct the
actual 201×201 production component operators without any Maxwell solve.  The
worst five-direction centered mapping-FD error is `1.3371e-9`, the worst
JVP/VJP transpose error is `5.3435e-15`, and the maximum field/index coordinate
mismatch is `6.7763e-21 m`.  Every active sparse-J row lies inside the exact
20×20×1 µm design support.  This closes the density-to-Yee material mapping,
but not the Maxwell/PTE adjoint or conservative thermal-remap gates.

The native component-Q was also partitioned by literal dual-cell/material
volume intersection.  Physical Si, bottom SiO2, finite 32×32 µm TaIrTe4, and
the effective design material receive `98.793556%` of full P_Q.  The artificial
long-TaIrTe4 background contributes `0.010320%`, while the `1.196124%`
air/interface cut-cell remainder is reported rather than forced into a nearby
material.  No Q rescaling was used.  The next gate is deposition of those
material-attributed contributions onto the actual 3D thermal grid.

That deposition passes on the `362×362×91` thermal grid: mapped power is
`7.20892118277057e-14 W`, exactly equal to the material-attributed input at
reported precision, and no nonzero source lies outside its own material. The
frozen thermal boundaries are far-x/y and bottom Dirichlet at 300 K. Every
solid/air exposed face uses a material-specific Robin condition: TaIrTe4 uses
`G=1 W/(m² K)`, while SiO2 and the gray design use `h=10 W/(m² K)`. Internal
interfaces remain explicit resistances, not external boundaries.

The same source was then solved on the production grid for grown/grown,
grown/evaporated, evaporated/grown, and evaporated/evaporated bottom/design
interface combinations. All forward and implicit-adjoint linear solves used
CUDA float64 with no CPU solve fallback. Across the four cases, the worst
linear residual is `9.813e-11`, worst energy-balance error is `1.304e-11`, and
worst Cauchy-normalized reciprocity error is `1.008e-15`. The uniform
45-degree weighting field is `(15625,15625) 1/m`, a unit-potential surrogate
across opposite diagonals of the finite 32 um flake; it is not a full electrode
model. The centered rho=0.5 response is a near-null numerical control, not an
optimized or experimental current. Broader combined Maxwell/thermal AD-FD
evidence still blocks optimization.

The fixed-Q thermal-material derivative is now independently certified on the
coarse 201×201 nodal canvas. Four-node averaging maps it to 200×200 thermal
cells and the exact transpose returns the gradient. At the grown/grown
endpoint the directional AD-FD error falls from `3.563e-5` at `h=0.01` to
`2.229e-6` at `h=0.0025`; the evaporated/evaporated endpoint gives
`1.119e-5` at `h=0.005`. The worst mapping transpose error is `1.811e-15`.
This check includes the rho-dependent gray bulk kappa and TaIrTe4/design
interface G, but freezes Maxwell Q to isolate the thermal branch. It therefore
does not replace the pending combined Maxwell/thermal AD-FD.

The thermal objective derivative has now also been pulled back through the
exact conservative material-intersection deposition to the native component
`Qx`, `Qy`, and `Qz` Yee grids. The memory-bounded transpose uses the same
one-dimensional overlap operators as the forward deposition and has a worst
random dot-test error of `4.097e-15`. Its Cauchy-normalized objective identity
error is `5.104e-14`; the raw relative metric is retained only as a near-null
diagnostic for the centered rho=0.5 control. No Maxwell solve, gradient
rescaling, or optimization occurred in this gate.

The first full-chain physical-density smoke now also passes. At the exact
nonuniform 201×201 baseline, the adjoint-aligned `h=0.005` derivative is
`8.502570281382e-20 A`, while centered FD gives `8.548619467411e-20 A`, a
`0.538674%` relative error below the 1% smoke gate. The initial source-mode
attempts were kept as failures: deleting or disabling the forward Gaussian
changed v261's auto-nonuniform mesh by as much as 87.5 nm. Retaining that
source enabled with amplitude exactly zero during the adjoint preserves the
mesh without adding illumination; the final source/mesh mismatch is at
roundoff and the forward/adjoint field-coordinate mismatch is zero. This is
one direction and one step only, not a multi-direction, gray-law, latent, or
optimization certificate.

The immutable combined physical-density gradient has now frozen the production
window. Every predeclared 12×6 µm strip and the centered 10×10 µm control
failed the 90% absolute-gradient-L1 retention gate. The smallest centered
square on the declared 0.2 µm span sequence that passes is 18.6×18.6 µm:
it retains `90.8872968%`, while 18.4×18.4 µm retains `89.4652232%` and fails.
The resulting production density grid is 373×373 at 50 nm. This was offline
selection from the SHA-pinned combined gradient; it launched no solver and no
optimizer.

The frozen 373×373 production grid now also has a solver-free finite mapping
certificate. A 500 nm conic filter uses zero padding and explicit edge-row
normalization; its exact transpose reverses that normalization order. Opposite
edge wrap and constant-preservation errors are zero, the worst JVP/VJP error is
`1.290e-15`, and the worst finest-step mapping FD error across five directions
and beta=2–32 is `8.706e-6`. The preserved first diagnostic failed only because
it incorrectly gated on the coarsest FD truncation error. This does not certify
exact-binary DRC or full latent Maxwell/thermal AD-FD.

The selected optical environment has now been rebuilt at the actual
18.6×18.6 µm, 373×373-node contract. Its GPU-only rho=0.5 forward completed in
230.229 s with `P_Q=7.219486641789115e-14 W`, six-face closure `6.422e-6`, and
auto-shutoff `8.280e-8`. Component-specific `J_c=d epsilon_Yee,c/d rho` was
then constructed on the same selected support without per-pixel Maxwell
solves. Its worst mapping-only FD error is `1.337e-9`, worst JVP/VJP error is
`1.659e-14`, and no active Jacobian row lies outside the exact support. This
does not yet certify the selected-grid thermal gray law, Maxwell adjoint, or
full latent chain.

The selected optical density and Q now also reach the explicit thermal grid
without reverting to the old 201×201/200×200 contract. The 373 nodes map to
186×186 thermal cells through exact bilinear area averaging, with transpose
error `3.786e-16` and mapping-FD error `1.730e-13`. Literal material-overlap
attribution retains `7.132301206388863e-14 W` as physical thermal source; the
air/interface remainder is reported rather than moved. Conservative 3D
deposition has total power error `3.539e-16`, worst component/material error
`9.190e-16`, and zero source cells outside their material. Thermal gray-law
solver AD-FD remains the next gate.

The selected fixed-Q thermal branch now passes gray-law AD-FD. For
`phi_p(rho)=rho^p`, grown/grown p=1,2,3 finest-step errors are
`2.290e-6`, `2.079e-6`, and `1.451e-6`; the selected
evaporated/evaporated p=1 endpoint gives `2.707e-6`. All thermal linear solves
used CUDA float64 without CPU fallback. Relative to p=1, p=2 and p=3 rotate
the thermal gradient by 5.363° and 9.800°, so p is an explicit material-model
choice. This fixed-Q checkpoint does not include the optical epsilon gray law
and is not a coupled or full-latent certificate.
