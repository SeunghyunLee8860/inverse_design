# Run 002 — 10 µm Gaussian current maximization

Status: `PRODUCTION_3D_THERMAL_Q_DEPOSITION_VALIDATED`. A homogeneous-air
source-only Maxwell gate, small CUDA thermal forward/adjoint controls,
uniform rho=0/0.5/1 scalar-vs-`importnk2` complex-material controls, and a
matched-volume rho=0.5 production-candidate GPU forward have run.  No
production thermal, PTE, combined finite-difference, or optimization solve has
started.

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
- a coarse 20×20 µm sensitivity canvas selects a smaller asymmetric design
  window before the iterative optimizer is enabled;
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

That deposition now passes on the `362×362×91` thermal grid: mapped power is
`7.20892118277057e-14 W`, exactly equal to the material-attributed input at
reported precision, and no nonzero source lies outside its own material. The
frozen thermal boundaries are far-x/y and bottom Dirichlet at 300 K, plus a
top exposed Robin boundary with `h=10 W/(m² K)`; internal interfaces remain
explicit resistances, not external boundaries. Temperature/PTE has not yet
been solved.
