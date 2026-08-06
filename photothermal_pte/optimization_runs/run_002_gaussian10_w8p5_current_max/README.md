# Run 002 — 10 µm Gaussian current maximization

Status: `UNIFORM_COMPLEX_MATERIAL_EQUIVALENCE_VALIDATED`. A homogeneous-air
source-only Maxwell gate, small CUDA thermal forward/adjoint controls, and
uniform rho=0/0.5/1 scalar-vs-`importnk2` complex-material controls have run.
No nonuniform material Jacobian, production thermal, PTE, combined
finite-difference, or optimization solve has started.

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
