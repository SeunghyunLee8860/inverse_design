# TaIrTe4 per-beam boundary-electrode adjoint design

This is an isolated follow-on project.  The baseline
`/home/seunghyun/tairte4/pte_electrode_optimizer` is imported read-only and was
not modified.

When this folder is copied into another repository, point the project to that
read-only baseline checkout with
`TAIRTE4_PTE_BASELINE=/path/to/pte_electrode_optimizer`.  The baseline itself
and its large saved thermal fields are intentionally not duplicated here.

## Current status

- Phase 1 complete: line-by-line FEM/electrode/objective audit and a numerical
  node-snapping experiment.
- Phase 2 complete: periodic full-perimeter contact model, compact smooth mask,
  Robin weak form, signed-current adjoint, dimensionless optimizer API,
  seam-free lifted centers, analytic constraints, and hard-contact validation.
- Phase 3 not yet claimed complete: `src/` contains only an isolated assembly
  prototype for the forthcoming FD/mesh/convergence tests.
- Production integration, SLSQP multi-start runs, and DE comparisons have not
  been run in this phase.

The workflow is **per beam**: each Gaussian center gets its own fixed thermal
field and its own independently optimized `(c0,L0,c1,L1)`.  It is not a
mean-current optimization, and `L0` and `L1` need not be equal.

## Documents

- `PHASE1_2_REPORT_KO.md`: Korean explanation and decisions.
- `audit/PHASE1_CURRENT_IMPLEMENTATION_AUDIT.md`: detailed audit, measured
  invariants, literature cross-check, and known model assumptions.
- `derivation/PHASE2_FORMULATION.md`: mathematical formulation and adjoint.
- `configs/phase3_validation.json`: declared mesh/quadrature/transition/FD gate
  matrix for the next phase.
- `tests/test_scaled_contract.py`: pre-Phase-3 signed-branch, scaling, periodic
  constraint, and seam-free-coordinate contract tests (no FD claim).
- `audit/audit_current.py`: reproducible audit program.
- `audit/snapping_sweeps.png`: measured staircase objective/contact behavior.

## Reproduce Phase 1

Use the existing scientific Python environment:

```bash
cd /home/seunghyun/tairte4/pte_electrode_boundary_adjoint
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python audit/audit_current.py
```

From a relocated checkout, use:

```bash
TAIRTE4_PTE_BASELINE=/path/to/pte_electrode_optimizer \
  python -m pytest -q
```

The script reads the already-computed 0.5 um beam fields; it does not rerun or
modify the baseline optimization.

## Deliberate stopping point

Do not interpret the prototype Robin result as the final physical electrode.
Phase 3 must first pass adjoint-versus-central-FD checks for all four variables,
both signed branches, mesh refinement, boundary-quadrature convergence,
smoothing-transition convergence, `g` continuation to hard Dirichlet,
swap/reflection/zero-source tests, and exact evaluation of a legacy DE
geometry.  Only then should a gradient optimizer be enabled.

Production will run `+I/I_ref` and `-I/I_ref` as separate dimensionless
branches.  `I^2` is diagnostic only.  Center variables are unbounded lifted
periodic coordinates, so SLSQP never sees an artificial `0/P` box seam.
