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
- Phase 3A complete: on the actual saved 0.5 um center-beam temperature, all
  four scaled raw-current derivatives passed adjoint versus central FD with a
  worst best-step component error of `4.58e-6` and second-order FD convergence.
- Phase 3B-1 complete: the fixed-mesh nodal-lumped Robin contact converges to
  the hard node contact.  At `g=1e18 S/m2`, relative errors are `1.03e-5` in
  current and `2.10e-6` in the weighting-potential L2 norm.
- The finite 0.5 um production relaxation is `g=1e12 S/m2`: `0.44%` current
  error at the validation geometry with non-collapsed sensitivities.  The
  independently revalidated 0.25 um relaxation is `g=1e14 S/m2`.  Final
  ranking is always based on hard-contact `abs(I)`.
- Phase 4 complete at 0.5 um: all nine beams were optimized independently with
  12 starts in each of the `+I` and `-I` branches (`216/216` SLSQP successes),
  followed by hard re-evaluation.  New candidates beat legacy DE on four beams;
  the legacy candidate remains the hard winner on five beams.
- The systematic 0.5 um search audit is complete: nested budgets of 12, 24,
  and 48 starts per signed branch produced identical best hard currents for
  all nine beams.  All 864 SLSQP runs succeeded and the declared 0.1% plateau
  gate passed, so the optional 96-start budget was not run.
- The 0.25 um explicit thermal model and electrical model were rebuilt.  The
  corrected production relaxation is `g=1e14 S/m2`; its actual-temperature
  adjoint/central-FD maximum component error is `2.87e-5`.
- The final 0.25 um common-seed transition-width audit passed over
  `0.25, 0.5, 0.75, 1.0 um`: maximum hard-current spread `0.0487%`, maximum
  symmetry-aligned geometry change `0.236 um`, and maximum smooth-hard current
  error `0.411%`.
- Boundary quadrature orders `3,5,7,9` give exactly invariant production
  nodal-lumped current, gradient, weighting potential, and contact integral.
- The 0.25 um designs are refined best-found results, but mesh convergence is
  not yet established: the 0.5-to-0.25 um same-geometry current changes by up
  to `13.0%`, dominated by the thermal discretization at corner beams.
- The next refinement is complete.  At 0.125 um, all nine thermal solves pass,
  the representative relaxation/adjoint gate passes, and all 48 selective
  signed SLSQP runs succeed.  None improves the transferred 0.25 um geometry.
- Direct current convergence is still open: 0.25-to-0.125 um changes are
  `5.14%` (corner) and `2.81%` (x-edge).  A targeted 62.5 nm solve reduces
  these to `1.87%` and `1.02%`; the observed order is approximately 1.4.

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
- `validation/phase3_gradient_check.py`: actual-baseline-temperature raw-current
  adjoint/central-FD check; it does not import or call an optimizer.
- `validation/PHASE3A_ACTUAL_BASELINE_GRADIENT_CHECK.md`: result and limits of
  the first Phase-3 gate.
- `validation/phase3_gradient_check.{json,csv,png}`: full numeric provenance,
  every FD sample, and the componentwise convergence plot.
- `validation/PHASE3B_ROBIN_HARD_CONVERGENCE.md`: fixed-mesh Robin-to-hard
  result, the contact-discretization correction, and the selected finite `g`.
- `validation/phase3_robin_hard_convergence.{py,json,csv,png}`: reproducible
  `g` sweep and machine-readable evidence.
- `optimization/PHASE4_500NM_OPTIMIZATION_REPORT_KO.md`: full Korean account of
  the optimizer, hard ranking, per-beam results, and geometry convention.
- `optimization/SEARCH_PLATEAU_500NM_REPORT_KO.md`: DE-seeded nested
  `12 -> 24 -> 48` systematic search and stopping-rule evidence.
- `optimization/run_center_beam_slsqp_multistart.py`: center-beam pilot.
- `optimization/run_all_beams_slsqp_multistart.py`: nine independent beam runs.
- `optimization/all_beams_slsqp_multistart.{json,png}`: all 216 runs and the
  final comparison with legacy DE.
- `optimization/search_plateau_results.{json,png}`: 864-run nested search
  audit and per-budget hard incumbents.
- `refinement/REFINEMENT_250NM_REPORT_KO.md`: complete Korean 0.25 um account,
  including the corrected relaxation, mesh-effect decomposition, failed first
  transition audit, common-seed closure, final geometry table, and limitations.
- `refinement/transition_width_final_250nm.{json,png}`: accepted nine-beam
  transition-width result after selective second-iteration closure.
- `refinement/boundary_quadrature_order_250nm.json`: joint four-width,
  four-order, nine-beam production invariance gate.
- `refinement/final_refinement_summary_250nm.json`: 0.5-to-0.25 um current and
  symmetry-aligned geometry comparison.
- `refinement_125nm/REFINEMENT_125NM_REPORT_KO.md`: full 0.125 um refinement,
  selective optimization audit, 62.5 nm targeted pilot, and mesh-series result.
- `refinement_125nm/final_125nm.{json,png}`: accepted 0.125 um hard currents
  and proof that the transferred geometries remain the local winners.
- `refinement_125nm/targeted_62p5nm_pilot.json`: direct corner/x-edge 62.5 nm
  evidence; thermal solve passes while the successive 1% mesh gate remains open.
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

## Production sequence

Do not interpret a smooth Robin optimum as the final physical electrode.  Every
accepted candidate is ranked by hard-contact `abs(I)`.  The 0.5 um systematic
search and the 0.25 um transition-width/quadrature gates are complete.  The
next blocker is a direct 31.25 nm corner/x-edge thermal check or an equivalent
AMG/adaptive-refinement implementation.  The electrode geometry is locally
stable through 0.125 um, but the 62.5 nm successive current change is still
`1.87%` at the corner.

Production will run `+I/I_ref` and `-I/I_ref` as separate dimensionless
branches.  `I^2` is diagnostic only.  Center variables are unbounded lifted
periodic coordinates, so SLSQP never sees an artificial `0/P` box seam.
