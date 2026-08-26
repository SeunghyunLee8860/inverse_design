# Lumerical Au production continuation

Current gate (2026-08-26): **do not launch or resume yet**. The prior run
omitted Au thermopower and is stale. Read `LUMERICAL_ONLY_EXECUTION.md` and
complete fresh Ea/Eb combined Maxwell/custom-PDE AD-FD with `S_Au` before
creating a new output root. The next run must start again from exact uniform
`rho=0.5`; an old MMA checkpoint cannot be reused after changing the objective
and gradient.

This is the production successor to the two-evaluation beta-4 smoke. The
entry point is `41_optimize_lumerical_4um_dualpol_continuation.py`; the site
launcher is `launch_lumerical_continuation_runres_gpu5.sh`.
On this host the launcher explicitly uses the audited reservation module in
`worktrees/pte_true_mma/tools/lumerical_runres`; without that override the
site `/home/dhkim/bin/runres` incorrectly searches the current user's absent
`~/dhkim_module` and exits before reserving a license.

## Optimization contract

- Initial latent and physical state: exact uniform `rho=0.5` at beta 1.
- Maxwell: Lumerical FDTD 2026 R1.2 only.
- Thermal/electrical: repository custom CUDA equations only; no Lumerical
  HEAT or CHARGE.
- Thermoelectric source: both anisotropic TaIrTe4 and in-plane floating-Au
  contributions. The Au bulk-reference scenario is `S_Au=+1.94 uV/K`; the
  numerical void floor has zero thermopower, and the unknown vertical
  Au/TaIrTe4 interface thermopower is explicitly zero rather than guessed.
- Alternative Maxwell solver: forbidden by `lumerical_only_boundary.py`.
- Objective: maximize the epigraph `t` with `t-I_Ea<=0` and `t+I_Eb<=0`.
- Target: `I_Ea>0` and `I_Eb<0`.
- Beta schedule: `1, 2, 4, 8, 16, 32, 64, 128`.
- Optimizer: NLopt `LD_MMA`, with a beta-dependent bounded move region.
- The run is resumable only under the same Git commit. Every completed
  attempt and beta stage has an external checkpoint.

The fabrication constraints activate gradually:

| beta | active Au design constraints |
|---:|---|
| 1, 2 | none; current objective establishes the topology |
| 4 | 250 nm minimum Au feature |
| 8 | 250 nm minimum Au feature and 250 nm minimum void/spacing |
| 16-128 | the two 250 nm constraints plus mean `4*rho*(1-rho)` grayness |

The three-constraint high-beta contract deliberately differs from the
referenced flake optimization. That run used terminal conductance as its
third constraint because the optimized flake joined measurement terminals.
Here Au is a floating absorber/shunt, so imposing terminal conductance would
be physically false. Explicit binarization is the third Au constraint.

## Final promotion

The continuous result is not promoted merely because beta reaches 128.
Promotion requires all of the following:

1. the signed current switch is present in the relaxed final evaluation;
2. calibrated smooth 250 nm solid and void caps pass;
3. mean grayness is no larger than `0.005`;
4. the thresholded 80x80 cell mask passes the independent exact 250 nm
   solid/void morphology audit;
5. `43_certify_lumerical_4um_exact_binary_lateral.py` rebuilds that arbitrary
   mask as coalesced Lumerical rectangles with the ordinary sampled
   dispersive Au material and freshly evaluates Ea/Eb at both 100-nm and
   50-nm flake/design lateral meshes;
6. for Ea and Eb separately, source-normalized Q, flux, complex E, and E2
   change by less than 0.5% from 100 to 50 nm;
7. the 50-nm raw Q passes script 42's adaptive 100/50/25/12.5-nm custom CUDA
   thermal/electrical convergence gate and selects the reference PDE step;
8. the 100-nm raw Q also passes adaptive PDE convergence and is forced through
   at least the fine-selected PDE step;
9. on that identical PDE grid, Ea and Eb each keep current, TaIrTe4
   temperature-field NRMSE, mean temperature, and peak-temperature changes
   below 0.5% from the 100-nm to 50-nm optical mesh;
10. the fine-reference exact-binary currents still satisfy
   `I_Ea>0`, `I_Eb<0`.

There is no post-hoc morphology repair and the final exact evaluation does
not use a gray `importnk` material. Script 42 alone cannot promote a result
because it covers only one optical Maxwell mesh.

Before a latest-commit production launch, set
`AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION` and
`AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION` to passed 50-nm source-only
JSONs from the same physical GPU UUID, accelerator policy, and solver build
as the 100-nm Ea/Eb calibrations. The launcher fails before the first
optimization solve if either path or any mesh/provenance gate is missing.

## Storage policy

The smoke retained about 5 GB per full physics evaluation, while the host had
about 252 GB free when this driver was added. Production therefore retains
the optimizer state, projected density, final gradients, small Jacobians,
JSON, commands, and logs, but removes completed continuous-evaluation `.fsp`,
`.h5`, native-Q raw NPZ, and CUDA-PDE pullback transients. Every removal is
listed in that evaluation's `ARTIFACT_RETENTION.json`. The final exact-binary
evaluation is exempt and retains its complete forward artifacts.

## Status files

The external output root contains:

- `production_manifest.json`: current beta, attempt, currents, FOM,
  constraints, and terminal status;
- `continuation_checkpoint.npz`: restart state;
- `stages/beta_*/stage_result.json`: one attempt's full status;
- `checkpoints/beta_*_completed.npz`: completed-beta handoff points;
- `final_exact_binary_cell_mask.npz` and
  the passing attempt's `exact_binary_certificate/` only after every
  continuous, optical-lateral, PDE-convergence, and sign gate passes. Failed
  beta-128 exact candidates remain in their own attempt directories and can
  never overwrite the final mask. The final mask is written atomically and is
  immutable; if a process dies between the passed manifest and terminal
  checkpoint writes, restart verifies the mask/state SHA and strict signs
  before repairing only the checkpoint.
