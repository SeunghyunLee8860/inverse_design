# Lumerical Au production continuation

This is the production successor to the two-evaluation beta-4 smoke. The
entry point is `41_optimize_lumerical_4um_dualpol_continuation.py`; the site
launcher is `launch_lumerical_continuation_runres_gpu5.sh`.

## Optimization contract

- Initial latent and physical state: exact uniform `rho=0.5` at beta 1.
- Maxwell: Lumerical FDTD 2026 R1.2 only.
- Thermal/electrical: repository custom CUDA equations only; no Lumerical
  HEAT or CHARGE.
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
5. `42_evaluate_lumerical_4um_exact_binary.py` rebuilds that arbitrary mask
   as coalesced Lumerical rectangles with the ordinary sampled dispersive Au
   material and freshly evaluates both polarizations through the custom CUDA
   thermal/electrical equations;
6. the exact-binary currents still satisfy `I_Ea>0`, `I_Eb<0`.

There is no post-hoc morphology repair and the final exact evaluation does
not use a gray `importnk` material.

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
  `final_exact_binary_evaluation/` only after every continuous gate passes.
