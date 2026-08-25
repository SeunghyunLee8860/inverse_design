# Active Lumerical production run

Snapshot time: 2026-08-25 17:32 UTC. The external manifest below is the
authoritative live state; this committed file is only the launch handoff.

- Branch: `agent/optimize-au-dualpol-4um-pte`
- Immutable run commit: `790a5ade69307ed1cf7ac5a9cbf3f9011d3321dc`
- Detached run worktree:
  `/home/seunghyun/tairte4/worktrees/au_lumerical_continuation_790a5ade`
- tmux session: `au4um_lum_prod_790a5ade`
- GPU: physical GPU 5, RTX 6000 Ada,
  `GPU-aa047452-9c73-d10f-675f-8af800915acf`
- Reserved license project: nine `lum_fdtd_solve` tasks held by `runres`
- External output root:
  `/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_790a5ade6930`
- Live manifest:
  `.../continuation_790a5ade6930/production_manifest.json`
- Resume checkpoint:
  `.../continuation_790a5ade6930/continuation_checkpoint.npz`

At this snapshot the job was in the first beta-1, exact-uniform-rho=0.5 Ea
forward. No current or FOM had been produced yet. The GPU engine log proved
the requested GPU UUID and reported normal simulation progress. There was no
optimization error.

Monitor without changing the run:

```bash
tmux capture-pane -pt au4um_lum_prod_790a5ade -S -120
jq '{status,latest,stage_count:(.stages|length),error}' \
  /home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_790a5ade6930/production_manifest.json
```

Do not run another copy against the same output root. If the process exits,
rerun the committed launcher from the same detached worktree; the driver will
accept only its same-commit checkpoint. Do not delete the `runres` parent
while a child is alive because the parent owns and releases the reservation.


## Post-launch audit -- 2026-08-25 20:27 UTC

No terminal manifest or result commit from the other RTX host was visible on
GitHub at this audit. The active output/tmux paths above are not mounted on the
current `seunghyun200@dgx-b200` host, and this host reports zero available
Lumerical solve licenses. Do not claim that the production run passed until
its external `production_manifest.json` is retrieved.

The shared branch now includes four post-launch, solver-free fixes:

- `5b9bf35c`: checkpoint stage DFM/grayness caps before the first Maxwell
  solve and key initialization from checkpoint state rather than an artifact
  directory suffix;
- `88492f00`: require both numerical gates and `I_Ea>0`, `I_Eb<0` before the
  standalone exact-binary evaluator returns `passed=true`;
- `fda31cf0`: parameterize the exact-binary mesh while preserving the
  production defaults;
- `12461bda`: add fail-closed `xy` refinement comparison for source-normalized
  Q, flux, complex E, and E2 with the existing 0.5% gate.

The full solver-free regression is `263 passed`. These newer commits do not
mutate the active immutable `790a5ade` worktree or its external checkpoint.

### Required final-mask lateral gate

The current run uses a 100-nm flake/design x-y mesh and is candidate generation,
not a mesh-converged final certificate. Once it produces
`final_exact_binary_cell_mask.npz`, do all of the following on the same
verified-idle physical GPU and solver build:

1. Run `25_run_lumerical_4um_exact_au_control.py --case source_only` for Ea and
   Eb with mesh label `fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps`, flake
   x-y 50 nm, outer x-y 200 nm, stack z 2.5 nm, and bulk z 50 nm. Each exact
   run must use its matching new source-only JSON; the 100-nm calibration is
   prohibited.
2. Run `42_evaluate_lumerical_4um_exact_binary.py` on the unchanged mask with
   those two fine source JSONs and the same 50/200/2.5/50-nm mesh arguments.
3. For Ea and Eb separately, compare the 100-nm and 50-nm forward JSONs with
   `27_compare_lumerical_4um_control_pair.py --refinement-axis xy`. Both
   comparisons must pass all four 0.5% Maxwell gates, and the fine exact
   result must retain `I_Ea>0`, `I_Eb<0`.

This closes only optical lateral convergence of the final mask. The custom
thermal/electrical operators still use a 100-nm core grid; their independent
100-to-50-nm temperature/current mesh convergence remains open and must not be
hidden by the optical pass.

## Replacement production run -- 2026-08-25 20:54 UTC

The original immutable `790a5ade` run stopped during evaluation 0 because its
physical contraction was almost exactly zero at uniform `rho=0.5`, while the
gate divided the transpose mismatch by that cancellation-dominated signed
value. The large positive/negative terms were about `1.43e-8` in aggregate,
the signed contraction was about `1.71e-16`, and an absolute mismatch of only
order `1e-25` therefore produced a misleading signed relative failure.

Commit `69b2bb4098523162421fa6078a3b1c1b3499dd00` replaces that ill-conditioned
gate with the standard Cauchy--Schwarz normwise bilinear-adjoint error. It
retains signed relative error and cancellation ratio as diagnostics; it does
not relax the `1e-12` threshold. The failed Ea artifact passed the repaired
gate offline with normwise error `1.194e-18`, and the Lumerical-specific
solver-free regression passed `110` tests.

The active replacement is:

- immutable run commit: `69b2bb4098523162421fa6078a3b1c1b3499dd00`;
- detached worktree:
  `/home/seunghyun/tairte4/worktrees/au_lumerical_continuation_69b2bb40`;
- tmux session: `au4um_lum_prod_69b2bb40`;
- output root:
  `/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_69b2bb409852`;
- manifest: `.../continuation_69b2bb409852/production_manifest.json`;
- resume checkpoint: `.../continuation_69b2bb409852/continuation_checkpoint.npz`;
- physical GPU 5, UUID
  `GPU-aa047452-9c73-d10f-675f-8af800915acf`;
- nine reserved `lum_fdtd_solve` tasks owned by the `runres` parent.

Evaluation 0 at beta 1 completed all two-forward, two-custom-CUDA-PDE,
component-Yee-Jacobian, and two-Maxwell-adjoint gates. It reported
`I_Ea=-1.709981e-7 nA`, `I_Eb=+8.443054e-7 nA`, and balanced utility
`min(I_Ea,-I_Eb)=-8.443054e-7 nA`; opposite switching is not yet present at
the exact-uniform starting point. Wall time was `1089.50 s`. The repaired
normwise contraction errors were `1.194e-18` for Ea and `4.183e-18` for Eb.
The optimizer then started evaluation 1 at a different density hash, proving
that the first design update is underway.

Monitor the replacement without launching another copy:

```bash
tmux capture-pane -pt au4um_lum_prod_69b2bb40 -S -120
jq '{status,latest,stage_count:(.stages|length),error}' \
  /home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_69b2bb409852/production_manifest.json
```
