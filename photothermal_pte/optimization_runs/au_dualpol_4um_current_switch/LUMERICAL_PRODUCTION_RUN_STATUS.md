# Active Lumerical production run

## Optimizer restart correction -- 2026-08-26 22:42 UTC

**THE `417372bd` PROCESS WAS INTENTIONALLY STOPPED AND MUST NOT BE
RESUMED.** Its last completed stage was beta 16 attempt 1, with
`I_Ea=+0.401006790 nA`, `I_Eb=-0.005742063 nA`, balanced FOM
`0.005742063 nA`, and grayness `0.982287727` against an infeasible fixed cap
of `0.35`. The manifest still says `RUNNING` because SIGINT interrupted the
Python process between atomic manifest updates; process inspection confirmed
that its runres, optimizer, Jacobian, and Lumerical children are all gone.

The complete stage audit showed that every finished beta 1--16 attempt ended
only at `NLOPT_MAXEVAL_REACHED` (result code 5). Beta was allowed to advance
without an objective-plateau gate, and every attempt permanently restricted
all 6561 latent variables to a narrow box around that attempt's start. Even
the union of all three beta-16 retry boxes had a rigorous grayness lower bound
of about `0.422`, so the hard `0.35` cap could not be reached. This was an
optimizer-policy failure, not a Maxwell/PDE or AD--FD failure.

The replacement code starts again from exact uniform `rho=0.5` and changes
the policy as follows:

- the initial exact linearized max-min symmetry-breaking step is `0.05`
  instead of `0.00625`;
- MMA uses physical latent bounds `[0,1]`; the beta-dependent values are
  initial step sizes, not permanent stage boxes;
- first-attempt budgets are 12/10/10/10/12/12/14/16 complete physics
  evaluations for beta 1/2/4/8/16/32/64/128;
- beta advancement requires passed design constraints, the target current
  signs, and a plateau of the actual feasible balanced utility;
- the checkpoint selects the highest-FOM feasible evaluated point rather than
  blindly accepting NLopt's terminal point;
- grayness is tightened by constraint continuation at beta 16/32/64/128,
  while the final beta-128 promotion gate remains exactly `0.005`;
- the three fabrication constraints remain 250-nm solid, 250-nm void, and
  grayness. No Au-terminal conductance constraint is introduced.

The focused regression passed `37` tests and the expanded Lumerical/custom-PDE
suite passed `178` tests. A solver-free runtime preflight passed all
Lumerical-only, 100/50-nm calibration, CV0, 2.5/50-nm z-mesh, GPU UUID,
no-HEAT/CHARGE, and no-FDTDX gates. The next section is retained as the
historical record of the stopped run and is no longer the active instruction.

## Active S_Au production -- 2026-08-26 06:18 UTC

**RUNNING.** The corrected `S_Au` continuation is active from exact uniform
latent/projected `rho=0.5` at beta 1. It passed the solver-free preflight and
successfully transitioned the same checkpoint to the full run. The uniform
evaluation 0000 completed in 1089.79 s with currents numerically at zero. The
bounded max-min warm start then completed evaluation 0001 in 1105.70 s and
already produced the target signs:

- `I_Ea=+0.0231089668 nA`;
- `I_Eb=-0.0231527341 nA`;
- balanced FOM `+0.0231089668 nA`;
- `opposite_current_switching_achieved=true`.

The warm-start latent range is `0.49375--0.50625`, so this is a beta-1
continuous switching point, not a fabricated binary result. Evaluation 0002
is active at this snapshot. The run must continue through all beta/DFM/
grayness and exact-binary promotion gates.

- immutable run commit: `417372bd73dc1af395b2fed2aac87f80a6be5d2f`;
- detached worktree:
  `/home/seunghyun/tairte4/worktrees/au_lumerical_sau_prod_417372bd`;
- tmux session: `au4um_lum_sau_prod_417372bd`;
- GPU: physical GPU 5, RTX 6000 Ada,
  `GPU-aa047452-9c73-d10f-675f-8af800915acf`;
- runres project: `PROJECT_seunghyun_au4um_lumerical_beta_continuation_gpu5_3563961`;
- external output root:
  `/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_sau_417372bd`;
- manifest: `.../continuation_sau_417372bd/production_manifest.json`;
- checkpoint: `.../continuation_sau_417372bd/continuation_checkpoint.npz`.

Monitor without changing the run:

```bash
tmux capture-pane -pt au4um_lum_sau_prod_417372bd -S -120
jq '{status,latest,stage_count:(.stages|length),error}' \
  /home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_sau_417372bd/production_manifest.json
```

Do not start a second process on this output root and do not attach any old
checkpoint. The runres parent owns the nine-task reservation. The production
commit includes the preflight-to-full-run fix and passed 163 Lumerical-only
tests before launch.

## Superseding status -- 2026-08-26

**SUPERSEDED STOPPED RUN.** The earlier continuation
under commit `ac077e4c` was stopped intentionally after an audit found that
the current included TaIrTe4 thermopower but omitted the floating Au sheet's
Seebeck source. Its external artifacts remain under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_production/continuation_ac077e4c8f2d`.
They are diagnostic evidence only and must not be resumed or promoted.

The corrected code uses Lumerical FDTD for Maxwell, custom CUDA PDEs for
thermal/electrical physics, and `S_Au=+1.94 uV/K`. At commit `80e3ef8a` it
passed the full combined Ea/Eb Lumerical-Maxwell/custom-PDE AD-FD through the
active 250-nm filter/projection mapping. The errors were `4.6651e-5` for Ea
and `1.1711e-4` for Eb. The signed balanced objective and both epigraph
constraints passed the 1% gate. No alternative Maxwell solver and no
Lumerical HEAT/CHARGE solver was called.

The validation baseline was `I_Ea=-7.667768 nA` and
`I_Eb=-14.655207 nA`. It is a beta-4 derivative test state, not an optimized
candidate; opposite-current switching has not yet been achieved by that
state. The 50-nm source-calibration blocker was closed later on 2026-08-26 at
commit `52b6ed79`. Ea and Eb both passed on physical GPU 5 with solver build
`8.35.4522`; their solver times were 82.59 s and 82.51 s. The common external
root is
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/r12_gpu5_xy50_source_only_cv0_52b6ed79`.
At this snapshot the numerical pre-launch gates are closed, but the new
production optimizer has not yet been launched.

The next allowed production launch must satisfy all of the following:

1. use the passed Ea and Eb 50-nm source-only calibrations from the root above;
2. commit and push this passing handoff state;
3. use a detached worktree at that exact commit;
4. use a new empty external output root;
5. start from exact uniform `rho=0.5`, beta 1.

Do not attach the new code to any earlier continuation checkpoint. The current
driver also refuses cross-commit resume.

The remainder of this file is historical launch chronology.

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

At that post-launch snapshot this closed only optical lateral convergence of
the final mask; the custom thermal/electrical 100-to-50-nm convergence path
was still open. The later adaptive implementation and script 43 integration
below close the code path, but the actual final-mask Ea/Eb result remains
unmeasured.

That paragraph describes the manual recovery path for the older immutable
run. Commit `e284ea08` introduced script 43 with the four exact 100/50-nm
forwards, both optical comparisons, and the fine-Q adaptive PDE sequence.
Commit `628a88da` additionally runs the coarse optical Q through adaptive PDE
and compares same-grid current/temperature against the fine optical result.
Script 42 by itself declares `final_lateral_certificate_claimed=false`.

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


## Final-mask custom PDE convergence path -- 2026-08-25 21:27 UTC

Commits `fb7ff239`, `b5b18174`, `27cf25fc`, `36d43542`, and
`d1ba0fd0` close the *implementation* gap for the custom thermal/electrical
lateral-grid check. They do not claim that the unavailable final Ea/Eb
artifacts have passed it.

- The exact 80x80 binary Au mask is replicated without interpolation on
  100/50/25/12.5-nm PDE grids. The design grid can reach 640x640 and the
  16-um TaIrTe4 grid can reach 1280x1280 while all physical spans, layer
  thicknesses, conductivities, interface conductances, contacts, and boundary
  conditions remain fixed.
- Each polarization uses one unchanged native Lumerical component-Yee raw Q.
  That Q is exact-overlap remapped independently to every executed thermal
  grid. No clipping, smoothing, gain, or global rescaling is allowed.
- The evaluator always runs 100 and 50 nm. If their pair fails, it adds 25 nm;
  if 50/25 also fails, it adds 12.5 nm. It stops at the first passing adjacent
  pair. The finer member becomes the reported reference. Every executed PDE
  resolution must pass remap/residual/energy/terminal gates, preserve current
  sign, and keep current change, aligned TaIrTe4 temperature-field NRMSE,
  mean-temperature change, and peak-temperature change below 0.5%.
- The result saves hash-bound evidence for every executed resolution. A copied
  raw NPZ is accepted only when its recorded byte size and SHA-256 match.
  Reused forward JSONs must also match mesh spec, accelerator policy,
  polarization, canonical binary-mask hash, and unmodified-Q policy.
- Custom CUDA execution now fails unless `CUDA_VISIBLE_DEVICES` contains
  exactly one physical index equal to `--gpu-index`; the actual GPU index,
  UUID, model, and local device 0 mapping are written to the result.

A custom-CUDA smoke used verified-idle B200 GPUs and the same continuous
asymmetric 285-uW synthetic heat source at all levels. Solver times were
4.03 s (100 nm), 8.60 s (50 nm), 30.86 s (25 nm), and 163.88 s (12.5 nm).
The complete 25-nm process used 42.39 s and 18.31 GB peak host RAM; 12.5 nm
used 3 min 28.30 s and 71.03 GB. A separate 12.5-nm rerun on physical GPU 7
observed 24,524 MiB of device memory while the solve was active and released
the allocation cleanly at exit. This supports GPU-memory fit on a nominal
48-GB device, but it does not certify that an RTX host has the required host
RAM or spare capacity. Keep the adaptive PDE fallback off the active RTX
Lumerical GPU and use a separately verified-idle, memory-qualified device.
The adjacent-pair metrics were:

- 100/50 nm: current 1.719%, field NRMSE 0.198%, mean 0.041%, peak 0.801%;
- 50/25 nm: current 1.014%, field NRMSE 0.0653%, mean 0.0214%, peak 0.295%;
- 25/12.5 nm: current 0.5314%, field NRMSE 0.0230%, mean 0.0102%, peak 0.0991%.

The adaptive gate correctly exhausts 12.5 nm and still fails this synthetic
case because current remains just above 0.5%. This is a negative-control
runtime/behavior check, not an Ea/Eb physical result and not evidence that the
final design is nonconverged. At that adaptive-PDE/symmetry-break commit, the
complete solver-free regression was `279 passed in 257.30 s`; the later
terminal-certificate count is recorded below.

If the exact forward/raw artifacts already exist, run no new Maxwell solve:

```bash
CUDA_VISIBLE_DEVICES=<verified-free-physical-index> \
photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/42_evaluate_lumerical_4um_exact_binary.py \
  --binary-mask-npz /absolute/final_exact_binary_cell_mask.npz \
  --binary-mask-key binary_mask \
  --output-dir /absolute/new_pde_convergence_output \
  --gpu-index <same-physical-index> --accelerator-policy development \
  --ea-forward-result /absolute/Ea_exact_forward.json \
  --ea-raw-npz /absolute/Ea_exact_forward_raw.npz \
  --eb-forward-result /absolute/Eb_exact_forward.json \
  --eb-raw-npz /absolute/Eb_exact_forward_raw.npz \
  --mesh-label fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps \
  --flake-dxy-nm 50 --outer-dxy-nm 200 \
  --stack-dz-nm 2.5 --bulk-dz-nm 50
```

This command runs the adaptive custom-PDE sequence for the particular raw Q
supplied; it stops at the first passing adjacent pair through 12.5 nm. Optical
x-y convergence still requires the separate 100/50-nm Lumerical comparison
described above. If the physical final-mask 25/12.5-nm pair also fails, do not
weaken 0.5%; add 6.25 nm only on a separately verified memory-qualified B200
rather than burdening the active RTX Lumerical production GPU.


## Post-launch uniform-symmetry optimizer repair -- 2026-08-25

Commit `ac077e4c` adds an exact linearized two-utility box max-min warm start
before the first beta-1 MMA call and sets stage `ftol_rel=xtol_rel=0` so the
expensive bounded stage ends by its explicit evaluation budget rather than a
near-symmetric numerical floor. It does not alter the signed epigraph,
DFM/grayness caps, move limit, or total stage physics-evaluation budget.

The documented external process is immutable commit `69b2bb40`; therefore it
cannot contain this later repair. The external tmux/manifest is not mounted on
this host, so its actual current state is unknown. Read that manifest before
acting. If it is stopped or shows no meaningful stage progress, launch a new
immutable latest-commit run with a new output root. Do not overwrite or resume
the old checkpoint with different code, and do not launch a duplicate while
the old process is still alive.


## Continuation wall-time budget

Commit `fc642e67` exposes the stage accounting in the serialized continuation
contract: 54 complete dual-polarization physics evaluations when every beta
stage passes on its first attempt, and 188 if every allowed retry is exhausted.
Each physics evaluation contains two Lumerical forwards, two Lumerical
adjoints, and the two downstream custom-CUDA chains.

At the measured evaluation-0 wall time of `1089.50 s`, those counts correspond
to 16.3425 h and 56.8961 h respectively. This is transparent budget
arithmetic, not an ETA guarantee; nonuniform geometry and later projection
stages can change solve time. The single evaluation is below 30 minutes, but
the complete continuation is an overnight-to-multiday production job. The
initial max-min warm start preserves the beta-1 evaluation budget rather than
adding another solve.


## Terminal lateral-certificate repair -- 2026-08-26

Commit `e284ea08` closes a mismatch between the written handoff and executable
success condition. Before this commit, script 41 could report a passed
exact-binary optimization after only one optical Maxwell mesh plus custom-PDE
convergence; the documented 100-to-50-nm Maxwell comparison was not connected
to `manifest["passed"]`.

The latest code now:

- requires passed Ea/Eb 100-nm and 50-nm source-only calibrations before the
  first continuation solve, all on the same GPU UUID, solver build, policy,
  CV0/2.5/50-nm z contract, 20-um span, 1-ps time, and PML 8;
- runs four fresh ordinary-dispersive exact-Au forwards per terminal candidate
  (100/50 nm times Ea/Eb);
- requires the 0.5% source-normalized Q, flux, complex-E, and E2 gates for both
  polarizations;
- maps both 50-nm and 100-nm optical raw Q through adaptive
  100/50/25/12.5-nm custom-PDE convergence and requires same-grid current and
  temperature changes below 0.5%;
- requires the fine-reference signs `I_Ea>0`, `I_Eb<0`;
- stores every failed exact candidate below its own beta-128 attempt and
  continues the configured retry policy instead of aborting immediately;
- prevents runtime-setup errors from writing a stray manifest inside Git;
- atomically freezes the final mask and can recover a terminal checkpoint only
  after rechecking the passed manifest's mask/state SHA and strict signs.

The full solver-free regression is `291 passed in 259.53 s`. This is code-path
validation, not a physical final-mask result.

The documented external process remains immutable `69b2bb40` and therefore
does not contain `ac077e4c`, `e284ea08`, or `628a88da`. Even if its old
manifest says `passed=true`, do not treat that as the terminal lateral
certificate. First retrieve its exact mask and run the latest script 43 with
fresh matching 100/50-nm Ea/Eb source-only calibrations. If the old
continuation is stopped or made no meaningful progress, launch a new
immutable latest-commit run only after setting:

- `AU_LUMERICAL_EA_FINAL_XY50_SOURCE_CALIBRATION`;
- `AU_LUMERICAL_EB_FINAL_XY50_SOURCE_CALIBRATION`.

The 54/188 continuation evaluation counts do not include script 43. Every
terminal candidate reaching it adds four fresh Maxwell forwards plus the
sequential fine-Q and coarse-Q adaptive PDE sequences, and up to eight
beta-128 candidates are allowed. Peak memory is not concurrent, but PDE wall
times add. No measured certificate wall time exists yet.


## Exact-binary electrical endpoint correction -- 2026-08-26

Commits `35412cb8` and `8db688b7` remove the continuous-path Au
sheet/contact conductivity floors from exact empty/full controls and final
binary promotion. Void Au nodes
and edges are omitted from the electrical reduced system; every executed final
PDE resolution must attest that the inactive-node count equals the zero-cell
count. Continuous optimization keeps its regularization and adjoint unchanged.
Terminal certificates now use schema v3, and continuation recovery rejects
v2 or a certificate from a different Git commit. The full current-HEAD
GPU-hidden regression is `292 passed in 260.03 s`.

This is a code-path correction, not a physical final result. No final-mask
Ea/Eb 100/50-nm Maxwell or downstream convergence artifact is available on
this host. The external `69b2bb40` process predates this commit and the later
terminal-certificate commits, so any retrieved mask must be certified anew by
latest script 43. No GPU was launched while making or testing this correction.
