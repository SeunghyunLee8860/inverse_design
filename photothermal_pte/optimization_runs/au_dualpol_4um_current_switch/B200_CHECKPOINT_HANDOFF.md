# B200 checkpoint handoff

## What moves to the new Codex session

The live chat is not the execution state.  A new Codex session on the B200
host can continue the inverse design from this Git checkout because the
portable state is committed in `b200_migration/`:

- `continuation_checkpoint.npz`: beta-1 logical-attempt-4 optimizer state;
- `terminal_stage_state.npz`: independently hashed attempt-3 terminal state;
- `restart_manifest.json`: relative-path, fail-closed restart provenance;
- `bundle_manifest.json`: sizes, SHA-256 values, currents, FOM, and exclusions.

The checkpoint and terminal latent arrays are bitwise identical.  The last
completed feasible point has `I_Ea=+3.975912039 nA`,
`I_Eb=-3.989552404 nA`, balanced FOM `3.975912039 nA`, and grayness
`0.661051131`.  It is beta 1 and is not binary or fabrication-certified.

The partial attempt-4 evaluation 7 is deliberately excluded.  That evaluation
finished both forwards but failed the layout-only component-Yee Jacobian
self-audit before an adjoint result could be accepted.  The original failure
was: global-uniform mapping FD error `0.1653%`, fixed-random `0.0612%`,
upper-endpoint `0.00310%`, and a weak central-localized direction `1.197%`
against the internal `0.05%` limit.  No tolerance has been loosened.  The B200
run restarts from the last completed attempt-3 state and recomputes attempt 4.

## Physics that must not change

- Maxwell: Ansys Lumerical FDTD only; no FDTDX Maxwell solve.
- Solver build: v261, 2026 R1.2 build 4522.  Using R1.3 is a numerical-model
  change and requires a new derivative certificate rather than this direct
  continuation.
- Thermal/electrical: repository custom CUDA PDEs; no Lumerical HEAT or CHARGE.
- Objective: maximize `min(I_Ea, -I_Eb)` with `Ea = E || a` and `Eb = E || b`.
- Coordinate mapping: Lumerical x = crystal b and y = crystal a.
- Au thermopower: `S_Au=+1.94 uV/K` remains in the forward and adjoint paths.
- Optimization mesh: 100-nm lateral and 2.5/50-nm stack/bulk z with CV0.
- Final exact-binary reevaluation: fresh 100-nm and 50-nm optical meshes plus
  adaptive custom-PDE convergence.
- Fabrication continuation: 250-nm solid, 250-nm void/spacing, then grayness;
  beta schedule remains 1, 2, 4, 8, 16, 32, 64, 128.

The mesh and combined Ea/Eb AD-FD evidence do not need to be recomputed merely
because the CUDA device changes.  The source-only calibration records do need
to be regenerated because the production loader binds them to physical GPU
UUID, accelerator policy, solver build, polarization, and mesh.

## B200 setup and launch

Clone and select the handoff branch:

```bash
git clone https://github.com/SeunghyunLee8860/inverse_design.git
cd inverse_design
git switch agent/optimize-au-dualpol-4um-pte
git pull --ff-only
```

On the B200 host, first discover and export these site-specific absolute paths:

```bash
export AU_LUMERICAL_PYTHON=/absolute/path/to/EIDL-Lumapi/bin/python
export AU_LUMERICAL_ROOT=/absolute/path/to/v261-R1.2-build-4522
export AU_RUNRES_BIN=/absolute/path/to/runres
export LUM_RESERVE_MODULE_DIR=/absolute/path/containing/lum_reserve.py
export LUMERICAL_B200_GPU_INDEX=<verified-idle-physical-B200-index>
export FDTD_THREADS=8
```

Do not guess these paths from the old server.  Confirm the GPU is idle, the
Lumerical VERSION file says R1.2 build 4522, `runres` can see `run`, and the
reservation module is readable.

Generate the four small, GPU-bound source-only calibrations (Ea/Eb at xy100
and xy50):

```bash
handoff_dir=photothermal_pte/optimization_runs/au_dualpol_4um_current_switch
"$handoff_dir/prepare_lumerical_b200_source_calibrations.sh" \
  /absolute/B200/raw/au4um_source_calibrations
```

Then launch the checkpoint continuation in a persistent terminal.  The output
root must not already exist:

```bash
tmux new-session -s au4um_b200_checkpoint
handoff_dir=photothermal_pte/optimization_runs/au_dualpol_4um_current_switch
"$handoff_dir/launch_lumerical_b200_checkpoint_continuation.sh" \
  /absolute/B200/raw/au4um_checkpoint_continuation \
  /absolute/B200/raw/au4um_source_calibrations
```

The launcher first runs a solver-free fail-closed preflight.  It verifies the
B200 GPU, exact Lumerical installation, all four source records, checkpoint,
terminal-state hash, and cross-commit restart provenance.  Only then does
`runres` reserve nine FDTD tasks and start the full continuation.

Do not run a second process against the same output root.  Do not copy
`~/.codex`, SSH keys, passwords, license files, the old RTX calibration JSONs,
or the 1.9-GB partial evaluation directory into Git.

## Prompt for the new B200 Codex session

> Checkout branch `agent/optimize-au-dualpol-4um-pte` and read
> `photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/B200_CHECKPOINT_HANDOFF.md`
> followed by `CODE_HANDOFF.md`. This is Lumerical-only Maxwell plus custom
> CUDA thermal/electrical PDE; never use FDTDX or Lumerical HEAT/CHARGE. Verify
> the committed `b200_migration` hashes, discover the B200 site's exact
> R1.2-build-4522 Python/Lumerical/runres paths, generate the four B200-bound
> source-only calibrations, run the fail-closed preflight, and resume the
> committed beta-1 attempt-4 checkpoint in tmux. Do not restart from uniform
> rho=0.5 and do not relax the component-Yee Jacobian gate silently. Report
> iteration, beta, FOM, Ia, Ib, grayness, active constraints, wall time, and
> any gate failure whenever I ask “지금은?”.
