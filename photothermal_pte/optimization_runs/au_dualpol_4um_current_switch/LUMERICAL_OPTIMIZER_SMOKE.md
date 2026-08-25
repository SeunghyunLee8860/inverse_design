# Lumerical 4-um dual-polarization optimizer smoke

## What this run does

The enabled entry point is a deliberately bounded development smoke, not a
production continuation. It uses the canonical 81x81 latent nodal topology,
the 500-nm conic filter, beta-4 tanh projection, the nonlinear Au
`n-k`-then-square carrier, Lumerical FDTD Maxwell forwards/adjoints, and the
repository custom CUDA thermal/electrical equations.

NLopt LD_MMA maximizes `t` subject to

- `t - I_Ea <= 0`,
- `t + I_Eb <= 0`,
- the smooth 500-nm solid opening residual cap, and
- the smooth 500-nm void opening residual cap.

The target current signs are `I_Ea > 0` and `I_Eb < 0`. The run is hard
limited to two distinct function evaluations. It does not use FDTDX Maxwell,
Lumerical HEAT, or Lumerical CHARGE.

## Validated development contract

- Lumerical 2026 R1.2 build 4522
- physical GPU 5, UUID `GPU-aa047452-9c73-d10f-675f-8af800915acf`
- MCM6 Au carrier
- conformal variant 0
- 2.5-nm thin-stack z mesh and 50-nm bulk/air/PML z mesh
- 100-nm flake/design in-plane mesh and 200-nm outer in-plane mesh
- 20-um lateral span, z = -3 to +3 um, eight PML layers, 1-ps window
- separately hash-bound Ea and Eb source-only calibration JSON files

The source calibrations are GPU-UUID-bound. Do not change the physical GPU
without supplying new matching Ea and Eb calibration records.

## Launcher

Set a brand-new raw output directory and run:

```bash
export LUMERICAL_GPU_INDEX=5
export FDTD_THREADS=8
export AU_LUMERICAL_OPT_OUTPUT_ROOT=/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_smoke/<new-run-name>
photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_lumerical_4um_dualpol_smoke_runres.sh
```

`runres` reserves nine `lum_fdtd_solve` tasks for the entire job and waits up
to six hours by default. Raw FSP/NPZ results stay outside Git. The run fails
closed on a nonempty output directory, source/mesh/GPU mismatch, any forward,
PDE, Jacobian, or adjoint gate failure, or anything other than exactly two
unique physics evaluations.

## Why it must stop after the smoke

At the time this launcher was added, the filesystem had only about 270 GB
free and was 96% full. The current audit-oriented forward and adjoint scripts
retain several large FSP files per evaluation, so a long run would consume
storage too quickly. Review the smoke result first. Before raising the
evaluation budget, implement hash-safe checkpoint/resume and bounded artifact
retention while preserving enough state for independent audit and binary Au
reevaluation.
