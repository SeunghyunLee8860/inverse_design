# PTE inverse-design optimization runs

This directory is the repository-side home for **future optimization jobs**.
Each optimization receives one immutable `run_NNN_slug/` directory.  It keeps
the run configuration, source commit, SHA-pinned inputs, checkpoints, compact
results, plots, and a raw-artifact manifest together.

The validated AD--FD certificate is the upstream dependency; it is not itself
an optimization result.  The first run below is therefore `PLANNED`, and no
Maxwell, thermal, adjoint, or optimizer solve was launched while creating this
layout.

## Current contract

- finite, nonperiodic 2 um x 2 um design ROI;
- 81 x 81 nodal latent density at 25 nm spacing;
- 500 nm nonperiodic conic filter and tanh projection;
- six optical PML boundaries and no periodic/Bloch wrapping;
- component-specific Yee material Jacobian and conservative optical-to-thermal
  power remap;
- anisotropic TaIrTe4 thermal conductivity and explicit named interfaces;
- uniform 45 degree PTE weighting surrogate used by the certified AD--FD path;
- raw FSP/NPZ/checkpoint binaries remain outside Git and are SHA-pinned.

The baseline uses gray-law exponent `p=1`.  `p=2` and `p=3` remain named
sensitivity scenarios, not confidence bounds.  A run must get a new ID when a
physical assumption, objective, illumination, domain, material law, or
optimizer setting changes.

## Create a run

From the repository root:

```bash
python -m photothermal_pte.optimization_runs.create_run \
  --slug example \
  --description "Short, explicit objective" \
  --source-commit "$(git rev-parse HEAD)"
```

Then edit only the newly created `run_config.json`, add it to `registry.json`,
and validate it:

```bash
python -m photothermal_pte.optimization_runs.validate_run \
  photothermal_pte/optimization_runs/run_NNN_example
```

Use `--require-external` immediately before a solver launch.  This makes the
preflight fail closed if a raw artifact is absent or has a different SHA-256.

## What belongs in Git

- `run_config.json`, `STATUS.json`, README and manifest JSON;
- compact per-iteration CSV/JSON summaries;
- final density arrays only when small and reviewable;
- plots and Markdown reports;
- checkpoint path, byte size, SHA-256, generation command and solver metadata.

Do not commit FSP files, raw field NPZ files, solver caches, temporary projects,
or large binary optimizer checkpoints.  Those stay in an external artifact
directory and are referenced from `manifests/RAW_ARTIFACT_MANIFEST.json`.

## Run lifecycle

`PLANNED -> PREFLIGHT_PASSED -> RUNNING -> COMPLETED`

Failures remain as `FAILED` or `BLOCKED`; do not erase or relabel them.  A
completed run is never overwritten.  A continuation or changed scenario gets
the next run number.

## Important current limitation

The repository has a validated end-to-end gradient certificate, but it did not
previously contain an iterative production optimizer.  This change adds the
run/provenance layer and a pinned baseline package; it does not silently turn
the expensive AD--FD validation script into an optimizer.  The first actual
optimization driver and update rule must be reviewed and committed separately
before any new optimization changes from `PLANNED`.
