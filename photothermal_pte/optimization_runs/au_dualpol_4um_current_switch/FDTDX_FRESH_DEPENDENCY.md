# Fresh FDTDX dependency and mesh bridge

Status: **GPU_RUNTIME_VALIDATED_PLACEMENT_NOT_YET_CERTIFIED**

The fresh exact-binary route pins the official FDTDX repository at:

```text
repository  https://github.com/ymahlau/fdtdx.git
commit      f26f84b70a8cceec9b889553955a868624736bf1
tree        43687e561d4bd2735f188149b2fc1bc50da82c47
describe    v0.6.2-70-gf26f84b
```

This commit was re-fetched from official `origin/main` on 2026-08-24. The
complete commit/tree, clean worktree, remote URL, and SHA-256 of the update,
dispersion, PML, material-object, detector, and package files are checked by
`fdtdx_dependency.py` against `fdtdx_dependency_lock.json`.

The persistent source checkout used on this host is:

```text
/home/seunghyun200/dependencies/fdtdx-f26f84b70a8cceec9b889553955a868624736bf1
```

Validate it without importing FDTDX:

```bash
python3 -m photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_dependency \
  --source /home/seunghyun200/dependencies/fdtdx-f26f84b70a8cceec9b889553955a868624736bf1
```

The current source audit passes every check. The checkout is intentionally
outside this repository; generated environments and upstream source must not
be committed to the inverse-design branch.

## Fresh mesh bridge

`fdtdx_fresh_mesh.py` converts a solver-independent `MeshSpec` into exact
NumPy edge arrays and a matching FDTDX `GridLayout` inside a temporary context.
It restores both globals even when placement fails. The bridge supports local
Au-window x/y refinement while preserving the 80 x 80 physical binary mask.

`fdtdx_4um_model.build_model()` now accepts an optional
`pml_face_parameters` argument. Existing historical callers omit it and retain
their historical behavior. The fresh bridge always supplies it.

`fdtdx_fresh_pml.py`:

- requires all six faces;
- requires all nine alpha/kappa/sigma solver values per face;
- calculates alpha from the 4 um carrier;
- calculates sigma using each face's physical PML thickness;
- strips audit metadata before constructing `BoundaryConfig`;
- rejects missing, non-finite, or non-lossy profiles.

This prevents the fresh campaign from inheriting FDTDX's locked-source
`alpha_start` default based on 1.55 um.

## Runtime preflight

The user-owned environment is
`/home/seunghyun200/.venvs/fdtdx-fresh-py312`. The exact Python, JAX, CUDA
plugin/library, FDTDX, NumPy, SciPy, and solver dependency versions are pinned
in `fdtdx_runtime_lock.json`. On 2026-08-24, a one-device JAX calculation
passed on an otherwise idle B200 without memory preallocation.

Every fresh GPU command must go through `run_fdtdx_fresh_gpu.sh`. The wrapper:

1. requires an explicitly selected physical GPU and a new explicit raw-output
   directory;
2. audits the complete locked source checkout before importing it;
3. rejects a GPU with another compute process, including a Lumerical engine;
4. rejects package drift and an FDTDX import from site-packages;
5. exposes exactly one GPU and disables JAX memory preallocation;
6. runs a small GPU calculation before executing the requested Python entry
   point.

Example (only after rechecking that the selected GPU is idle):

```bash
mkdir -p /home/seunghyun200/fdtdx_results/fresh_placement_001
FDTDX_FRESH_GPU_INDEX=7 \
FDTDX_FRESH_OUTPUT_DIR=/home/seunghyun200/fdtdx_results/fresh_placement_001 \
  ./run_fdtdx_fresh_gpu.sh -m your.module
```

The remaining blockers are the anchor-model placement/readback audit and a
source-only field solve. No reference sweep or optimizer is authorized by the
runtime smoke certificate alone.
