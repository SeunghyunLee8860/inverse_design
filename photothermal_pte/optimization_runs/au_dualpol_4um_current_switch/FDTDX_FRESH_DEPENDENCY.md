# Fresh FDTDX dependency and mesh bridge

Status: **PINNED_SOURCE_VALIDATED_RUNTIME_NOT_YET_CERTIFIED**

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

## Remaining runtime blocker

No user-owned FDTDX/JAX/CUDA environment existed at the start of this audit.
The source checkout is validated, but no package lock, GPU device identity,
minimal placement, or source-only field solve has yet been certified.

Do not run a reference sweep until a fresh runtime preflight proves:

1. exact Python/JAX/JAXLIB/CUDA/cuDNN/FDTDX package identities;
2. FDTDX imports only from the locked source path;
3. one explicitly selected idle GPU is visible;
4. a tiny GPU JAX calculation succeeds without using the Lumerical GPU;
5. the anchor model places exact material/source/monitor bounds with explicit
   PML readback;
6. raw results use a portable, explicitly configured directory.
