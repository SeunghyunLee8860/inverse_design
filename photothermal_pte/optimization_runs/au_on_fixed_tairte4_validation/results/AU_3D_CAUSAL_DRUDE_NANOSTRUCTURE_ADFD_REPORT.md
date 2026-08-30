# 3-D causal-Drude Au nanostructure AD-FD control

Status: `VALIDATED_3D_CAUSAL_DRUDE_AU_ADFD_CONTROL`

## What this validates

This checkpoint resolves the optical-gradient blocker for a **fixed-grid Au
nanoantenna density**, not for a v261 moving/conformal metal boundary.  The
design variable is a 2-D physical density extruded through a fixed Au
thickness.  It scales a passive Drude pole as `s(rho)=rho^3`; air and Au are
the exact endpoints.  The complete 3-D time-domain Maxwell trajectory and
Au absorption are differentiated with checkpointed reverse-mode AD on GPU.

This is an algorithmic control in air.  It is not yet the coupled TaIrTe4,
thermal, electrical, PTE, or production optimization result.

## Material and discretization

- wavelength: `1.00000000e-05 m`
- frozen Au endpoint: `n=12.1`, `k=69.2`
- target epsilon: `-4642.23000000 + 1674.64000000i`
- Drude omega_p: `1.364472551689e+16 rad/s`
- Drude gamma: `6.793629134628e+13 rad/s`
- endpoint fit relative error: `0.000e+00`
- grid: `[40, 40, 40]`, resolution `1.000e-07 m`
- six PML boundaries: `8` cells per face
- realized design cells: `[10, 10, 2]`
- Au time-resolution check: `omega_p*dt=0.656937`
- total simulation: `18` optical periods, `12471` steps

The initial Courant `0.95` debug run was rejected because the explicit Au ADE
became non-finite (`omega_p*dt` was about 2.5).  The promoted run uses Courant
`0.25` and remains finite.  This is a physical
time-resolution correction, not gradient fitting or rescaling.

## Gates

- Au absorbed power: `1.746995375000e-18 W` under the control-source normalization
- previous-to-late phasor-window change: `0.000293%`
- gradient L2 norm: `2.842981944411e-18 W/rho`
- maximum strong-direction error at `h=0.005`: `0.014047%`
- maximum multi-direction gradient-normalized error: `0.002736%`
- near-null direction retained as diagnostic: `central_localized`
- finite arrays: `True`
- GPU-only: `True`

| direction | strong | AD (W) | central FD (W) | relative error | error / gradient L2 |
|---|---:|---:|---:|---:|---:|
| uniform | True | 9.913251e-19 | 9.912875e-19 | 0.003790% | 0.001321% |
| smooth_asymmetric | True | -5.538778e-19 | -5.538000e-19 | 0.014047% | 0.002736% |
| central_localized | False | 1.866967e-21 | 1.925000e-21 | 3.014708% | 0.002041% |
| design_edge_localized | True | 1.192291e-18 | 1.192300e-18 | 0.000792% | 0.000332% |
| fixed_seed_random | True | 6.661976e-19 | 6.661500e-19 | 0.007149% | 0.001675% |

The central-localized direction is near-null, so its relative error with the
tiny FD denominator is not used as a strong-direction gate.  It is retained
and passes the global gradient-normalized gate.  No direction is deleted.

## Reproduction

```bash
env PYTHONPATH=/home/seunghyun/.local/au_fdtdx \
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/39_validate_3d_drude_nanostructure_adfd.py \
  --output-dir photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/results

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/40_summarize_3d_drude_nanostructure_adfd.py
```

The external FDTDX/JAX environment is not vendored.  Its pinned source commit
and targeted upstream tests are recorded in the manifest.  Raw compiler and
runtime caches are not committed.

## Next fail-closed gate

Add a fixed anisotropic TaIrTe4 layer and independently account for `Q_Au`
and `Q_TaIrTe4`; then cross-check exact-binary endpoints against Lumerical.
Thermal/PTE coupling and Au topology optimization remain blocked until that
combined optical checkpoint passes.
