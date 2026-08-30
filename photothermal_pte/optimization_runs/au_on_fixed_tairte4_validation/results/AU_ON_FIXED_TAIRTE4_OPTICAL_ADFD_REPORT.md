# Au nanostructure on fixed TaIrTe4: optical AD-FD control

Status: `VALIDATED_AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_CONTROL`

## Outcome

This checkpoint validates a differentiable **Au nanocube/nanoantenna design
material**, not an Au electrode.  A two-dimensional density is extruded
through a fixed Au thickness above a fixed TaIrTe4 slab.  The full 3-D causal
dispersive Maxwell trajectory is differentiated on GPU.  Au absorption,
TaIrTe4 absorption, and their sum are kept separate.

The production v261 moving/conformal-Au route remains blocked.  This result
instead establishes a working fixed-grid dispersive route whose total optical
gradient agrees with central finite differences.

## Materials and axes

- wavelength: `1.00000000e-05 m`
- Au endpoint: `n=12.1`, `k=69.2`
- Au epsilon: `-4642.23000000 + 1674.64000000i`
- TaIrTe4 epsilon_a: `-39.87819057 + 187.50005695i`
- TaIrTe4 epsilon_b: `13.77852773 + 23.68846319i`
- TaIrTe4 epsilon_c: `13.77852773 + 23.68846319i`
- solver axes: `x=b`, `y=a`, `z=c=b closure`
- permittivity table SHA-256: `d66eb034cb977be9ef843dd0972fcb7628ea28d168ab771f5c3a757bf5e0d499`

The TaIrTe4 `c=b` value is the repository's explicit 3-D closure, not a
directly measured independent c-axis response.  Each axis and Au use a
passive one-pole ADE fitted to the exact finite-time-step harmonic response at
10 um.  This is an exact single-frequency causal closure, not a measured
broadband pole fit.

The gray Au law is `pole strength = rho^3`.  It preserves exact air/Au
endpoints on a fixed Yee support.  It is a numerical topology relaxation and
is not called a physical gray effective medium.

## Geometry and numerics

- domain cells: `[40, 40, 40]` at `1.000e-07 m`
- six PML boundaries: `8` cells each
- Au design cells: `[10, 10, 2]`
- fixed TaIrTe4 cells: `[14, 14, 2]`
- direct optical Au/TaIrTe4 face contact: `True`
- optical periods: `18`; time steps: `12471`
- two independent phasor windows: `4` periods each
- gradient: `checkpointed JAX reverse-mode AD`

This is a small optical algorithmic control in air.  It does not yet include
SiO2/Si, thermal contact conductance, electrode collection, PTE current, or a
production-size nanoantenna optimization.

## Absorption and settling

- `P_Au = 1.049336500000e-18 W`
- `P_TaIrTe4 = 9.551596250000e-19 W`
- `P_total = 2.004496125000e-18 W`
- Au previous/late change: `0.000214%`
- TaIrTe4 previous/late change: `0.000393%`
- total previous/late change: `0.000299%`

The powers use the control source normalization and are not scaled to 285 uW.
No clipping, smoothing, gain, or post-hoc power rescaling is applied.

## AD-FD certificate

- total gradient L2: `1.507419273461e-18 W/rho`
- `g_total-(g_Au+g_TaIrTe4)` relative norm: `6.885e-07`
- maximum total strong-direction error at `h=0.005`: `0.014831%`
- maximum total multi-direction gradient-normalized error: `0.004100%`

| total-power direction | strong | AD (W) | central FD (W) | relative error | error / gradient L2 |
|---|---:|---:|---:|---:|---:|
| uniform | True | 5.431931e-19 | 5.432125e-19 | 0.003567% | 0.001285% |
| smooth_asymmetric | True | -2.719905e-19 | -2.720125e-19 | 0.008084% | 0.001459% |
| central_localized | False | 1.900699e-21 | 1.962500e-21 | 3.149113% | 0.004100% |
| design_edge_localized | True | 5.804590e-19 | 5.804000e-19 | 0.010158% | 0.003911% |
| fixed_seed_random | True | 3.194276e-19 | 3.194750e-19 | 0.014831% | 0.003143% |

Near-null directions are retained in the CSV and judged with the global
gradient-L2 normalization rather than a tiny directional denominator.

## Reproduction

```bash
env PYTHONPATH=/home/seunghyun/.local/au_fdtdx   CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false   /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python   photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/41_validate_au_on_fixed_tairte4_optical_adfd.py

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python   -m pytest -q photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/tests/test_au_on_tairte4_contract.py

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python   photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/42_summarize_au_on_fixed_tairte4_optical_adfd.py
```

## Next fail-closed gate

Cross-check exact-binary air/Au endpoints for the same fixed TaIrTe4 optical
stack against v261 Lumerical, including material readback, component powers,
PML/mesh convergence, and source normalization.  Thermal/PTE coupling and a
production Au topology optimization remain blocked until that endpoint check
is closed.
