# Production execution-path audit

This audit was performed before applying the production optical contract. The proposed patch was treated as review input and was not applied verbatim.

## Active server execution chain

1. `inverse_design/launch_G_constrained_500nm_20260724.sh` is the latest production shell launcher.
2. It calls `inverse_design/run_constrained_inverse_design.py`.
3. The runner constructs `VolumeCurrentEvaluator` for x/y channels; `prepare()` calls `eqc_lib.build_control_base()`.
4. `eqc_lib.bootstrap_env()` is the single hard-pinned source/material/mesh contract, and `assert_production_contract()` reads the realized FSP before every solve.
5. Forward and any future adjoint call both pass through `eqc_lib.run_project()`, so they share the same pre-solve assertion. No adjoint or optimization was run in this change.

The GitHub repository is intentionally the photothermal/PTE-only export. It includes the shared production core (`photothermal_pte/eqc_lib.py`, model, and Lumerical helper), the forward exporter, regression runner, and compact evidence. Optimizer/mapping implementation files remain outside this export.

## Override audit

| Concern | Finding and disposition |
|---|---|
| Source creation | `bundle/tairte4_volume_model.py` creates one plane source with 3–6 µm start/stop. |
| Runtime/export overwrite | `eqc_lib.py` and `02_export_fdtd_qon.py` now set 3–6 µm; the previous exporter 4/4 µm overwrite is removed. |
| `TARGET_WL_UM` aliasing | It controls only the one-point 4 µm target. Separate `SOURCE_WL_START_UM`, `SOURCE_WL_STOP_UM`, `MATERIAL_FIT_START_UM`, and `MATERIAL_FIT_STOP_UM` control the other ranges. |
| Material table | 600 samples are generated at 2700–13200 nm because `eps_flake()` accepts nm. The proposed patch incorrectly passed µm into that helper. |
| PVA enforcement | Production `eqc_lib.bootstrap_env()` hard-pins conformal variant 1. Remaining PVA strings in the full server tree belong to historical AD/FD and TMM diagnostics, not the production optical execution chain. |
| Global uniform mesh | The simulator helper has an explicit creation flag; production passes false. The model, solver pin, and realized-FSP assertion each fail if `global_uniform_mesh` exists. |
| Restart/stale FSP | An existing control FSP is read and fully asserted before reuse. Old 4/4 µm, PVA, uniform, wrong material range, monitor, version, resource, or dt settings stop before a solve. |
| Monitor sampling | Power/field/index/Pabs/FoM monitors have one effective custom or uniform sample at 4 µm. Inactive raw properties are recorded separately and are not mistaken for active sampling. |
| Latest launcher | The server launcher G was corrected from PVA/regional/accuracy2/2 ps labels to CV1/auto/accuracy5/4 ps and explicit source/material ranges. The core runtime assertion remains authoritative. |

## Requested string-search disposition

- `precise volume average`: historical gradient/TMM scripts and historical documents only; no active production runtime path.
- `global_uniform_mesh`: guarded legacy constructor branch plus absence assertions; never created in production.
- `BULK_MESH_MODE`: hard-pinned to `auto` by the production bootstrap and current entrypoints.
- `MSOPT_MESH_REFINEMENT_PIN`: hard-pinned to `conformal variant 1`.
- `wavelength start` / `wavelength stop`: source helper, production global-source pin, realized-FSP assertion, and validation diagnostics.
- `bandwidth`: helper API and validation report naming; the active plane source realizes Lumerical `pulse type=broadband` for 3–6 µm.
- `TARGET_WL_UM`: one scalar 4 µm analysis target only.
- `mesh refinement` / `mesh accuracy`: CV1/5 production pin plus realized-FSP assertions; historical diagnostics are explicitly non-production.

## Unchanged scope

Geometry, PBC/PML, source and monitor positions, TaIrTe4 tensor axes, Pabs formula, Qy, normalization, HEAT code, optimizer logic, and mapping logic were not changed. No gain, clipping, or Q rescaling is present.
