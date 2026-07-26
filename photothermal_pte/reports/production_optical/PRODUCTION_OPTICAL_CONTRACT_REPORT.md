# Production optical contract regression

**PASS.** Production entrypoints reproduce the validated 3–6 µm optical contract. HEAT, optimization, and adjoint solves were not run.

## Realized FSP contract

- Source: 3–6 µm, broadband pulse.
- Power/field/index/Pabs/FoM analysis: one effective frequency point at 4 µm.
- TaIrTe4 sampled material: 600 points over 2.7–13.2 µm.
- Mesh: auto non-uniform, conformal variant 1, accuracy 5; `global_uniform_mesh` count 0.
- TaIrTe4 z mesh: 5 nm; Pabs analysis padding: 50 nm on each side.
- Solver: v261 / 8.35.4413, GPU; realized dt = `1.5888123912757566e-17 s`.

## Fresh production-entrypoint results

| Case | A_Qx | A_Qy | A_Qz | A_Q | A_local | A_six-face | closure | baseline closure | TMM error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_x | 0.257592496 | 0.000000000 | 0.000000000 | 0.257592496 | 0.257584434 | 0.257584434 | 0.003130% | 0.003011% | 0.335564% |
| flat_y | 0.000000000 | 0.422757308 | 0.000000000 | 0.422757308 | 0.422755539 | 0.422755539 | 0.000418% | 0.000418% | 0.115412% |
| flat_45 | 0.128794827 | 0.211378655 | 0.000000000 | 0.340173482 | 0.340169019 | 0.340169019 | 0.001312% | 0.001312% | 0.054570% |
| disk_x | 0.440196743 | 0.026212537 | 0.000000000 | 0.466409280 | 0.466123967 | 0.466123967 | 0.061209% | 0.061206% | — |

All closure and flat-stack TMM gates are below 0.5%. All four Pabs x/y/z mesh coordinate arrays equal the selected bandwidth-sweep arrays, dt is identical, the 4 µm source normalization is identical, and fitted-epsilon relative differences are at numerical roundoff.

## Independent audit of the proposed patch

The proposed patch was not applied verbatim. Its material sampling changed the wavelength array to micrometers but still passed it to `eps_flake()`, whose contract is nanometers. That would silently clamp/interpolate the wrong part of the table. The production implementation instead samples 2700–13200 nm, then converts only the frequency column to SI. The audit also found and removed the exporter source-range overwrite and the latest launcher G PVA/regional settings.

`eqc_lib.assert_production_contract()` reads the realized FSP before every solve and fails on source start/stop and pulse type, material range/sample count, monitor effective wavelength/count, mesh type/refinement/accuracy, global mesh count, flake dz, v261 build, CPU/GPU resources, requested GPU ID, and dt. A stale restart FSP is rejected instead of silently reused.

## Configuration-dependent failures

- v261 2.67–8 µm disk-x: reproduces the solver divergence marker; all post-divergence optical values are invalid.
- Shared Lumerical Resource Manager: concurrent sessions using one config home can overwrite the GPU ID. The final runs were sequential. Disk-x was launched after a GPU-0 pre-run assertion, and its solver log records `CUDA_VISIBLE_DEVICES=0`, `Detected GPU 0`, and successful completion. A later observed GPU-2 setting belonged to shared post-run state, not the completed engine.
- During regression development, a 1 nm-per-side Pabs volume under-integrated flat-x by 3.126%. Restoring the validated 50 nm nonabsorbing padding reproduced the exact baseline mesh and closure. No gain, clipping, or rescaling was used.

## Deliberately unchanged / not run

Geometry, PBC/PML, source and monitor positions, tensor axes, Pabs formula, Qy, normalization, HEAT code, optimizer, and mapping logic were unchanged. HEAT, the optimizer, and all adjoint/gradient solves were not run. The next permitted gate is forward–adjoint agreement plus directional finite differences under this CV1 production contract.
