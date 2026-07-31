# W12 50-nm Maxwell vs analytic explicit-3D thermal sanity

Status: `COMPLETED_W12_50NM_MAXWELL_ANALYTIC_EXPLICIT3D_THERMAL_SANITY`

This is the primary **existing inverse-design explicit-3D thermal FVM
sanity comparison**. It is not a paper reproduction. The separately
checkpointed thickness-integrated paper-reduced sheet calculation is an
optional control only and is not used by any result below.

## Fixed contracts

- completed 50-nm straight-edge a/b optical artifacts; no new 25-nm or
  12.5-nm optical refinement
- full Maxwell `Qx+Qy+Qz` retained on the 3D source grid
- same 285-µW incident power; no polarization matching or Q rescaling
- exact-overlap plus nearest-TaIrTe4-support conservative 3D remap used by
  the existing edge-a downstream audit
- one common explicit-3D operator: 60-µm lateral domain, 20-µm Si depth,
  100-nm core x/y, 10-nm TaIrTe4 dz, 285-nm SiO2, 600-nm air
- `kTaIrTe4=(3.8,14.4,1.0) W/(m K)` in lab `(x=b,y=a,z=c)`;
  `kSiO2=1.38`, `kSi=145`, `kair=0.026 W/(m K)`
- `G(TaIrTe4/air)=1`, `G(TaIrTe4/SiO2)=7.37e6`,
  `G(SiO2/Si)=1.1e9 W/(m² K)`
- far x/y and bottom fixed `DeltaT=0`; exposed-surface `h=10 W/(m² K)`
- no PTE, weighting potential, adjoint, AD-FD, or optimization

The analytic source is the volumetric Gaussian--Beer--Lambert law integrated
exactly over every target cell. It is not collapsed to a 130-nm sheet.

## Primary same-incident-power results

| case | Pabs (W) | Tmax (K) | TaIrTe4 mean (K) | max \|dT/dn\| (K/m) | max \|grad T\| (K/m) |
|---|---:|---:|---:|---:|---:|
| Maxwell a | 2.836587092e-05 | 1.344649423e-01 | 7.131426730e-03 | 1.039612712e+04 | 1.583967968e+04 |
| Maxwell b | 3.735675195e-05 | 1.356193572e-01 | 9.388746976e-03 | 9.502313182e+03 | 1.487210513e+04 |
| analytic a | 2.524570212e-05 | 9.559856274e-02 | 6.441342143e-03 | 5.891000140e+03 | 1.032092469e+04 |
| analytic b | 3.760968229e-05 | 1.383321822e-01 | 9.473877959e-03 | 8.674267737e+03 | 1.515947328e+04 |

| model | Pabs b/a | Tmax b/a | mean T b/a | max \|dT/dn\| b/a | max \|grad T\| b/a |
|---|---:|---:|---:|---:|---:|
| Maxwell | 1.316961 | 1.008585 | 1.316531 | 0.914024 | 0.938915 |
| analytic | 1.489746 | 1.447011 | 1.470793 | 1.472461 | 1.468810 |

The analytic volumetric source reproduces the paper-like `b>a` temperature
and gradient ordering. The Maxwell source gives `b>a` absorbed power, Tmax,
and mean temperature, but its local edge-gradient maxima remain `b<a`.
Because every case used one identical explicit-3D operator, this remaining
reversal is attributed to the Maxwell spatial/depth source distribution
within this named model, not to changing the thermal boundary contract.

## Maxwell--analytic differences

| polarization | P ratio M/A | volumetric-Q NRMSE | 3D T NRMSE | flake 3D T NRMSE | gradient-vector NRMSE |
|---|---:|---:|---:|---:|---:|
| a | 1.123592 | 34.442852% | 20.550570% | 21.625773% | 73.779257% |
| b | 0.993275 | 11.029048% | 3.623110% | 4.243271% | 28.659637% |

Equal-absorbed-power comparisons are stored as separately named linearity
diagnostics. They do not modify either primary source.

All remap errors, residuals, energy balances, source moments, boundary powers,
surface/midplane maps, all five in-plane derivative fields, depth profiles,
and edge-normal profiles are in the JSON/NPZ/figures.

## Provenance

- generation commit: `23bdc05f005a86dd15691da8934cb1f1bffad210`
- generation command: `/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/validation/paper_ir_sanity/compare_w12_50nm_maxwell_analytic_explicit3d.py --edge-a-dir /data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_edge45_a_L60_nested_xy50_h22_dz5_pml24_t4_gpu5_20260731 --edge-b-dir /data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_edge45_b_L60_nested_xy50_h22_dz5_pml24_t4_gpu5_retry1_20260731/readonly_recovery_b_reference_retry1 --incident-reference-npz /home/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_planar_empty_reference_a_L60_xy100_dz5_pml24_t4_gpu4_retry4_20260731/readonly_recovery_v2/incident_reference.npz --output-dir /home/seunghyun/tairte4_artifacts/paper_ir_w12_explicit3d_thermal50_20260731 --report-dir /home/seunghyun/tairte4/pte_inverse_design_adfd/photothermal_pte/reports/paper_ir_w12_explicit3d_thermal50 --thermal-step-nm 50`
- raw NPZ/FSP files remain external and are path/size/SHA-256 inventoried
