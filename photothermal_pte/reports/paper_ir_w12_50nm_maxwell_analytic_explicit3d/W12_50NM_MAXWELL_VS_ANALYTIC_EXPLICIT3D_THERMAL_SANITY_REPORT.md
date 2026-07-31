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
| Maxwell a | 2.836587092e-05 | 1.333181015e-01 | 7.131823191e-03 | 1.499970587e+04 | 1.859566976e+04 |
| Maxwell b | 3.735675195e-05 | 1.355304372e-01 | 9.388405424e-03 | 1.319392977e+04 | 1.689368223e+04 |
| analytic a | 2.530481291e-05 | 9.569412297e-02 | 6.455165654e-03 | 6.322082941e+03 | 1.056678943e+04 |
| analytic b | 3.769774234e-05 | 1.384742657e-01 | 9.494234814e-03 | 9.325735209e+03 | 1.553112831e+04 |

| model | Pabs b/a | Tmax b/a | mean T b/a | max \|dT/dn\| b/a | max \|grad T\| b/a |
|---|---:|---:|---:|---:|---:|
| Maxwell | 1.316961 | 1.016594 | 1.316410 | 0.879613 | 0.908474 |
| analytic | 1.489746 | 1.447051 | 1.470796 | 1.475105 | 1.469806 |

The analytic volumetric source reproduces the paper-like `b>a` temperature
and gradient ordering. The Maxwell source gives `b>a` absorbed power, Tmax,
and mean temperature, but its local edge-gradient maxima remain `b<a`.
Because every case used one identical explicit-3D operator, this remaining
reversal is attributed to the Maxwell spatial/depth source distribution
within this named model, not to changing the thermal boundary contract.

## Maxwell--analytic differences

| polarization | P ratio M/A | volumetric-Q NRMSE | 3D T NRMSE | flake 3D T NRMSE | gradient-vector NRMSE |
|---|---:|---:|---:|---:|---:|
| a | 1.120967 | 34.382169% | 19.877187% | 21.022594% | 72.921777% |
| b | 0.990955 | 11.693254% | 4.032584% | 4.659923% | 29.010282% |

Equal-absorbed-power comparisons are stored as separately named linearity
diagnostics. They do not modify either primary source.

All remap errors, residuals, energy balances, source moments, boundary powers,
surface/midplane maps, all five in-plane derivative fields, depth profiles,
and edge-normal profiles are in the JSON/NPZ/figures.

## Provenance

- generation commit: `42245b9848f725b3b29a2ecb71ecd9d63535058d`
- generation command: `/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/validation/paper_ir_sanity/compare_w12_50nm_maxwell_analytic_explicit3d.py --edge-a-dir /data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_edge45_a_L60_nested_xy50_h22_dz5_pml24_t4_gpu5_20260731 --edge-b-dir /data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_edge45_b_L60_nested_xy50_h22_dz5_pml24_t4_gpu5_retry1_20260731/readonly_recovery_b_reference_retry1 --incident-reference-npz /home/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_planar_empty_reference_a_L60_xy100_dz5_pml24_t4_gpu4_retry4_20260731/readonly_recovery_v2/incident_reference.npz --output-dir /data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_50nm_maxwell_analytic_explicit3d_20260731 --report-dir /home/seunghyun/tairte4/pte_inverse_design_adfd/photothermal_pte/reports/paper_ir_w12_50nm_maxwell_analytic_explicit3d`
- raw NPZ/FSP files remain external and are path/size/SHA-256 inventoried
