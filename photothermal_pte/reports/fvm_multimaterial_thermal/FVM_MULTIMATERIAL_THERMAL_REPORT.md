# Multi-material anisotropic finite-G FVM thermal report

## Status

`VALIDATED_MULTIMATERIAL_FVM_PRODUCTION_CONVERGENCE`

This is an independent conservative Cartesian Python/SciPy FVM result. It is
not a Lumerical HEAT result. The common scalar-isotropic/perfect-contact 3D
subset was separately cross-validated against v261 HEAT before this extended
model was used.

## Production reference

- Geometry: 2 um x 2 um x 100 nm TaIrTe4 flake, 285 nm bottom SiO2,
  600 nm high centered SiO2 disk of radius 1.5 um, and Si substrate.
- Conductivity: TaIrTe4 diag(14.4, 3.8, 1.0), SiO2 1.38, Si 145 W/(m K).
- Interfaces: G_bottom = G_top = 7.37e6 W/(m2 K);
  G_SiO2/Si = 1.1e9 W/(m2 K).
- Reference domain: 32 um lateral span and 20 um Si depth.
- Boundary condition: DeltaT=0 K on the far lateral and bottom boundaries;
  exposed surfaces adiabatic in the reference case.
- Source normalization: incident intensity 1 W/m2.
- Preserved optical power: 2.56071371086521e-12 W.
- Temperature quantity: DeltaT / incident intensity [K/(W/m2)].
- Reference Tmax: 3.120021567716e-07 K/(W/m2).
- Reference TaIrTe4 volume-average DeltaT:
  2.255081306258e-07 K/(W/m2).
- Reference active cells: 1,625,064.
- Reference energy-balance relative error:
  3.361661e-12.

The source was imported without clipping, smoothing, gain, global rescaling,
periodic tiling, or deletion outside a stored mask. Coarse/refined sensitivity
meshes use conservative source-energy restriction or piecewise-constant
subdivision, respectively; every case retained exactly the same total source
power.

## Domain, depth, and mesh convergence

The gate requires Tmax, TaIrTe4 volume-average temperature, and the common
TaIrTe4 3D probe-field NRMSE all to be below 1% for the final pair.

| family | comparison | Tmax change | flake average change | 3D probe NRMSE |
| --- | --- | --- | --- | --- |
| lateral_domain_um | domain_L4 → baseline_L8um_Si5um | 0.195503% | 0.268869% | 0.205604% |
| lateral_domain_um | baseline_L8um_Si5um → domain_L16 | 0.0608977% | 0.0840805% | 0.0643354% |
| lateral_domain_um | domain_L16 → domain_L32 | 0.00489969% | 0.00676634% | 0.00517751% |
| Si_depth_um | depth_D2_L32 → depth_D5_L32 | 0.169956% | 0.236324% | 0.181075% |
| Si_depth_um | depth_D5_L32 → depth_D10_L32 | 0.0589565% | 0.0816595% | 0.0625353% |
| Si_depth_um | depth_D10_L32 → final_native | 0.0178338% | 0.0246859% | 0.0189037% |
| thermal_mesh | mesh_coarse → final_native | 0.266464% | 0.332189% | 0.233076% |
| thermal_mesh | final_native → mesh_refined | 0.140694% | 0.0933887% | 0.066659% |

All final-pair metrics pass the 1% gate. The refined mesh keeps the native
optical x/y control volumes, subdivides source cells by two in z, and refines
the surrounding material mesh.

## Interface and boundary sensitivity

Changes below are relative to the 32 um x 32 um lateral, 20 um Si-depth native
reference.

| case | Tmax | flake average | Tmax change | average change |
| --- | --- | --- | --- | --- |
| Gbottom_1e6 | 8.829001043e-07 | 7.802225178e-07 | +182.979% | +245.984% |
| Gbottom_3e6 | 4.470438571e-07 | 3.527650809e-07 | +43.2823% | +56.4312% |
| final_native | 3.120021568e-07 | 2.255081306e-07 | +0% | +0% |
| Gbottom_1p5e7 | 2.621761266e-07 | 1.807499026e-07 | -15.9698% | -19.8477% |
| Gbottom_3e7 | 2.368894617e-07 | 1.589342407e-07 | -24.0744% | -29.5217% |
| Gbottom_1e8 | 2.182795550e-07 | 1.434554881e-07 | -30.0391% | -36.3857% |
| Gbottom_perfect | 2.098963610e-07 | 1.366881667e-07 | -32.726% | -39.3866% |
| Gtop_7p37e4 | 3.353679162e-07 | 2.253275850e-07 | +7.48897% | -0.0800617% |
| Gtop_7p37e5 | 3.297741068e-07 | 2.253790770e-07 | +5.6961% | -0.0572279% |
| Gtop_7p37e7 | 3.012621192e-07 | 2.255637168e-07 | -3.4423% | +0.0246493% |
| Gtop_perfect | 2.992757596e-07 | 2.255725132e-07 | -4.07895% | +0.02855% |
| oxide_si_perfect | 3.116148563e-07 | 2.251096798e-07 | -0.124134% | -0.17669% |
| convection_h10 | 3.119974382e-07 | 2.255041995e-07 | -0.00151237% | -0.00174325% |

G sweeps quantify physical-parameter sensitivity and are not numerical
convergence gates. The adiabatic top SiO2 disk has essentially zero net
steady heat removal; changing G_top changes local heat redistribution. The
h=10 W/(m2 K) case applies the Robin condition to every exposed solid-air
surface, including exact cell-center-to-surface conduction resistance.

## Gate accounting

- Cases executed: 22.
- Every case equation converged and conserved:
  `true`.
- Q mapping error in every case: 0.
- Required Q mapping error: <0.5%.
- Required energy-balance error: <1%.
- Production convergence gate: passed.
- Full field artifacts remain in the ignored validation output directory and
  are indexed by SHA-256 in `RAW_ARTIFACT_MANIFEST.json`.

The numerical model is now suitable as the steady thermal production path
for this geometry and source, subject to the stated material-property
assumptions. In particular, TaIrTe4 kz=1.0 W/(m K) remains an estimated input,
and the G sweeps should be retained when interpreting uncertainty.
