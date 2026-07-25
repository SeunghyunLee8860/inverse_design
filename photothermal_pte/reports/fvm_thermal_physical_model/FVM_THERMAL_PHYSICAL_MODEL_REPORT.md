# FVM thermal physical-model sensitivity report

## Status and scope

**Status: `VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS` with
`BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED`.**

This report does not promote one arbitrary parameter set as a final
experimental prediction. It separates:

1. numerical convergence of the independent conservative Cartesian FVM; and
2. physical-model variation caused by uncertain material, interface,
   boundary, and fabrication assumptions.

It is not a Lumerical HEAT result. No optical geometry or Q was changed, and
no transient, PTE, adjoint, gradient, or optimization calculation was run.

## Immutable numerical checkpoint

PR #4 commit
`437ec0644b15a4b9a6919a0151e4aa531fb1e0ab` remains the immutable numerical
checkpoint. Its final-pair numerical metrics are:

| Numerical refinement | Tmax | flake average | common 3D flake NRMSE |
| --- | ---: | ---: | ---: |
| lateral 16 to 32 um | +0.00489969% | +0.00676634% | +0.00517751% |
| Si depth 10 to 20 um | +0.0178338% | +0.0246859% | +0.0189037% |
| native to refined mesh | +0.140694% | +0.0933887% | +0.066659% |

The promoted publication metadata now records
`provisional_until_sensitivity_passes=false` and `next_required_gate=null`.
Raw per-case JSON retains its original provisional fields as immutable
provenance.

## G_top and TaIrTe4 kz scenarios

`G_top=7.37e6 W/(m2 K)` is the PR #4 numerical-convergence checkpoint
scenario. `G_top=7.37e4 W/(m2 K)` is the earlier contract's named
evaporated-SiO2 estimate scenario. The repository contains no traceable
literature source that establishes either as uniquely correct, so neither is
promoted.

TaIrTe4 uses fixed `kx=14.4` and `ky=3.8 W/(m K)`. The values
`kz=0.5, 1.0, 2.0 W/(m K)` are numerical scenarios, not a confidence
interval, because the repository does not establish a sourced physical
range.

| scenario | Tmax | flake average | Tmax change | average change | 3D NRMSE | hotspot (x,y,z) m | mean top jump K | max top jump K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scenario_control_Gtop_7p37e6_kz_1 | 3.120021568e-07 | 2.255081306e-07 | +0% | +0% | +0% | (1.333333e-08, -9.933333e-07, -1.500000e-08) | 7.705867e-09 | 3.908260e-08 |
| scenario_evaporated_SiO2_estimate_Gtop_7p37e4 | 3.353679162e-07 | 2.253275850e-07 | +7.48897% | -0.0800617% | +2.15495% | (1.333333e-08, -9.933333e-07, -5.000000e-09) | 2.530374e-08 | 9.582006e-08 |
| scenario_kz_0p5 | 3.504130502e-07 | 2.490834788e-07 | +12.3111% | +10.4543% | +8.8041% | (1.333333e-08, -9.933333e-07, -1.500000e-08) | 7.639571e-09 | 3.847880e-08 |
| scenario_kz_2p0 | 2.923568781e-07 | 2.137021666e-07 | -6.29652% | -5.23527% | +4.40346% | (1.333333e-08, -9.933333e-07, -1.500000e-08) | 7.728442e-09 | 3.923746e-08 |

Direct G_top comparison:

- checkpoint scenario Tmax:
  `3.120021567716e-07 K/(W/m2)`;
  evaporated-estimate scenario Tmax:
  `3.353679161923e-07 K/(W/m2)`.
- checkpoint/evaporated mean top-interface jump:
  `7.705867068658e-09` /
  `2.530373807392e-08 K`.
- checkpoint/evaporated maximum top-interface jump:
  `3.908259531196e-08` /
  `9.582006449597e-08 K`.

## Boundary-condition robustness

Every case uses a fixed bottom temperature on the same 32 um lateral,
20 um Si-depth geometry. Artificial lateral/bottom truncation-boundary flux
is reported only as a **numerical boundary flux**, not as a physical
heat-path fraction.

| scenario | Tmax | Tmax change | 3D NRMSE | bottom numerical fraction | lateral numerical fraction |
| --- | --- | --- | --- | --- | --- |
| scenario_control_Gtop_7p37e6_kz_1 | 3.120021568e-07 | +0% | +0% | 0.196019 | 0.803981 |
| scenario_far_xy_adiabatic_bottom_fixed | 3.121505974e-07 | +0.0475768% | +0.0503934% | 1.000000 | 0.000000 |
| scenario_exposed_convection_h5 | 3.119997974e-07 | -0.000756197% | +0.00066373% | 0.196018 | 0.803974 |
| scenario_exposed_convection_h10 | 3.119974382e-07 | -0.00151237% | +0.00132744% | 0.196016 | 0.803966 |
| scenario_exposed_convection_h20 | 3.119927199e-07 | -0.00302462% | +0.00265477% | 0.196012 | 0.803952 |

Exposed convection was evaluated at `h=0,5,10,20 W/(m2 K)`. Its small effect
under these numerical boundaries does not validate an experimental ambient
heat-transfer coefficient.

## Top-disk fabrication geometry

Repository optical geometry defines a radius-1.5-um SiO2 disk at
`z=0...600 nm` touching the 2x2-um flake, but does not establish how the disk
outside the flake is fabricated or thermally supported. Therefore:

- scenario A: suspended/overhanging disk outside the flake;
- scenario B: a 100 nm SiO2 support annulus fills the gap outside the flake
  and connects the disk to the surrounding bottom oxide.

Scenario B changes Tmax by
`-39.7356%` and the flake-average
temperature by `-37.043%`.
Its common 3D flake-field NRMSE is
`+27.5386%`.
This large difference is why fabrication geometry remains a blocker.

For the maximum-variation scenario B, native-to-refined numerical changes
are:

- Tmax: `+0.78917%`;
- flake average:
  `+0.74338%`;
- common 3D flake-field NRMSE:
  `+0.522514%`.

The refined scenario-B Tmax is
`1.865539374507e-07 K/(W/m2)`. The physical
support-geometry variation is much larger than the associated numerical
mesh error.

## Optical dependency and fail-closed reproduction

PR #3 commit `053260da6fd0caec28ce155221bd18f683a0e5e7` is not in PR #4
ancestry. A clean checkout must supply the external raw PR #3 NPZ:

```bash
python photothermal_pte/validation/photothermal_stage1/40_reproduce_fvm_thermal_physical_model.py \
  --pr3-q-artifact /absolute/path/to/finite_q_on_artifact.npz \
  --output-root /absolute/path/to/new_output
```

The entry point verifies SHA-256
`7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794` before creating output or starting an import/thermal
solve. Missing or mismatched artifacts fail closed. The raw NPZ is not
committed.

All ten cases preserve `P_Q=2.56071371086521e-12 W`; mapping error is zero.
Clipping, smoothing, gain, global rescaling, tiling, and source deletion are
all false. Every energy-balance error is below 1%, every linear residual is
below `1e-8`, and every Q mapping error is below 0.5%.
