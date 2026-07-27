# Large-background plane-wave optical AD–FD certificate

Status: `VALIDATED_MIXED_CPU_TFSF_GPU_FIELDREGION_OPTICAL_ADFD`

The local optical solver chain is validated. This is not yet a final
plane-wave thermal/PTE prediction because the physical illumination footprint
and the actual thermal TaIrTe4 lateral footprint are not yet fixed.

## Fixed optical contract

- non-periodic large TaIrTe4/SiO2/Si background extended through lateral PML;
- finite design: `x,y=[-1,1] um`, `z=[0,0.6] um`;
- material interpolation:
  `epsilon(rho)=1+rho*(1.38^2-1)`;
- normal-incidence x-polarized TFSF, 3–6 um source, 4 um analysis;
- six PML boundaries and no periodic/Bloch boundary;
- fixed Q/flux volume:
  `x,y=[-1.15,1.15] um`, `z=[-0.15,0.75] um`;
- central incident intensity: `1 W/m2`, realized `0.9990759709401065 W/m2`;
- no Q clipping, smoothing, gain, global rescaling, tiling, or old-artifact
  crop.

The standard 24-layer geometry has outer x/y bounds `[-3.2,3.2] um`.
Realized PML-inner x/y bounds are `[-2.0,2.0] um`; z bounds are
`[-2.0096774,1.81] um`. The minimum realized TFSF-to-PML-inner gap is
`209.677 nm`, and every geometry gate passed.

## Forward and convergence gates

The flat-background PML-24 baseline gives:

- `P_Q = 1.3567412718462558e-12 W`;
- `P_six = 1.3567343935235152e-12 W`;
- six-face closure `5.0697637e-6`;
- source-off maximum `E^2 = 0`.

Relative changes in `P_Q` from this baseline are:

| variation | relative change |
|---|---:|
| matched-inner PML 24→32 | `9.6610160e-6` |
| TaIrTe4 dz 5→2.5 nm | `2.7071825e-4` |
| transverse mesh 50→25 nm | `2.0356961e-6` |
| TFSF span 2.6→3.0 um | `2.3772721e-8` |
| TFSF span 2.6→3.4 um | `1.0346612e-8` |

Endpoint and gray closures also pass: rho=0 gives `P_Q=1.356746074711707e-12
W`; rho=1 gives `P_Q=2.0897370029545985e-12 W` with closure
`4.9672456e-4`; rho=0.5 gives `P_Q=1.6891228343790494e-12 W` with closure
`3.0154292e-4`.

The standard PML FieldRegion adjoint diverged at a material interface. The
forward and adjoint operators were therefore both changed to x/y stabilized
PML with 32 layers while preserving the non-PML interior. This changes the
rho=0.5 `P_Q` by only `1.8479963e-5`; it is not an adjoint-only change.

## Direct mixed CPU/GPU AD–FD

The production combination was tested directly:

- forward: CPU TFSF;
- adjoint: GPU vector FieldRegion with the TFSF object absent;
- equivalence control: CPU FieldRegion adjoint;
- scalar physical density: rho=0.5;
- objective: native-Yee absorbed power in the fixed Q volume.

Results:

| quantity | value |
|---|---:|
| stabilized base `P_Q` | `1.689091619450848e-12 W` |
| stabilized base `P_six` | `1.6895947794697648e-12 W` |
| six-face closure | `2.9779923e-4` |
| GPU adjoint gradient | `7.316714058728351e-13 W/rho` |
| CPU adjoint gradient | `7.316713533711392e-13 W/rho` |
| CPU/GPU gradient difference | `7.1755845e-8` |
| CPU/GPU complex-field NRMSE | `2.1997802e-5` |
| centered FD, h=0.01 | `7.317295351329038e-13 W/rho` |
| mixed AD–FD relative error | `7.9440910e-5` |
| FD h=0.02→0.01 change | `1.0504746e-5` |

The initially observed 4.96% mismatch was traced to integrating staggered
design-monitor samples outside the finite design support. The old overlap
volume was 3.431% too large for Ex/Ey and 7.151% too large for Ez. The
corrected metric intersects each raw Yee control cell with the exact finite
design box. It integrates exactly `2.4e-18 m3` for every component and uses
no fitted scale factor.

## Local optical-to-thermal Q mapping

The stabilized rho=0.5 native Q was embedded into a 100 nm thermal
certificate grid without deleting any nonzero source cell:

- native `P_Q = 1.6890916194508481e-12 W`;
- mapped `P_Q = 1.6890916194508477e-12 W`;
- relative power error `2.3912071e-16`;
- transpose dot-test error `8.0786570e-16`.

The target-only thermal cells receive exact zero. This proves the mapping
operator and its transpose, but it does not assert that the local TFSF Q is
the complete physical plane-wave heating distribution.

## Remaining physical gate

Before fixed-Q thermal, combined physical-density, or full-latent PTE AD–FD
is promoted, two physical inputs must be fixed:

1. the actual finite illumination footprint used to construct
   `Q_flat,physical`;
2. the actual thermal TaIrTe4 lateral footprint.

No thermal solve, PTE solve, adjoint optimization, gradient-based
optimization, or final experimental prediction is reported here.
