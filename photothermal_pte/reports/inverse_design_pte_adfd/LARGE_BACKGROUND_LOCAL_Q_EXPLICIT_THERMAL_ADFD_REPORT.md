# Large-background local-Q explicit thermal AD–FD

Status: `VALIDATED_NAMED_LOCAL_Q_EXPLICIT_THERMAL_ADFD_SCENARIOS`

This certificate validates the discrete thermal-material/interface gradient
for two named large-TaIrTe4 footprint scenarios. It does **not** promote a
complete ideal-plane-wave thermal source, a PTE current, an experimental
prediction, or an optimization result.

## Fail-closed source-support correction

The first geometric embedding conserved total optical power but placed
staggered Yee control-volume boundary pieces in adjacent oxide/design/air
thermal cells. That precheck was rejected and is not promoted. For the
dominant `Qx` component, the affected power fraction was
`0.0312183341197`.

The validated map preserves every native source-cell energy and relocates only
the z-directed boundary pieces to the nearest exact TaIrTe4 thermal cell in
the same x-y column. It is a fixed linear operator with an exact transpose:
no clipping, deletion, smoothing, empirical gain, global rescaling, or tiling.

- native and mapped `P_Q`:
  `1.6890916194508477e-12 W`;
- relative power error:
  `2.39120708e-16`;
- 4 um / 6 um transpose errors:
  `2.30079961e-15` /
  `3.63081412e-16`;
- power and nonzero cells outside TaIrTe4: `0 W / 0` in both scenarios.

## Explicit thermal contract

- thermal domain: `32 × 32 um`, Si depth `20 um`;
- named TaIrTe4 footprints: `4 × 4 um` and `6 × 6 um`;
- protected design: `2 × 2 × 0.6 um`;
- core grid: `100 nm`; TaIrTe4 z cells: `25 nm`;
- TaIrTe4 kappa: `diag(14.4, 3.8, 1.0) W/(m K)`;
- bulk kappa: SiO2 `1.38`, Si `145`, air `0.026 W/(m K)`;
- gray design: `k=0.026+rho*(1.38-0.026)`;
- TaIrTe4/air sidewalls: `G=1 W/(m2 K)`, not adiabatic;
- TaIrTe4/bottom-SiO2: `G=7.37e6 W/(m2 K)`;
- gray top design contact:
  `G=1+rho*(7.37e4-1) W/(m2 K)`;
- SiO2/Si: named candidate `G=1.1e9 W/(m2 K)`;
- exposed top surface: Robin `h=10 W/(m2 K)`;
- far x/y and bottom Si: fixed `DeltaT=0` numerical truncation reservoirs.

The objective is the volume-average `DeltaT` in the central `2 × 2 um`
TaIrTe4 region. `Q` is fixed during these thermal-material FD checks.
The optical source came from a uniform `rho=0.5` optical forward, whereas
the thermal control uses
`rho=0.5+0.04*cos(pi*xhat)*cos(pi*yhat)` (range
`0.460545547864` to
`0.54`). This deliberate mismatch excites the
thermal material/interface derivatives; it is not a self-consistent combined
optical-thermal design state.

## Scenario results

| quantity | 4 um flake | 6 um flake | 6 vs 4 |
|---|---:|---:|---:|
| central 2 um average `DeltaT` | `8.277330691347e-08 K` | `8.060575597349e-08 K` | `-2.61866%` |
| TaIrTe4 `Tmax` | `1.070236178596e-07 K` | `1.052159876462e-07 K` | `-1.689%` |
| whole-flake average `DeltaT` | `4.086985646559e-08 K` | `1.873637791982e-08 K` | `-54.156%` |
| worst AD–FD relative error | `1.03181156e-04` | `1.30270810e-04` | — |
| energy-balance error | `3.18138146e-12` | `3.50000980e-12` | — |
| forward residual | `1.01479339e-11` | `1.01595281e-11` | — |
| adjoint residual | `1.00292822e-11` | `9.75025429e-12` | — |

Both global hotspots are inside TaIrTe4 at approximately
`(x,y,z)=(0.05,0.05,-0.0125) um`. The much lower whole-flake average for the
6 um case mostly reflects averaging the same local source over a larger
unilluminated flake volume; the central 2 um objective is the more comparable
quantity.

## Interfaces and external boundaries

For 4 um / 6 um, the mean **contact-only** jumps are:

- TaIrTe4/bottom-SiO2:
  `1.42759589e-08` /
  `6.34909582e-09 K`;
- SiO2/Si:
  `2.62085477e-11` /
  `1.72999385e-11 K`;
- gray top design contact:
  `3.84693774e-08` /
  `3.08691897e-08 K`.

The reported adjacent-cell jump is kept separately because it also contains
the two half-cell conduction drops and is not equal to `q''/G`.

Approximately
`80.4772%` /
`80.5105%` leaves through the four far lateral
reservoirs, while
`19.5224%` /
`19.4889%` leaves through the
bottom reservoir. These are numerical boundary flux partitions, not physical
heat-path fractions.

## Numerical error versus physical-model variation

The worst discrete AD–FD error is
`1.30270810e-04`;
the worst energy error is
`3.51866121e-12`;
and the worst linear residual is
`1.02258687e-11`.
These certify the assembled discrete equations and gradient.

The 4-to-6 um differences are a named footprint-scenario variation, not a
confidence interval. This run does not add a new spatial mesh-convergence
bound. More importantly, the source is only the validated local `Omega_Q`
certificate; the absorption outside that local volume under a truly extended
ideal plane wave is not included. Therefore these temperatures are not a
final plane-wave or experimental prediction.

No terminal PTE, transient, adjoint optimization, gradient-based optimization,
or geometry update was run.
