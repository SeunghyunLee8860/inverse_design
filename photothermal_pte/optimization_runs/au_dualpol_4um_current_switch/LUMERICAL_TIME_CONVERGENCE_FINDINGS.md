# Lumerical MCM6 time/decay convergence findings

Date: 2026-08-24. These are local RTX 6000 Ada development results from
Lumerical v261 solver `8.35.4413`; they are not B200 promotion evidence. Raw
FSP/NPZ/JSON/log files remain outside Git under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.

## Controlled comparison

Both durations use Ea, CV0, the same 100 nm x/y, 5 nm stack-z, 50 nm
bulk/air/PML-z mesh, eight PML layers, the same domain and source object, Au
MCM6, and the same exact empty/full geometry hashes. Each duration uses its
own passed all-air source calibration. Q, flux, and endpoint fields are
normalized by the corresponding source-only incident power before pairwise
comparison.

| contract | final auto-shutoff | source-only incident power (W) | empty closure | full closure |
|---|---:|---:|---:|---:|
| 1 ps / 1e-7 | 9.41e-8 full | 3.178309584e-14 | 0.01644% | 0.08935% |
| 2 ps / 1e-9 | 9.88e-10 full | 3.178310249e-14 | 0.01756% | 0.07394% |

The 2-ps source-only control independently passed with a realized effective
waist of 4.001790 um, Gaussian-fit NRMSE of 0.08415%, incident-power closure
of 0.06180%, and final auto-shutoff of 8.50e-10.

## Pairwise stationarity

All quantities below use the 2-ps result as the denominator/reference. The
contract limit is 0.5% for every scalar and field metric.

| case | normalized Q change | normalized flux change | complex field NRMSE | E2 NRMSE | result |
|---|---:|---:|---:|---:|:---:|
| exact empty | 0.000085% | 0.001034% | 0.001979% | 0.001294% | pass |
| exact full Au MCM6 | 0.004558% | 0.010858% | 0.001835% | 0.001351% | pass |

For exact-full, the source-normalized mean-E2 change is 0.000908%. The
source-only incident-power difference between the two contracts is
0.0000209%. These results close the MCM6 duration/decay axis for this Ea exact
control on the current RTX development host. They also show that the failed
5/50-to-2.5/25-nm z pair is spatial, not a 1-ps transient artifact.

The selected efficient development setting remains 1 ps / 1e-7 because its
observables are stationary against the stricter result. Final promotion still
requires both polarizations and the final optimized binary geometry to repeat
the duration/decay check on the B200; a topology can support a lifetime not
present in the full-sheet control.
