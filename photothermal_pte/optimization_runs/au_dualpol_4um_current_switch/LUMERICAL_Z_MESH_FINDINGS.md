# Lumerical exact-Au z-mesh findings

Date: 2026-08-24. These are local RTX 6000 Ada development results from
Lumerical v261 solver `8.35.4413`; they are not B200 promotion evidence. Raw
FSP/NPZ/JSON/log files remain outside Git under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.

## Fixed numerical and physical inputs

All exact-full rows use Ea, CV0, Au MCM6, the same sampled physical material
tables, a 100 nm x/y flake mesh, a 200 nm outer x/y limit, eight PML layers,
the 20 x 20 x 6 um domain, the 1 ps / 1e-7 decay contract, and the same exact
8 x 8 x 0.05 um full-Au geometry hash
`152a9104b2135a152ba75cc9a2d52917133bffc3fabdfa9d8f98679d9c88e2d1`.
Each numerical mesh uses its own passed all-air source calibration. Q, flux,
and fields are compared after division by that mesh's incident power (field
amplitude by the square root of incident power). No field or Q array is
rescaled in the saved solver result.

| stack z limit | bulk/air/PML z limit | realized grid | source-only incident power (W) | native Q (W) | Q/flux closure |
|---:|---:|---:|---:|---:|---:|
| 5 nm | 50 nm | 183 x 183 x 212 | 3.178309584e-14 | 1.019653272e-15 | 0.08935% |
| 2.5 nm | 50 nm | 183 x 183 x 303 | 3.178303778e-14 | 1.033393483e-15 | 0.09120% |
| 2.5 nm | 25 nm | 183 x 183 x 410 | 3.180049736e-14 | 1.035053925e-15 | 0.09263% |

The linked 2.5/25 nm all-air source gate passed with a realized effective
waist of 4.001893 um, Gaussian-fit NRMSE of 0.08537%, and incident-power
closure of 0.06172%. Its matching exact-empty control passed Q/flux closure at
0.01453%. The full result therefore is not relying on a failed source,
background stack, or closed-surface control.

## Pairwise result

The convergence contract requires every scalar and field metric to be below
0.5%. NRMSE denominators below are the finer result. Complex-field NRMSE is
computed jointly over Ex, Ey, and Ez on the common fixed 81 x 81 endpoint
plane.

| isolated change | source-normalized Q change | source-normalized flux change | complex field NRMSE | E2 NRMSE | result |
|---|---:|---:|---:|---:|:---:|
| stack 5 -> 2.5 nm, bulk fixed at 50 nm | 1.3298% | 1.3316% | 0.9850% | 1.1618% | fail |
| bulk/air/PML 50 -> 25 nm, stack fixed at 2.5 nm | 0.1056% | 0.1070% | 0.1087% | 0.0556% | pass |
| linked 5/50 -> 2.5/25 nm | 1.4340% | 1.4372% | 1.0109% | 1.1977% | fail |

The current dominant z error is the thin Au/TaIrTe4/SiO2 stack, not the
50-to-25 nm bulk/air/PML refinement. The 2.5/25 nm mesh is not a z-converged
production mesh because the linked finest pair fails the 0.5% gate. After the
MCM6 duration/decay axis is independently closed, extend the full-domain z
axis to 1.25 nm stack and 12.5 nm bulk/air/PML, with a new matching source-only
control. Do not proceed to x/y convergence or optimization on the present
2.5/25 nm result.

The coordinate-only material partition is not a material-resolved convergence
metric. Refining z changes which staggered/interface samples are assigned to a
named layer; for the isolated stack refinement it changed the reported Au
partition by about +56% and the air partition by about -98.8% while total
native Q changed only 1.33%. Use total/component native-Yee Q, saved epsilon,
conservative Q remapping, and downstream temperature/current for promotion.

## Time-order caveat

The earlier 2 ps / 1e-9 strict-decay diagnostic used the now-rejected Au MCM20
fit. It proves that longer decay did not repair the MCM20 energy failure, but
it is not a duration-convergence certificate for MCM6. The correct next run is
a matching MCM6 source-only/empty/full 1 ps versus 2 ps duration/decay
comparison. Spatial z evidence above remains a development diagnostic until
that prior contract axis is closed.
