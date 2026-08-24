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
| 1.25 nm | 12.5 nm | 183 x 183 x 807 | 3.180486510e-14 | 1.042597023e-15 | 0.09328% |
| 0.625 nm | 6.25 nm | 183 x 183 x 1600 | 3.180594992e-14 | 1.046347307e-15 | 0.09339% |

The linked 2.5/25 nm all-air source gate passed with a realized effective
waist of 4.001893 um, Gaussian-fit NRMSE of 0.08537%, and incident-power
closure of 0.06172%. Its matching exact-empty control passed Q/flux closure at
0.01453%. The full result therefore is not relying on a failed source,
background stack, or closed-surface control.

The extended 1.25/12.5 nm source gate also passed: realized effective waist
4.001931 um, Gaussian-fit NRMSE 0.08603%, incident-power closure 0.06173%, and
the requested GPU/time/mesh log gates all true. Its matching exact-empty and
exact-full controls passed Q/flux closure at 0.01523% and 0.09328%,
respectively. The three runs used the same source-calibration hash
`ae62968dba0ae59cf84353a2068898d672dc4e445005e67c473343df0e7d0c80`.

The 0.625/6.25 nm source, empty, and full runs also passed every individual
solver gate. The source realized a 4.001947 um effective waist, 0.08636%
Gaussian-fit NRMSE, and 0.06170% incident-power closure. The realized maximum
z steps were 0.625 nm in the stack, 6.240 nm in Si, and 6.249 nm in the upper
air/PML. Empty and full Q/flux closure were 0.01679% and 0.09339%. All three
used source-calibration hash
`b6bbdc7265d6df1e1c904ccfb0c57cb232b1163743d8b9a0124b984b8753c801`.
The full run took 2001.9 s on the RTX development GPU; its raw FSP and NPZ
were 1.47 GB and 384 MB and remain outside Git.

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
| linked 2.5/25 -> 1.25/12.5 nm, exact empty | 0.4915% | 0.4909% | 0.3418% | 0.6043% | fail |
| linked 2.5/25 -> 1.25/12.5 nm, exact full | 0.7099% | 0.7105% | 0.5270% | 0.6237% | fail |
| linked 1.25/12.5 -> 0.625/6.25 nm, exact empty | 0.2413% | 0.2398% | 0.1698% | 0.3003% | pass |
| linked 1.25/12.5 -> 0.625/6.25 nm, exact full | 0.3550% | 0.3551% | 0.2669% | 0.3176% | pass |
| staircase linked 5/50 -> 2.5/25 nm, exact empty | 0.9522% | 0.9541% | 0.6656% | 1.1752% | fail |
| staircase linked 5/50 -> 2.5/25 nm, exact full | 1.3921% | 1.3954% | 1.0105% | 1.1966% | fail |

The 5-to-2.5 nm isolated result showed that the thin Au/TaIrTe4/SiO2 stack was
then the dominant z error, not the 50-to-25 nm bulk/air/PML refinement. The
subsequent linked 2.5/25-to-1.25/12.5 nm refinement reduced every exact-full
error substantially but still failed. The final linked
1.25/12.5-to-0.625/6.25 nm pair passes every scalar and common endpoint-plane
metric for both exact-empty and exact-full Ea. This closes that Maxwell
sub-gate and establishes that the coarser member is within 0.5% of the finer
reference for these observables. It is not yet a full z-mesh or production
certificate. The subsequent material-aware conservative remap and custom CUDA
thermal comparison failed: empty/full remapped-Q NRMSE is 0.9730%/1.8576%
and TaIrTe4 temperature NRMSE is 1.0058%/1.7397%. See
`LUMERICAL_Z_MULTIPHYSICS_FINDINGS.md`. Required Eb, simple-geometry,
final-topology, and B200 controls also remain open. Do not begin x/y
convergence or optimization until those same-axis gates are closed.

After the official material filter rejected the CV0 downstream certificate,
the fixed-mesh interface triage selected staircase as the next development
candidate. The first staircase linked pair used passed, hash-matched
source-only, exact-empty, and MCM6 exact-full results on 183 x 183 x 212 and
183 x 183 x 410 grids. It failed all four Maxwell observables for both
controls as shown above. This does not invalidate the interface choice; it
proves that 2.5/25 nm is not yet a converged staircase reference. Because the
Maxwell prerequisite failed, script 28 did not run downstream PDE comparison.
The next staircase pair is 2.5/25 to 1.25/12.5 nm.

The new `27_compare_lumerical_4um_control_pair.py` reproduces the four table
metrics. It verifies the raw NPZ SHA-256 values; exact case, polarization,
geometry, GPU UUID, and solver version; all fixed non-z mesh fields; and each
run's source-calibration and solver gates before comparing anything. The raw
comparison JSON files are stored beside the corresponding finest empty/full
artifacts, not in Git.

The coordinate-only material partition is not a material-resolved convergence
metric. Refining z changes which staggered/interface samples are assigned to a
named layer; for the isolated stack refinement it changed the reported Au
partition by about +56% and the air partition by about -98.8% while total
native Q changed only 1.33%. Use total/component native-Yee Q, saved epsilon,
conservative Q remapping, and downstream temperature/current for promotion.

## Time-order resolution

The earlier 2 ps / 1e-9 strict-decay diagnostic used the now-rejected Au MCM20
fit. It proves that longer decay did not repair the MCM20 energy failure, but
it was not a duration-convergence certificate for MCM6. A matching MCM6
source-only/empty/full 1 ps versus 2 ps duration/decay comparison has now
passed by a wide margin. See `LUMERICAL_TIME_CONVERGENCE_FINDINGS.md`. The
earlier z failures were therefore spatial rather than transient; the extended
Maxwell endpoint sub-gate now passes, while its downstream and promotion scope
remains explicitly open as described above.
