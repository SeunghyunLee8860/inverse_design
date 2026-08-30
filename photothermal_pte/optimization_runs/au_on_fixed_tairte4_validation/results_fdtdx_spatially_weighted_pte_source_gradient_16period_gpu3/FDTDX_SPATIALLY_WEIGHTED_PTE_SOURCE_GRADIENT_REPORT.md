# FDTDX native-Yee spatially weighted PTE-source gradient

Status: **VALIDATED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT**

This checkpoint differentiates the Maxwell heat-source branch only. Native
Yee-cell powers `Q_c dV_c` for Au, TaIrTe4, and SiO2 are contracted with the
frozen explicit-thermal source adjoint in A/W. Each electric component remains
on its own staggered physical coordinates. The thermal/electrical direct
density terms are not included here and no optimization is run.

| metric | value |
|---|---:|
| weighted source objective | 2.684900000000e-18 A |
| Stage-68 source contraction | 2.684875446978e-18 A |
| contraction checkpoint difference | 0.000914486% |
| total optical P_Q | 2.477961052160e-13 W |
| gradient L2 norm | 6.328594206499e-18 A |
| strongest-direction AD-FD error | 0.004634203% |
| gradient-L2-normalized error | 0.004634418% |
| matched-volume Q/flux closure | 0.123096% |
| late-Q change | 0.019832% |
| weighted-objective late change | 0.038413% |
| reverse AD runtime | 2920.021 s |
| central-FD forward runtime | 216.371 s |

The raw gradient NPZ is outside Git and pinned by SHA-256 `235f3ed60bd04d089d7437da7e72e4658eaa389695dff986b10973532d3037a3`. No
clipping, smoothing, gain, objective matching, or gradient rescaling is used.
Passing this report does **not** certify the full PTE gradient. The next gate
must recompute Maxwell Q, explicit thermal transport, and the Au-aware
electrical weighting/current for each central-FD perturbation and compare that
end-to-end derivative with the sum of the three analytic/adjoint branches.
