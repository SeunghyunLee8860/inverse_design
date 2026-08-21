# Full combined FDTDX--thermal--weighting PTE directional AD--FD

Status: **VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_DIRECTIONAL_ADFD**

This smoke recomputes the entire forward chain at `rho +/- 0.01 d`: FDTDX
native-Yee Au/TaIrTe4/SiO2 Q, conservative material-overlap remap, explicit
3-D thermal transport/contact, and Au-aware electrical weighting/current.
The analytic derivative is the unscaled sum of Maxwell-source, direct thermal,
and direct electrical/weighting branches.

| quantity | value |
|---|---:|
| optical-source AD contribution | 6.150491294735e-18 A |
| direct thermal AD contribution | -3.092173992360e-19 A |
| direct electrical/weighting AD contribution | 6.117611627053e-19 A |
| combined AD | 6.453035058204e-18 A |
| end-to-end central FD | 6.452153139305e-18 A |
| strong-direction error | 0.013666730% |
| gradient-L2-normalized error | 0.013666730% |
| central midpoint objective error | 0.001117696% |
| worst linear residual | 8.252e-10 |
| worst thermal energy balance | 0.000000001% |
| worst terminal balance | 0.000000000% |

No Q clipping, smoothing, gain, global rescaling, density clipping, or gradient
rescaling is used. Raw FDTDX/thermal artifacts stay outside Git and are pinned
by SHA in the manifest. A one-direction smoke is not yet a multi-direction or
latent/filter/projection certificate and does not authorize optimization.
