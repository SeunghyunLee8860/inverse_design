# Production CUDA thermal/PTE scenario controls

Status: `VALIDATED_PRODUCTION_CUDA_THERMAL_PTE_SCENARIOS`

The exact material-attributed volumetric Q was applied to the same explicit
3D anisotropic thermal operator for all four named bottom/design interface-G
scenarios. Matrix assembly was performed on the host; every production linear
forward and implicit-adjoint solve used CUDA float64. There was no CPU linear
solve fallback, Q modification, empirical normalization, or gradient
rescaling.

| bottom/design scenario | all-material Tmax rise (K) | TaIrTe4 Tmax rise (K) | TaIrTe4 average rise (K) | PTE response (A/W incident) | forward residual | adjoint residual | energy balance |
|---|---:|---:|---:|---:|---:|---:|---:|
| grown_grown | 5.903873e-10 | 2.433861e-10 | 3.113191e-11 | 2.184409e-13 | 8.217e-11 | 9.218e-11 | 9.700e-13 |
| grown_evaporated | 4.643457e-09 | 1.838899e-10 | 3.106044e-11 | 2.183537e-13 | 9.423e-11 | 7.607e-11 | 1.665e-12 |
| evaporated_grown | 3.723436e-09 | 3.495034e-09 | 8.853306e-10 | 8.537981e-14 | 9.248e-11 | 9.771e-11 | 2.678e-12 |
| evaporated_evaporated | 6.735633e-09 | 3.227252e-09 | 8.851586e-10 | 9.042585e-14 | 9.813e-11 | 8.113e-11 | 1.304e-11 |

The current is a near-null diagnostic: uniform rho=0.5 and the centered source
are symmetric under the present uniform 45-degree weighting surrogate. It is
not an optimized current or an experimental prediction. Consequently, the raw
relative difference between two approximately `1e-26 A` reciprocal forms is
ill-conditioned and is retained only as a diagnostic. The scale-aware Cauchy
normalized reciprocity error is used for the numerical gate; its worst value
is `1.008309e-15`. The largest PTE
bilinear signal is only `7.641437e-12` of its Cauchy scale,
which quantitatively identifies this baseline as a cancellation-dominated
null control.

The weighting field is `(15625, 15625) 1/m`, corresponding to a unit potential
difference across opposite diagonal equipotential lines of the finite 32 um
flake. It is a production surrogate, not the full experimental electrode
operator. Maxwell adjoint, combined AD-FD, coarse-gradient design-window
selection, and optimization have not run.
