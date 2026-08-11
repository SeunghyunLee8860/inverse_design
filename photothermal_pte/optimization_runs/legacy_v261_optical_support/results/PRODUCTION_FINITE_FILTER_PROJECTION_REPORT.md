# Production finite filter/projection validation

Status: `VALIDATED_PRODUCTION_FINITE_FILTER_PROJECTION`

The frozen 373×373, 50 nm nodal window uses a 500 nm conic radius and tanh
projection with eta=0.5. The filter is finite and nonperiodic. Its forward
operator is `D^-1 C`, where `C` is zero-padded convolution and `D` is the
truncated edge-kernel sum. The exact transpose is `C D^-1`; using the forward
normalization order as the transpose would be wrong at the boundary.

## Gates

| gate | result | limit |
|---|---:|---:|
| constant preservation | 0.000e+00 | <1e-14 |
| opposite-edge wrap | 0.000e+00 | exactly 0 |
| worst JVP/VJP Cauchy error | 1.290e-15 | <1e-12 |
| worst mapping FD at h=2.5e-4 | 8.706e-06 | <1e-5 |
| non-monotone h→h/2 trajectories | 0 | 0 |

Five directions (uniform, smooth asymmetric, central localized,
design-edge localized, and fixed-seed random) were checked at beta
2, 4, 8, 16, and 32. Centered FD used h=0.001, 0.0005, and 0.00025 without
latent clipping. All 25 trajectories converge monotonically under h→h/2.

The first execution is retained as a fail-closed diagnostic. It used the
maximum error over *all* FD steps as the final gate, so the expected beta=32
coarse-step truncation error `1.393e-04`
failed. No result was rescaled. The corrected certificate separately requires
monotonic step convergence and the declared finest-step tolerance.

This validates only the latent→finite-filter→projection mapping and its exact
transpose. It is not an exact-binary DRC certificate, gray-law certificate,
full Maxwell/thermal latent AD-FD certificate, or authorization to optimize.
No Maxwell solve, thermal solve, or optimizer iteration ran here.
