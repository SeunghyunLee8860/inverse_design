# Au latent/filter/projection mapping validation

Status: **VALIDATED_AU_LATENT_FILTER_PROJECTION_MAPPING**

The current 20x20 Au design layout uses 500 nm pixels. This solver-free gate
tests a finite nonperiodic 750 nm conic-filter scenario and eta=0.5 tanh
projection at beta 1, 2, 4, and 8. The filter is row-normalized at truncated
boundaries and its exact transpose is used.

| metric | value | gate |
|---|---:|---:|
| constant preservation max error | 0.000e+00 | <1e-12 |
| opposite-edge wrap | 0.000e+00 | =0 |
| worst JVP/VJP dot error | 2.642e-17 | <1e-12 |
| worst mapping-only FD / gradient-norm error | 8.721e-07 | <1e-6 |
| h-to-h/2 regression count | 0 | 0 |

The 750 nm radius is an explicit numerical scenario, not a final fabrication
minimum-feature claim. This checkpoint validates only the mapping calculus;
the full latent Maxwell/thermal/electrical AD--FD remains required before
optimization.
