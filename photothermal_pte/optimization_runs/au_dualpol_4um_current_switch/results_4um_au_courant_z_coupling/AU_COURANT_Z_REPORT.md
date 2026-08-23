# Au-only z/Courant coupling diagnostic

Status: `DIAGNOSED_AU_ADE_TIME_STEP_DEPENDENT_INSTABILITY`

Inference: `COURANT_HALVING_REMEDIATES_IDENTICAL_FINE_Z_GEOMETRY`

| case | dt (s) | E spatial NRMSE | E2 change | Q (W) | closed TD (W) | closed phasor (W) | failed |
|---|---:|---:|---:|---:|---:|---:|:---:|
| partial_f1_cf0p5 | 3.209721e-17 | 0.0022% | 0.0003% | 9.794223e-14 | 1.006416e-13 | 1.001498e-13 | False |
| partial_f2_cf0p5 | 1.651388e-17 | 0.0049% | 0.0001% | 1.041408e-13 | 1.072478e-13 | 1.072978e-13 | False |
| partial_f4_cf0p5 | 8.318327e-18 | 0.0103% | 0.0003% | 1.058260e-13 | 1.095991e-13 | 1.096017e-13 | False |
| partial_f8_cf0p5 | 4.166939e-18 | 80.7866% | 5.3991% | 3.414582e-13 | -1.056204e-12 | 7.394306e-14 | True |
| partial_f8_cf0p25 | 2.083470e-18 | 0.1124% | 0.0019% | 1.091537e-13 | 1.104976e-13 | 1.104562e-13 | False |

This Au-only partial-material-z sweep diagnoses time/spatial coupling; it is not a mesh certificate.
A production z-mesh sweep must use the stable time contract and refine the full z domain.
