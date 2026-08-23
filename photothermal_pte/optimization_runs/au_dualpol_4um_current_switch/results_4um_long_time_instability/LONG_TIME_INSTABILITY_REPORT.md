# Long-time FDTD instability isolation

Status: `DIAGNOSED_LONG_TIME_INSTABILITY_DRIVER_AU_DISPERSION`

| case | E spatial NRMSE | E2 change | closed TD (W) | closed phasor (W) | unstable |
|---|---:|---:|---:|---:|:---:|
| substrates_only | 0.0007% | 0.0001% | -3.330890e-17 | -5.743854e-17 | False |
| au_only | 80.7866% | 5.3991% | -1.056204e-12 | 7.394306e-14 | True |
| tairte4_only | 0.0013% | 0.0002% | 8.794632e-13 | 8.794765e-13 | False |
| full_dispersion | 53.7683% | 3.3093% | -6.625158e-13 | 2.907341e-13 | True |

Inferred driver: `AU_DISPERSION`.

This is an isolation diagnostic on the partial factor-8 grid, not a mesh certificate.
