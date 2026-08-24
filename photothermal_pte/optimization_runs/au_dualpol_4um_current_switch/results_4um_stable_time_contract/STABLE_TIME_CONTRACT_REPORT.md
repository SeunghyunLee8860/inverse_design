# Factor-8 reduced-Courant time/material validation

Status: `VALIDATED_FACTOR8_CF0P25_TIME_MATERIAL_CONTRACT`

| case | E NRMSE | E2 change | Q (W) | Q/phasor | TD/phasor | pass |
|---|---:|---:|---:|---:|---:|:---:|
| au_only_32p | 0.1468% | 0.0039% | 1.092229e-13 | 1.1634% | 0.0374% | True |
| full_dispersion_32p | 0.1834% | 0.0124% | 3.492413e-13 | 0.3549% | 0.0126% | True |
| full_dispersion_40p | 0.1878% | 0.0022% | 3.492436e-13 | 0.3567% | 0.0126% | True |

Full-material 32/40-period Q change: 0.0007%.

This validates a time/material contract only on the partial factor-8 grid.
It is explicitly not an optical, thermal, or electrical mesh certificate.
