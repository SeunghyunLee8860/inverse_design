# Au gray-law mismatch diagnostic

Status: `DIAGNOSED_GRAY_LAW_MISMATCH_SIGN_SENSITIVE`

This is a forward-only diagnostic at the robust beta=256 checkpoint. No gray law is promoted as a physical Au material.

## Findings

1. The implemented production relaxation is O3/TE1: optical Drude strength is rho^3, while thermal conductivity/contact and electrical conductivity/contact are linear in rho.
2. The mismatch is functionally large in the dilated eta=0.35 projection. Changing only TE1 to TE3 changes Ib from -2.033166 to -0.281729 nA, removing about 86% of the opposite-sign margin, although the sign remains negative.
3. The full factorial is sign-sensitive: at eta=0.35, O1/TE1 gives Ib=-0.172040 nA, while O1/TE3 gives Ib=+1.324187 nA.
4. A separate, more fundamental robust-objective omission is present. The optimizer used eta=0.35 and eta=0.65 but did not include nominal eta=0.50. Under the production O3/TE1 law, nominal Ib=+8.386359 nA and fails the requested sign.
5. The grayness constraint was evaluated only on nominal density. Nominal grayness is 0.3871%, but eta=0.35 grayness is 2.6246% with 382 cells in 0.01<rho<0.99.
6. Therefore the inconsistent law is a confirmed risk and performance amplifier, but it is not the sole cause of exact-binary sign failure. The missing nominal scenario and gray-only robust projections independently invalidate promotion.

| density | optical exponent | thermal/electrical exponent | Ia (nA) | Ib (nA) | min(Ia,-Ib) (nA) | physics gates | sign gate |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| eta_0.50_nominal | 1 | 1 | +6.360976 | +8.310396 | -8.310396 | True | False |
| eta_0.50_nominal | 1 | 3 | +6.347458 | +8.298428 | -8.298428 | True | False |
| eta_0.50_nominal | 3 | 1 | +6.439485 | +8.386359 | -8.386359 | True | False |
| eta_0.50_nominal | 3 | 3 | +6.429252 | +8.378101 | -8.378101 | True | False |
| eta_0.35 | 1 | 1 | +3.377372 | -0.172040 | +0.172040 | True | True |
| eta_0.35 | 1 | 3 | +3.706080 | +1.324187 | -1.324187 | True | False |
| eta_0.35 | 3 | 1 | +2.846991 | -2.033166 | +2.033166 | True | True |
| eta_0.35 | 3 | 3 | +3.278781 | -0.281729 | +0.281729 | True | True |
| eta_0.65 | 1 | 1 | +2.086029 | -1.803267 | +1.803267 | True | True |
| eta_0.65 | 1 | 3 | +2.072818 | -1.800106 | +1.800106 | True | True |
| eta_0.65 | 3 | 1 | +2.018108 | -1.881556 | +1.881556 | True | True |
| eta_0.65 | 3 | 3 | +2.005176 | -1.882478 | +1.882478 | True | True |

The reported 0.395% nominal value is a global grayness metric, not a gray-cell area fraction. The JSON records gray-cell counts separately for nominal, eta=0.35, and eta=0.65 projections.

The production mismatch is O3/TE1. O1/TE1 changes only the optical relaxation; O3/TE3 changes only the thermal/electrical relaxation. O1/TE3 closes the factorial. All four share identical rho=0 and rho=1 endpoints.

No clipping, smoothing, gain, current rescaling, or Q rescaling is used.
