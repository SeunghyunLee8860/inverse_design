# Finite T/Z Maxwell–thermal–electrical/PTE and Au-effect report

Status: **VALIDATED_FINITE_T_Z_MULTIPHYSICS_AND_AU_EFFECT_FORWARD**

The periodic stage was used only to screen optical absorption. This report uses a finite 20 x 20 um TaIrTe4 flake, finite Gaussian Maxwell Q, explicit 32 x 32 um thermal domain, and finite top-bottom / left-right TaIrTe4 terminals.
All values below correspond to 285 uW incident power through the certified linear source-power scaling. Raw Maxwell Q is unchanged.

## Primary results

| case | absorbed (uW) | Tmax (K) | Ta avg (K) | I top-bottom (nA) | I left-right (nA) |
|---|---:|---:|---:|---:|---:|
| T_Ea_Au_on | 54.6744 | 0.660367 | 0.150907 | 0.00278213 | -3.1086e-08 |
| T_Eb_Au_on | 54.2192 | 0.728208 | 0.148533 | -0.00754081 | 1.0728e-06 |
| T_Ea_Au_off | 54.8635 | 0.653010 | 0.151449 | 1.4087e-08 | -5.24672e-08 |
| T_Eb_Au_off | 53.8326 | 0.613307 | 0.147452 | 1.04777e-08 | 5.09252e-08 |
| Z_Ea_Au_on | 60.6438 | 0.446962 | 0.070421 | 0.221742 | 0.331194 |
| Z_Eb_Au_on | 186.6095 | 1.315716 | 0.214447 | 0.486825 | 0.681091 |
| Z_Ea_Au_off | 72.6329 | 0.694147 | 0.084458 | -7.01997e-08 | 7.38679e-08 |
| Z_Eb_Au_off | 219.8617 | 2.017318 | 0.253340 | -1.22497e-08 | 2.48554e-08 |

The Au-off cases are laterally symmetric and therefore their signed terminal currents are near-null even though local J is nonzero. The asymmetric T/Z top Au breaks cancellation.

## Absorbed-power location in Au-on cases

| case | TaIrTe4 | top Au | Au mirror |
|---|---:|---:|---:|
| T_Ea_Au_on | 99.366% | 0.111% | 0.523% |
| T_Eb_Au_on | 95.513% | 0.258% | 4.229% |
| Z_Ea_Au_on | 99.269% | 0.592% | 0.139% |
| Z_Eb_Au_on | 98.512% | 0.417% | 1.071% |

Direct top-Au absorption is small in total power. Au can still have a large electrical effect because a highly conducting floating metal redistributes the weighting field/current collection.

## Exact Au contribution to current

The following four terms telescope exactly from the full Au-on current to the independent Au-off current: floating-Au electrical shunt, direct top-Au heating, top-Au thermal shunt, and Au-induced optical redistribution in non-Au materials.

| case/orientation | electrical (nA) | direct heat (nA) | thermal (nA) | optical redistribution (nA) | total on-off (nA) |
|---|---:|---:|---:|---:|---:|
| T_Ea_Au_on / top_bottom | -3.07376e-06 | 0.000844958 | -0.000413228 | 0.00235346 | 0.00278211 |
| T_Ea_Au_on / left_right | -9.27233e-09 | 9.51755e-09 | 2.54579e-10 | 2.08814e-08 | 2.13812e-08 |
| T_Eb_Au_on / top_bottom | 0.00434704 | 0.0021795 | 0.00068647 | -0.0147538 | -0.00754082 |
| T_Eb_Au_on / left_right | 1.83026e-08 | -1.85151e-08 | -2.33299e-10 | 1.02232e-06 | 1.02187e-06 |
| Z_Ea_Au_on / top_bottom | 0.242356 | 0.000789117 | -0.000160145 | -0.0212438 | 0.221742 |
| Z_Ea_Au_on / left_right | 0.357249 | 0.000774729 | 0.00526397 | -0.0320935 | 0.331194 |
| Z_Eb_Au_on / top_bottom | 0.45168 | 0.00144921 | 0.00816429 | 0.0255315 | 0.486825 |
| Z_Eb_Au_on / left_right | 0.818491 | 0.00174724 | 0.0164212 | -0.155569 | 0.681091 |

For Z, the floating-Au electrical term is the dominant positive contribution in this contact scenario, while Au-induced optical redistribution partly cancels it in several cases. For T, the absolute current is pA-scale and the optical/direct-heating terms are comparable or competing.

## Limits

This is a validated forward numerical scenario, not yet an experimental prediction. Au/TaIrTe4 thermal and electrical contact values are not measured for this device; varying them is the next physical-uncertainty gate. The Au Seebeck coefficient is zero in this collection/shunting control.
