# Paper-reduced rho-dependent thermal-material AD–FD

**Status: `VALIDATED_PAPER_REDUCED_THERMAL_MATERIAL_ADFD`**

This is the isolated fixed-Q certificate for the reduced TaIrTe4 surface-boundary model. It is not the older Si/SiO2/TaIrTe4 bulk FVM model and it is not a final terminal-current prediction.

## Contract

- Optical material label: **n=4 optical proxy + paper SiO2 thermal boundary**.
- TaIrTe4 kappa: diag(14.4, 3.8, 1.0) W/(m K).
- Bath temperature: 300 K. The numerically solved unknown is theta=T-300 K.
- Bottom substrate Robin G: 7.37e6 W/(m2 K).
- Top design law: `G=1+rho_bar*(G_SiO2-1)` W/(m2 K).
- No bulk k_air, k_SiO2, k_Si, or G_SiO2/Si is introduced.
- The finite local PTE mask remains a numerical A m surrogate; a weighting-potential/finite-contact solve is still blocked.

## Discrete derivative

`g_i=A_i*G(rho_i)`, `(K_T)_ii += g_i`, and the absolute-temperature load is `b_i += g_i*T_bath`. In the exactly shifted theta system, `b=0` and the same thermal derivative is

`dF/drho_i = -lambda_i*A_i*(G_SiO2-G_air)*theta_i`.

## Scenario results

| Scenario | G_SiO2 (W/m2K) | AD | best FD | rel. error | energy | residual | pass |
|---|---:|---:|---:|---:|---:|---:|---|
| thermally_grown_SiO2_baseline | 7.370000e+06 | -1.603018e-21 | -1.603018e-21 | 1.868866e-08 | 1.837273e-15 | 6.833334e-12 | True |
| evaporated_SiO2_sensitivity | 7.370000e+04 | -3.392358e-23 | -3.392358e-23 | 1.170519e-11 | 3.674547e-16 | 6.627450e-12 | True |

The thermally-grown value is the paper baseline. The evaporated value is a named fabrication sensitivity, not a confidence interval and not a replacement baseline.

## Retained blocker

- `BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`
