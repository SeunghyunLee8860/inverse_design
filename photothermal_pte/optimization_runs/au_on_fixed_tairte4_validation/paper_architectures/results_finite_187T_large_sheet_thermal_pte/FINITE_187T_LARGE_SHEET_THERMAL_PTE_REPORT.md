# Finite 187-T large-sheet thermal/PTE diagnostic

Status: `VALIDATED_LARGE_SHEET_DIAGNOSTIC_THERMAL_WEIGHTING_PTE`

## Scope

This is a **large finite computational TaIrTe4 sheet with ideal full-width y-edge contacts**.  It retains the entire validated component-specific Yee heat source and is not an experimental finite-contact prediction.  Axis mapping is `x=b, y=a, z=c`.

The optical-closure stack is air / finite 187 Au inverse-Ts / TaIrTe4 100 nm / Al2O3 35 nm / Au mirror 200 nm / SiO2 285 nm / Si.  Lateral and top thermal faces are adiabatic; the Si bottom is fixed at DeltaT=0.  SiO2/Si uses G=1.1e9 W/(m2 K); the other interfaces are explicitly perfect-contact diagnostic assumptions.

## Certified results at 285 uW incident power

- mapped absorbed power: `2.094231629e-05 W`
- Q mapping error: `0.000e+00`
- flake Tmax: `3.109227236e-02 K`
- flake area-average DeltaT: `2.861487252e-03 K`
- max strict-centered |grad T|: `3.006797000e+04 K/m`
- short-circuit terminal current: `-1.169299482e-12 A`
- open-circuit voltage: `2.381465341e-11 V`
- CUDA solve: `3.961 s`, `3400 iterations`
- residual: `9.433e-11`
- energy-balance error: `4.583e-13`

The terminal value is small because positive and negative current-integrand regions largely cancel under the symmetric ideal-contact diagnostic. It must not be interpreted as the finite experimental device current.

## Fields

- [all optical/thermal/electrical fields](finite_187T_large_sheet_all_fields.png)
- [Q and temperature cross-sections](finite_187T_Q_temperature_cross_sections.png)
- [signed central profiles](finite_187T_signed_central_profiles.png)
- [summary JSON](finite_187T_large_sheet_summary.json)
- [cases CSV](finite_187T_large_sheet_cases.csv)
- [artifact manifest](RAW_ARTIFACT_MANIFEST.json)

No Q clipping, smoothing, gain, global shape rescaling, or source deletion was used. The only scaling is certified linear scaling from the source-only incident power to 285 uW.
