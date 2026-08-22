# Reconstructed Z2022 M2 selected volumetric Q

Status: `LEGACY_WRONG_AXIS_Z2022_M2_DIAGNOSTIC`

> **Correction notice.** This archived result interchanged the Fig. 1b
> `W1/W2` and `L1/L2` directions. It is retained only for numerical
> provenance and must not be shown as the paper Z geometry. The corrected
> contract uses `W1/W2` along the short-period axis and `L1/L2` along the
> long-period axis, and requires a paired `E||a`/`E||b` calculation.

This is a real v261 GPU Maxwell result for the **legacy axis-swapped corner-joined reconstruction**. It is not author CAD and is not the corrected Fig. 1b geometry. The active 2-D layer is replaced by fixed 100-nm anisotropic TaIrTe4 (`x=b, y=a, z=c=b closure`).

At 5.25 um, LH CP+ gives `P_Q=6.058561095e-15 W/cell` and LH CP- gives `P_Q=6.089416296e-15 W/cell`, hence `g=-0.005080`. Closures are `0.3228%` and `0.3213%`; both auto-shutoff and Q gates pass. The component-specific conservative common-grid powers differ from the periodic-`pabs` totals by `0.3189%` and `0.3158%`. Equal-power spatial-Q NRMSE is `62.5653%` and correlation is `0.53167367`.

CP+ and CP- retain explicit solver phase definitions and are not silently renamed LCP/RCP. No thermal/PTE result is claimed for the periodic unit cell: finite Gaussian illumination and finite electrical contacts must be defined first.

- [Q comparison](Z2022_M2_selected_Q_CP_comparison.png)
- [summary JSON](Z2022_M2_SELECTED_Q_SUMMARY.json)
- [cases CSV](Z2022_M2_selected_Q_cases.csv)
- [raw manifest](RAW_ARTIFACT_MANIFEST.json)
