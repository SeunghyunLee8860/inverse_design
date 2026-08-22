# Reconstructed Z2022 M2 selected volumetric Q

Status: `VALIDATED_Z2022_M2_RECONSTRUCTED_SELECTED_Q_PAIR`

This is a real v261 GPU Maxwell result for the **explicit corner-joined reconstruction** of the published M2 scalar dimensions. It is not author CAD. The active 2-D layer is replaced by fixed 100-nm anisotropic TaIrTe4 (`x=b, y=a, z=c=b closure`).

At 5.25 um, LH CP+ gives `P_Q=6.058561095e-15 W/cell` and LH CP- gives `P_Q=6.089416296e-15 W/cell`, hence `g=-0.005080`. Closures are `0.3228%` and `0.3213%`; both auto-shutoff and Q gates pass. The component-specific conservative common-grid powers differ from the periodic-`pabs` totals by `0.3189%` and `0.3158%`. Equal-power spatial-Q NRMSE is `62.5653%` and correlation is `0.53167367`.

CP+ and CP- retain explicit solver phase definitions and are not silently renamed LCP/RCP. No thermal/PTE result is claimed for the periodic unit cell: finite Gaussian illumination and finite electrical contacts must be defined first.

- [Q comparison](Z2022_M2_selected_Q_CP_comparison.png)
- [summary JSON](Z2022_M2_SELECTED_Q_SUMMARY.json)
- [cases CSV](Z2022_M2_selected_Q_cases.csv)
- [raw manifest](RAW_ARTIFACT_MANIFEST.json)
