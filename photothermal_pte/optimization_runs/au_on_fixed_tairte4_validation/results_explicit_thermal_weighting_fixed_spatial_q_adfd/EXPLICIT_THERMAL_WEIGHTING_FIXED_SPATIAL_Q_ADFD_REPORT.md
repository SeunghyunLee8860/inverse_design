# Explicit thermal/weighting fixed-spatial-Q AD--FD

Status: **VALIDATED_EXPLICIT_THERMAL_WEIGHTING_FIXED_SPATIAL_Q_ADFD**

The certified spatial Maxwell source is held fixed while the same 20x20 Au
density changes the explicit 3-D Au conductivity, parallel-area Au/TaIrTe4
thermal contact, floating-Au electrical conductivity, and finite vertical
electrical contact. The 40x40 electrical temperature pullback is transposed
through the 500-nm-to-100-nm and thickness-averaging maps before the thermal
adjoint solve.

The independent thermal-matrix directional derivative audit has worst error
`0.000056%`. Across thermally-grown and evaporated
interface scenarios, the finest-step worst strong-direction error is
`0.000023%` and the worst gradient-L2-normalized error is
`0.000006%`. Maximum linear residual is
`8.261e-10`; thermal energy balance is `0.000000%`.

This certificate contains no Maxwell optical derivative. The next gate is the
native-Yee spatial-Q weighted optical adjoint, followed by a full combined
directional AD--FD. No optimization is authorized yet.
