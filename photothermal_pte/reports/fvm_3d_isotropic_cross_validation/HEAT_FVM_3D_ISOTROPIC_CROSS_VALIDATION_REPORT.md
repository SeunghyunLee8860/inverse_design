# 3D isotropic HEAT-FVM cross-validation

**Status: `VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION`.**

This control uses the same 3D two-material Cartesian geometry, scalar
conductivities, perfect contact, asymmetric synthetic volumetric Q, bottom
300 K boundary, and adiabatic remaining exterior boundaries in v261
Lumerical HEAT and the independent Python/SciPy FVM.

It does not use the finite optical-Q artifact or the production device.

| Metric | Result | Gate |
|---|---:|---:|
| Tmax difference / max FVM rise | 0.226837% | <1% |
| mean-T difference / max FVM rise | 0.0440171% | <1% |
| 3D field NRMSE / max FVM rise | 0.107756% | <1% |
| 3D field correlation | 0.999983190756 | >0.999 |
| source-power cross error | 0.400326% | <1% |
| boundary-power cross error | 0.400034% | <1% |
| Lumerical energy error | 0.000290729% | <1% |
| FVM energy error | 1.76324e-09% | <1% |

The v261 unstructured temperature field is linearly interpolated to all
`48000` FVM cell centers on the 50 nm common grid. There are no
NaN/Inf samples. Material values are exactly `[2.0, 10.0]`
W/(m K) in both solvers.

Non-gating diagnostics are retained rather than hidden: the 99th-percentile
pointwise field error is
`0.45207%` and the
single worst source-edge point is
`1.05266%`. The global
field NRMSE and correlation are the declared field gates.

The 3D common-physics gate passes. The next step is only the conservative
finite optical-Q mapping/reintegration gate; anisotropic and finite-G
production physics must still wait until that import passes.
