# Fixed-K thermal/PTE physical-density AD–FD report

**Status: `VALIDATED_FIXED_K_THERMAL_PTE_PHYSICAL_DENSITY_ADFD`**

This is the first certificate that connects the actual periodic
inverse-design Maxwell field to the fixed-K thermal/PTE objective and compares
the resulting physical-density gradient with an independent v261 central
finite difference.

## Complete discrete chain

Forward:

`rho_geom -> rho_solver -> epsilon_Yee -> E_native`

`E_native -> Q_native -> C -> Q_common -> R_ot -> Q_thermal`

`K_T theta = M_V Q_thermal`

`F_local = c_T^T theta`.

Reverse:

`K_T^T lambda_T = c_T`

`w_native = V_native^-1 C^T R_ot^T M_V^T lambda_T`

`q_E,c = (omega epsilon0 / 2 I_inc)
         Im(epsilon_c) V_native,c w_native,c E_c`

`q_E -> FieldRegion adjoints -> dF/d epsilon_Yee`

`-> measured 27-color transpose -> dF/d rho_solver`

`-> (1-2 delta) -> dF/d rho_geom`.

The division by `V_native` and the later multiplication by `V_native` are the
coefficient-to-quadrature-weight conversion. They cancel algebraically; the
native optical volume is owned exactly once.

## Solver and grid metadata

- Lumerical v261: `8.35.4522`.
- Geometry: actual 6 um-periodic inverse-design cell; no disk.
- Physical design grid: `241 x 241 x 13`.
- Latent grid defined by the repository: `240 x 240` at 25 nm spacing.
- Optical mesh: auto non-uniform, conformal variant 1, accuracy 5.
- TaIrTe4 z mesh: 5 nm.
- Thermal grid: `24 x 24 x 8`; 250 nm x/y, nonuniform z.
- TaIrTe4 kappa: `diag(14.4,3.8,1.0) W/(m K)`.
- Interfaces: `G_top=7.37e6`, `G_bottom=1.1e9 W/(m2 K)`.
- Thermal x/y: periodic; bottom `DeltaT=0`; top adiabatic.
- Readout: finite local numerical mask; units `A m`, not terminal current.

## AD–FD result

The baseline objective is

`F_local/I_inc = -2.942908016968589e-21 A m/(W/m2)`.

For the uniform physical-density direction and `h=0.0025`:

- adjoint directional derivative:
  `3.6997999575952875e-20`;
- central FD:
  `3.7409704548466367e-20`;
- plus objective:
  `-2.850974907623474e-21`;
- minus objective:
  `-3.038023430365806e-21`;
- relative error:
  `1.1005298691415874%`.

The 5% gate passes.

Additional gates:

- periodic source pairing: `1.048115071343203e-14 < 1e-13`;
- live Ex/Ey profile round trip: exact zero;
- 27-color owner leakage: exact zero;
- gradient finite and nonzero.

The baseline and plus/minus forward FSPs were reused. Only the two
thermal-weighted Ex/Ey adjoint solves were new.

## Exact scope

Validated:

- fixed thermal operator;
- named interface-G numerical scenario;
- finite-local-mask PTE functional;
- conservative native/common/thermal source maps and all transposes;
- TaIrTe4 absorption Wirtinger source;
- FieldRegion Maxwell adjoints;
- solver-realized conformal epsilon transpose;
- solver-safe affine chain to `rho_geom`.

Still pending:

- latent `240 x 240` filter/projection/fencepost end-to-end Maxwell FD;
- a physical electrode or electrical weighting-potential functional;
- thermal properties and interface laws for the optical design material;
- the matrix derivative
  `-lambda_T^T (dK_T/d rho) theta`.

Therefore this is not yet a full fabrication-material inverse-design
gradient and not a terminal photocurrent prediction.
