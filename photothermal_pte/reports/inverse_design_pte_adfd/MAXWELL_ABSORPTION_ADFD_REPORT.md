# Maxwell absorption physical-density AD–FD report

**Status: `VALIDATED_MAXWELL_ABSORPTION_PHYSICAL_DENSITY_ADFD`**

This certificate uses the actual 6 um-periodic inverse-design geometry, not a
disk proxy. It validates a frozen native-Yee absorption weight. It does not
yet claim the full thermal-adjoint or latent-variable chain.

## Forward objective and adjoint source

For component `c` on its native shifted Yee coordinates,

`Q_c = (omega epsilon0 / 2 I_inc) Im(epsilon_c) |E_c|^2`.

For a frozen density weight `w_c` and shifted-coordinate trapezoid volume
`V_c`,

`F = sum_c sum_i V_ci w_ci Q_ci`

and the literal Wirtinger source is

`q_E,c = dF/dE_c*
       = (omega epsilon0 / 2 I_inc)
         Im(epsilon_c) V_c w_c E_c`.

The periodic source right-inverse folds duplicate x/y seam coefficients into
the active FieldRegion copy. The measured source pairing error was
`8.623640797858466e-16`.

TaIrTe4 has fitted v261 loss
`Im(epsilon)=[50.85010970213534, 9.289194655416972, 0]`; therefore Ex and Ey
adjoints were run and the Ez adjoint was exactly absent.

## Epsilon and density chain

Each adjoint field is paired with the forward design field using the existing
component-correct design-monitor Yee volumes. The conformal material
derivative is not an analytic effective-medium guess: it is the 27-color,
centered, solver-measured `rho_solver -> epsilon_Yee` transpose. Owner leakage
was exactly zero.

Production density uses

`rho_solver = delta + (1-2 delta) rho_geom`,

with `rho_step=0.0025`, `delta=0.002501`, and chain factor `0.994998`.
Consequently

`dF/d rho_geom = (1-2 delta) dF/d rho_solver`.

The live FieldRegion profile round trips were exact zero before both adjoint
solves. Reopening the saved Ex FSP introduced only
`4.577566798522237e-16` relative serialization error; Ey remained exact.

## Directional finite differences

| direction | h | adjoint | central FD | relative error | result |
|---|---:|---:|---:|---:|---|
| uniform | 0.0025 | -4.716716167139546e-12 | -4.833696135289278e-12 | 2.4200935448899785% | PASS |
| oscillatory/cancellation-prone | 0.02 | -7.750926106180087e-16 | -5.838636193698367e-16 | 24.671760332703768% | FAIL/noise-floor diagnostic |

The uniform direction is the certificate direction and passes the predeclared
5% gate. The oscillatory direction is retained rather than discarded. Its
directional derivative is about 6000 times smaller than the uniform
derivative, and its plus/minus objective difference is only
`2.3354544774e-17 m2`; it is not a reliable finite-difference signal at this
solver tolerance.

## Claims and remaining blockers

Validated here:

- native shifted-Yee absorption functional and transpose;
- FieldRegion Ex/Ey volume-current sources;
- periodic endpoint folding;
- solver-measured conformal `rho -> epsilon` transpose;
- solver-safe affine density chain;
- one robust physical-density central-FD direction.

Not yet validated:

- exact pullback of the thermal adjoint onto native optical samples;
- latent filter/projection/fencepost end-to-end Maxwell FD;
- `rho`-dependent thermal conductivity or interface conductance;
- physical electrode/weighting-potential terminal current.

The last two retain
`BLOCKED_FULL_RHO_DEPENDENT_THERMAL_MATERIAL_MODEL` and
`BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`.
