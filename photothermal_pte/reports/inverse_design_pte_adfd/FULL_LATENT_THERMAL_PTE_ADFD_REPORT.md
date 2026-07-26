# Full fixed-K latent thermal/PTE AD–FD report

**Status: `VALIDATED_FIXED_K_LATENT_THERMAL_PTE_ADFD`**

The actual periodic inverse-design chain, not either disk proxy, passed a
v261 central finite-difference check. This closes the fixed-thermal-operator
optical chain through the production latent variable. It does not close the
unknown thermal-material model of the optical design material or convert the
local PTE surrogate into terminal current.

## Forward model frozen for this certificate

### Inverse-design and optical grids

- latent: `240 x 240`, endpoint-free periodic torus;
- latent spacing: `25 nm x 25 nm`;
- mapping: periodic conic filter of radius `0.5 um`, nominal tanh projection,
  exact periodic fencepost, 13-layer z extrusion;
- physical density: `241 x 241 x 13`;
- physical design spacing: `25 nm x 25 nm x 50 nm`;
- design volume: `6 x 6 x 0.6 um`;
- FDTD grid realized by v261: `243 x 243 x 173`;
- optical x/y: periodic; optical z: PML;
- optical mesh: auto non-uniform, conformal variant 1, accuracy 5;
- TaIrTe4 z override: `5 nm`;
- source: 3–6 um broadband, x polarization;
- analysis wavelength: 4 um;
- incident intensity:
  `1.3272093648958553e-3 W/m2` for the solver source amplitude;
- Lumerical: v261 `8.35.4522`;
- actual resource: `GPU 0`, NVIDIA RTX 6000 Ada, 16 threads.

The two disk geometries were not loaded or differentiated.

### Thermal grid, materials, interfaces, and boundaries

The frozen native-Yee weight came from the independently checked
`24 x 24 x 8` conservative Cartesian FVM:

- lateral thermal cell size: `250 nm`;
- thermal x/y: periodic;
- bottom Si: fixed `Delta T=0`;
- top exposed face: adiabatic;
- TaIrTe4:
  `kappa=diag(14.4,3.8,1.0) W/(m K)`;
- SiO2: `1.38 W/(m K)`;
- Si: `145 W/(m K)`;
- TaIrTe4/top-interface numerical scenario:
  `G_top=7.37e6 W/(m2 K)`;
- SiO2/Si numerical scenario:
  `G_bottom=1.1e9 W/(m2 K)`.

These `G` and `kz` values are named numerical scenarios, not unique measured
truth. Side periodic faces, internal finite-G faces, the top physical exposed
face, and the bottom truncation reservoir are separate face classes.

The optical high-index endpoint has only `n=4` in the repository. It has no
declared fabrication identity, kappa, interface G, or differentiable thermal
mixing law. Consequently the thermal matrix was held fixed:

`BLOCKED_FULL_RHO_DEPENDENT_THERMAL_MATERIAL_MODEL`.

### PTE readout

The objective is the declared finite local numerical readout

`F_local = c_T^T theta`

with `a=x`, `b=y`,
`sigma_a=4.91e5 S/m`, `sigma_b=1.10e5 S/m`,
`S_a=-6e-6 V/K`, and `S_b=27e-6 V/K`.

Its unit is `A m`, so the normalized objective is reported as
`A m/(W/m2)`. It is not terminal current. A physical finite flake/contact
model or an electrical weighting potential is still required:

`BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`.

## Exact adjoint and chain rule

The forward thermal problem is

`K_T theta = M_V R_ot C Q_native(E) + b`.

For the fixed operator:

`K_T^T lambda_T = c_T`.

The optical absorption covector is returned by literal transposes:

`w_common = R_ot^T M_V^T lambda_T`

`a_native = C^T w_common`

`w_native = a_native / V_native`.

The last division converts an integrated native coefficient to a density
weight. The native optical quadrature later multiplies by `V_native` exactly
once.

For component `c`, the complex-field Wirtinger source is

`q_E,c = (omega epsilon_0 / (2 I_inc))`

`        Im(epsilon_c) V_native,c w_native,c E_c`.

The source is folded onto the active periodic FieldRegion representation.
Two independent Ex/Ey volume-current adjoints are solved. Their design-field
sensitivities are passed through the measured 27-color conformal
epsilon-to-density transpose, then through

`rho_solver = delta + (1-2 delta) rho_geom`

with `delta=0.002501`, and finally through

`J_map^T = J_filter^T J_projection^T`

including exact accumulation of the duplicated x/y fenceposts and all 13
extruded z layers.

The explicit-loss term is zero in this certificate because the design
material is above the fixed TaIrTe4 loss region. If thermal properties later
depend on density, the missing contribution must be added before the mapping
VJP:

`-lambda_T^T (dK_T/drho) theta + lambda_T^T db/drho`.

## v261 AD–FD result

The test used:

- beta: `8`;
- structured latent baseline;
- direction: uniform latent perturbation;
- central-FD step: `h=0.005`;
- identical probe-safe affine in baseline, plus, minus, and AD.

Results:

- baseline:
  `1.3070866221153408e-20 A m/(W/m2)`;
- plus:
  `1.565664622317477e-20 A m/(W/m2)`;
- minus:
  `1.0389691598489054e-20 A m/(W/m2)`;
- AD directional derivative:
  `5.376561779173102e-19`;
- central FD:
  `5.266954624685716e-19`;
- relative error:
  `2.0386105282369334%`;
- gate:
  `2.0386% < 5%`, pass.

Additional exact or bounded checks:

- periodic source pairing: `2.9929828170064076e-15 < 1e-13`;
- live Ex/Ey source import roundtrip: exactly zero;
- measured epsilon-transpose owner leakage: exactly zero;
- x/y fencepost errors: exactly zero;
- z-extrusion error: exactly zero;
- physical and latent gradients: finite and nonzero.

## Raw storage and unit-label erratum

Large FSP/NPZ files remain outside Git under
`/home/seunghyun/tairte4/artifacts/pte_adfd/`. The checked-in manifest records
their absolute paths, byte sizes, SHA-256 values, and generation command.

The immutable generated summary has a stale nested generic-evaluator label:
`metadata.objective_unit="m2 for dimensionless native density weight"`.
That label belongs to the evaluator's default dimensionless diagnostic weight,
not to the thermal/PTE weight used here. The top-level generated field and
this published report have the correct unit:

- objective: `A m/(W/m2)`;
- supplied native density weight: `A m/W`.

The raw generated summary is retained unchanged for provenance. The runner is
patched so future executions replace the generic nested label before writing
their summary.

## Numerical verification versus physical uncertainty

Numerically validated:

- fixed-K thermal solve and transpose;
- conservative optical/thermal remaps and transposes;
- native-Yee absorption and adjoint sources;
- Maxwell adjoints;
- solver-realized epsilon transpose;
- solver-safe density affine;
- production filter/projection/fencepost/extrusion VJP;
- complete fixed-K latent directional derivative.

Not physically identified:

- material represented by optical endpoint `n=4`;
- its kappa and thermal mixing law;
- its internal interface conductance laws;
- confidence interval for TaIrTe4 `kz`;
- terminal electrical weighting field or finite contacts.

Therefore the result is a validated numerical gradient for a named fixed-K
scenario, not a final experimental PTE prediction and not yet authorization
to start optimization.
