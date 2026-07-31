# W12 explicit-3D 100-to-50 nm thermal-gradient convergence

Status: `PARTIAL_W12_EXPLICIT3D_TEMPERATURE_CONVERGED_GRADIENT_MAGNITUDE_UNCONVERGED`

The same native GPU Maxwell Q artifacts, analytic sources, 60-µm explicit-3D
thermal geometry, materials, interface G values, boundary conditions, and
10-nm flake z cells were used.  Only the 24-µm core x/y thermal step changed
from 100 to 50 nm.

## Solver and scalar temperature

All four 50-nm cases passed existing source mapping, residual, and energy
gates.  The largest per-case Tmax change is
`0.852892%`;
the temperature gate `<1%` therefore passes.  The fine grid has
`[530, 530, 86]` cells and required
`3400`–
`3497` CG
iterations.

## Original staircase-edge Maxwell b/a

| metric | 100 nm | 50 nm | relative change |
|---|---:|---:|---:|
| FD raw max | 0.879613 | 0.914024 | 3.765% |
| FD p99 | 0.879578 | 0.913987 | 3.765% |
| FD RMS | 0.886038 | 0.915278 | 3.195% |
| FD mean | 0.890100 | 0.912302 | 2.434% |

The raw/p99/RMS/mean values remain below one at both meshes, so the Maxwell
ordering `b<a` is not a single-cell artifact.  However, the changes exceed
1%, so the numerical value `0.879613` is not mesh converged.

## Least-squares sensitivity

| LS physical radius | 100-nm raw-max b/a | 50-nm raw-max b/a | relative change |
|---|---:|---:|---:|
| 0.2 µm | 0.783776 | 0.700540 | 11.882% |
| 0.3 µm | 0.662673 | 0.578599 | 14.531% |
| 0.4 µm | 0.582121 | 0.490422 | 18.698% |

All tested Maxwell combinations—two meshes, FD/three LS radii, raw/p99/RMS/
mean, the original edge and fixed 0.1–0.3 µm inside band—remain `b<a`.
All corresponding analytic controls remain `b>a`, with a maximum ratio
change of `0.238%`.

## Decision

The reversal direction is robust and is therefore consistent with the
spatial Maxwell Q distribution.  Its exact magnitude is unresolved:
`0.879613` is a diagnostic 100-nm FD value, not a final experiment-prediction
ratio.  A further quantitative certificate would require a predeclared
physical gradient functional and another mesh level or a higher-order
cut-cell/finite-element edge treatment.

Raw artifacts were not modified.  Exact paths, sizes, and SHA-256 values are
recorded in the summary JSON.
