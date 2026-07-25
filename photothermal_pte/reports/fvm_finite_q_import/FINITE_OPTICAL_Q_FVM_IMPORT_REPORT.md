# Finite optical-Q FVM import report

**Status: `VALIDATED_FINITE_OPTICAL_Q_FVM_IMPORT`.**

The PR #3 artifact SHA-256 is exactly
`7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`. Its array order is `x,y,z`,
shape is `[80, 80, 41]`, and incident-intensity normalization is
`1.0 W/m2`.

No thermal solve or production full-device calculation was run in this gate.

## Mapping

Every original `Q[i,j,k]` value is copied element-for-element to one FVM
source control volume. The cell widths are exactly the original 1D
trapezoidal quadrature weights. There is no interpolation and no source-value
change.

The original and mapped Q-array SHA-256 values are both
`ff1484537aadfc36d90c2035280da9ad3a2e59895e9ba06a65bea30623e3715d`.

| Power check | W |
|---|---:|
| expected PR #3 P_Q | 2.56071371086521e-12 |
| original nested trapezoidal integration | 2.56071371086521e-12 |
| FVM sum(Q*dV) | 2.56071371086521e-12 |

The FVM mapping relative error is
`0`, below the required 0.5%.
The algebraic quadrature-equivalence error is
`0`.

## Prohibited operations

Clipping, smoothing, gain, total-Q rescaling, periodic crop/tiling, and
outside-flake deletion are all `false`. There are
`5772`
nonzero samples excluded by the stored boolean mask only because their
`z=5.79026e-23 m` coordinate is infinitesimally above zero.
They remain inside the explicit `1e-15 m` roundoff-inclusive physical mask
used by PR #3, and the mapper preserves them without deletion or alteration.
Q is exactly zero outside that physical mask.

## Gate

The finite optical-Q conservative import gate passes. The next permitted
step is the first anisotropic, finite-G, multi-material FVM thermal solve.
Its result must be reported as a unit-intensity response
`Delta T / I_inc [K/(W/m2)]`, not as a physical laser temperature.
