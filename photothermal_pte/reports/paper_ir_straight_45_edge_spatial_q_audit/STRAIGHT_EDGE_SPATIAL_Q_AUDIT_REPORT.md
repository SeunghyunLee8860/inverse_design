# Straight-edge spatial-Q, remap, and gradient audit

**Status: `UNRESOLVED_STRAIGHT_EDGE_OPTICAL_AND_THERMAL_SPATIAL_CONVERGENCE`**

This checkpoint reuses the saved GPU Lumerical artifacts; no new FDTD, AD/FD,
or optimization run was performed.  The straight-edge subgate intentionally
has no weighting/PTE evaluation.  A separate saved-Q Device-A calculation
was rerun only to quantify the corrected local-contact weighting operator.

## What was wrong and what was corrected

The old `x/y/z/x` support projection was coordinate-order dependent.  The
named symmetric Gaussian regression reproduces a 50.0% relative L1 difference
between `x/y/z/x` and `y/x/z/y` while preserving identical total power.  The
straight-edge path now uses one physical-3D nearest-support operator and
splits exact distance ties uniformly.  Power, transpose, and reflection
symmetry tests pass.

The 50.0% value is a structural synthetic regression, not an estimate of the
saved Maxwell-Q error.  On the actual raw Q, the two historical axis orders
differ by
0.002586%
for a polarization and
0.002235%
for b.  Historical-to-new physical-nearest differences are
0.019838% and
0.004467%.
Thus the old operator is invalid in principle, but its actual contribution
does not explain the observed ~20% gradient-order reversal.

The reported area/volume averages now use literal cell area/volume.  The
weighting contact uses each contacted cell's local half width.  Five separate
gradient observables are retained.  The paper comparator is
`max_abs_grad_T_x_K_m`; edge-normal gradient is not substituted for it.

The legacy Device-A expanded currents change from
24.047863/
20.218345 nA
to
24.654870/
20.743573 nA
for a/b polarization.  The corrected ratio is
1.188555.
The exact old values remain provenance diagnostics, not silently overwritten.

## Separated sanity checks

With analytic Gaussian–Beer–Lambert Q and the paper Eq. S4 reduced Robin
model, the `max|dT/dx|` ratio is
**1.446770** at
100 nm and
**1.446954 (b/a)** at
50 nm.
The expected b>a order is reproduced.

With the saved finite-edge Lumerical Q on that same reduced thermal operator,
the ratio is
**0.805447**
at 100 nm and
**0.881330**
at 50 nm.
With the expanded production FVM it is
**0.817054** at 80 µm.
The paper comparator and four of five 50-nm gradient ratios remain below
one; `max|dT/dy|` is numerically near-null at
1.000035.
Thus the apparent inversion is source-spatial-distribution sensitive, but
the unconverged peak-gradient estimator prevents promotion as a physical
reversal. It is not explained solely by choosing edge-normal rather than
x-gradient.

The worst 100-to-50 nm change among the five paper-reduced gradient
observables is **67.861930%**.  The
predeclared 1% thermal-gradient mesh gate is
**not passed**.

## Thermal-domain audit

The far x/y and bottom Dirichlet powers are numerical truncation-boundary
fluxes, not intrinsic physical heat-path fractions.  Although their share
changes strongly with domain size, the central edge metrics converge much
more tightly.  Relative changes to the 80 µm case are stored explicitly in
the summary JSON.  Whole-half-plane averages are not used for this comparison;
the report uses the fixed |x|,|y| <= 12 µm ROI.

## Remaining blockers

- Solver-native lateral Yee-mesh readback/refinement is not certified.
- Fitted sampled-material epsilon readback is absent.
- epsilon_c=16+0i forces Qz=0 and is not validated for edge scattering.
- Scalar- versus vector-Gaussian edge-Q sensitivity is absent.
- Paper-reduced edge-gradient 100-to-50 nm convergence exceeds 1%.

The common absorption artifact has 33.9703 nm x/y and 10 nm z spacing, but
that is not relabeled as native Yee-mesh readback.  The remote polygon faces
are at x=+25 µm and y=-25 µm, beyond the actual +/-24 µm FDTD outer boundary;
the flake therefore does not terminate inside the lateral PML in these saved
artifacts.

Published figures are `STRAIGHT_EDGE_AUDIT_METRICS.png`,
`STRAIGHT_EDGE_THERMAL_MESH_CONVERGENCE.png`, and
`STRAIGHT_EDGE_Q_PROFILES.png`.
