# Paper-like IR material, source, and edge controls

**Status: `PARTIAL_PAPER_IR_CONTROL_VALIDATION_BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC`**

## What is validated

The offline analytic thermal controls pass.  At 50 nm, the paper-like
absorbed-power control gives
`max|dT/dx|_b/max|dT/dx|_a =
1.446954`.
After forcing the two analytic sources to the same absorbed power, the ratio
is
`0.967078`.
The exact-identical-Q symmetry control gives `1.000000` with zero field
difference.  Thus the original b>a thermal trend is predominantly the
pre-supplied polarization-dependent TMM absorbed power, not an independent
Lumerical discovery.

The production 3D material contract is now `epsilon_c(lambda)=epsilon_b(lambda)`.
It is the explicit paper-consistent closure used to extend the reported
in-plane epsilon_a/epsilon_b data to finite-edge 3D Maxwell; it is not called
a direct c-axis measurement.  The legacy lossless `epsilon_c=16` model remains
diagnostic only.

At 11 µm, fitted epsilon_z and epsilon_x differ by
`0.000e+00` and the finite-dt values
differ by `0.000e+00`.  The
legacy artifact has integrated `Qz=0.000000e+00 W`; production Qz is
`not available W`.

## What remains unresolved

The robust physical-line fit uses exact
`n=(-x+y)/sqrt(2), t=(x+y)/sqrt(2)` coordinates and treats raw cell maxima as
diagnostic only.  Analytic-source fitted dx strip mean changes by
`0.610%` from 100 to 50 nm, but the
legacy Maxwell-Q b-polarization value changes by
`12.851%`.  Fit-band sensitivity also
exceeds 10%.  Therefore no 50 or 100 nm edge-gradient mesh is promoted.
The next numerical method candidate is a conservative exact-half-plane
cut-cell treatment, not ad-hoc local cells.

The v261 contract-only session and material fit succeeded.  3 attempts stopped before timestepping because the requested `lum_fdtd_solve` task count was unavailable.  A later GPU-only attempt acquired the licenses, meshed `1461 x 1461 x 161` gridpoints, and started timestepping, but the Lumerical API/engine communication failed after the log reached `3.3357%`.  The incomplete HDF5 output is provenance only and is not treated as a recoverable optical result.  No CPU FDTD fallback was used.  Production Qx/Qy/Qz, P_Q, closure, native Yee coordinates, and the edge-normal Q profile therefore remain blocked.

The scalar/thin-lens comparison remains plan-only.  See
`SCALAR_VECTOR_GAUSSIAN_MATCH_PLAN.md`.

No PTE-current solve, adjoint, gradient, or optimization was run.
