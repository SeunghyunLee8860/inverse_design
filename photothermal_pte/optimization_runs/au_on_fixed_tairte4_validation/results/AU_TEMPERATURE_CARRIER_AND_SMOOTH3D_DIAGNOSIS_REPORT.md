# Au temperature-carrier and smooth-3D gradient diagnosis

Status: `BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_NO_STABLE_GPU_DIFFERENTIABLE_AU_PATH`

## Result

The v261 GPU temperature-grid coupling is real, but it does **not** provide a
stable exact-Au topology path.  With conformal variant 1, the moderate control
`n=2, k=0.5` reproduces `epsilon=3.75+2i` on every component grid.  It closes
`P_Q` against the six faces to `0.015910%`
and reaches auto-shutoff `6.235810e-08`.
This numerical attribute is never a physical temperature.

The same mechanism diverges at the exact 10-um Ordal Au endpoint
`n+ik=12.1+69.2i`.  The failure remains for a 50-nm film, forward and reverse
base directions, linear and table models, and carrier spans of 1 K and 1000 K.
The latter reduces the recorded sensitivities to `dn/dT=0.0111` and
`dk/dT=0.0692`, so the failure is not cured by coefficient scaling.  CPU FDTD
fallback was prohibited.

## Boundary root cause

The installed LumOpt kernel is the standard tangential-E/normal-D continuous
shape derivative.  On the fully smooth 3-D ellipsoid, offsets from -100 to
+100 nm were evaluated without changing the surface or fitting to FD.  None
reproduces the independent central-FD sign; the best relative mismatch is
`100.221%`.

The solver-discrete conformal-Yee diagonal epsilon derivative also fails.  At
`h=0.05 um`, its contraction is
`4.443254125834e-28` J-proxy/um,
whereas FD is `-2.665238476050e-30` J-proxy/um.  The
sign is wrong and the discrete derivative changes by
`68.114%` between the
two steps.  This is not a coordinate-pairing error: forward/adjoint mismatch is
`0.000e+00` m.

## Decision

Exact scalar Au remains a valid forward GPU material, and the fixed-geometry
material adjoint remains valid.  What is blocked is a representation that is
both stable at exact Au and differentiable for topology optimization.  No Au
thermal, electrical, PTE, adjoint-chain, or optimization result is promoted.
No clipping, smoothing, empirical normalization, or gradient rescaling was
used.

A future route must expose a causal dispersive Drude/ADE oscillator-strength
Jacobian on the spatial grid, or use another Maxwell backend with a certified
metal topology adjoint.  A per-pixel full-Maxwell finite-difference Jacobian is
not an acceptable production method.

Official v261 implementation inspected:
`/opt/lumerical/v261/api/python/lumopt/utilities/gradients.py`.
Official temperature-grid documentation:
https://optics.ansys.com/hc/en-us/articles/360034901773-Temperature-dependent-refractive-index-models
