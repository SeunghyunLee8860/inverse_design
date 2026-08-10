# Run 007 corrected E||b contract

- Solver coordinates: `x=b, y=a, z=c`.
- Optical source: `0 deg = E||b`.
- TaIrTe4 kappa xyz: `(3.8, 14.4, 1.0) W/(m K)`.
- Electrical coefficients xy: `(sigma_b,S_b)`, `(sigma_a,S_a)`.
- Objective: `-I_b/P_incident`, with raw signed `I_b` retained.
- 500 nm solid and void minimum-feature constraints; full beta continuation to exact binary.
- No CPU Maxwell/thermal fallback, clipping, repair, or empirical gradient scaling.
