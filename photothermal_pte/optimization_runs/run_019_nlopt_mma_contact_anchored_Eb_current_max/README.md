# Run019 — NLopt LD_MMA, E parallel b

This run starts only after Run018 completes.  It uses the same exact uniform
`rho=0.5` start, geometry, constraints and NLopt `LD_MMA` contract, with only
the source polarization changed to `E||b` (`Lumerical x=b`, `y=a`, `z=c`).

- Manual move limit: none
- Adam or normalized-gradient update: none
- Raw FSP/NPZ artifacts are not committed
- Final exact binary design is evaluated by a fresh GPU-Maxwell/CUDA-thermal/
  electrical solve
