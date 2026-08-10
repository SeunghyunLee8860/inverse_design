# Run018 — NLopt LD_MMA, E parallel a

Fresh contact-anchored TaIrTe4 topology optimization from exact uniform
physical density `rho=0.5`.  Lumerical coordinates are `x=b`, `y=a`, `z=c`.

- Optimizer: NLopt 2.11 `LD_MMA`
- Manual move limit: none
- Adam or normalized-gradient update: none
- Objective: maximize signed full-flake PTE current under `E||a`
- Continuation: beta `1,2,4,8,16,32,64,128`
- 500 nm differentiable solid/void inequalities begin at beta 8
- Raw FSP/NPZ artifacts remain under `/data` and are not committed

Every JSON/PNG is an NLopt full-physics function evaluation, not a falsely
labelled accepted custom update.  Final validation requires a separately
re-solved exact binary structure.
