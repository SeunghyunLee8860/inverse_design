# Run 016 — fresh true-MMA E||a optimization

This run starts from exact uniform physical density `rho=0.5` in the
contact-anchored TaIrTe4 topology geometry. Lumerical coordinates are
`x=b`, `y=a`; `E||a` is polarization angle 90 degrees.

The update is persistent method of moving asymptotes (MMA), not Adam. There is
no S-shaped seed, imposed symmetry or material-volume constraint. The 500 nm
solid/void constraints are diagnostic during the low-beta topology-search
phase and become explicit MMA inequalities from beta 8. Terminal conductance
is an inequality from the first update.

Only accepted true-MMA updates are published as numbered iteration figures.
Raw FSP/NPZ files stay outside GitHub and are SHA-256 referenced by the
manifest.
