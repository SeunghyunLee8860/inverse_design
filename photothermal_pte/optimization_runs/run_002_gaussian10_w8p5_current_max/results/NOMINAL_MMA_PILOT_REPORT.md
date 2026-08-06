# Run 002 nominal MMA pilot

Status: `RUNNING_NOMINAL_MMA_PILOT`

The first actual GPU-Maxwell/CUDA-thermal MMA candidate is accepted. The signed PTE objective increased from `1.351217541492e-20 A` to `1.472735412129e-20 A`, a `8.993213%` improvement.

The optimizer uses 0≤latent≤1, a finite nonperiodic 500 nm conic filter, beta=2 tanh projection, and a fixed objective nondimensionalization of `1.000e+12 W/A × I/P_incident`. It does not dynamically rescale gradients. No volume or symmetry constraint is imposed in this nominal pilot. Exact binary 500 nm solid/void DRC is still required before fabrication promotion.

This is not a final optimized structure or an experimental-current prediction. It is iteration 1 of the grown/grown, +I, uniform-45° weighting-surrogate stage.
