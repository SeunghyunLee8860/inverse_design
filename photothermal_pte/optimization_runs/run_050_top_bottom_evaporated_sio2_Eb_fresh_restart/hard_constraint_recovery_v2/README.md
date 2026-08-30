# Run050 hard-constraint recovery v2

Recovery v1 proved that the inverse-filter seed and both explicit LD_MMA
inequalities were feasible, but its inherited beta-scaled `rho_init=136.86`
made the second latent update effectively zero (RMS step 1.68e-9).

Version 2 preserves the same exact-feasible beta=8 seed, filter radius,
physical objective, gradients, and inequality caps. It removes the inherited
custom CCSA curvature parameters in hard-constraint mode and lets NLopt use its
native `rho_init`, `always_improve`, and `inner_gradients` defaults. The
projection-aware initial-step scale remains 0.0067578 in latent density.

This is not a gradient rescaling and does not alter the Maxwell, thermal,
electrical, or adjoint operators.
