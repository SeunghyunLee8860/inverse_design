# Run 009: exact-uniform E||a optimization

- Start from exactly uniform `latent = filtered = physical rho = 0.5` on all 373×373 nodes.
- No random perturbation, S-shaped seed, previous optimizer checkpoint, or fixed internal design mask is allowed.
- Solver axes remain `x=b`, `y=a`, `z=c`; illumination is `E||a` at 90 degrees.
- The objective maximizes the magnitude of the corrected-axis signed PTE current per incident power.
- Beta 2 and 4 are objective/topology-exploration stages; exact 500 nm DRC counts are diagnostics there.
- Smooth morphology feasibility begins at beta 8, phase-wise nonincrease begins at beta 8, and exact-count nonincrease begins at beta 32.
- Beta changes require measured FOM and density convergence; fixed iteration counts do not authorize a beta change.
- Final promotion requires an exact binary density, zero exact 500 nm solid/void violations, and a fresh GPU Maxwell/CUDA thermal evaluation.
