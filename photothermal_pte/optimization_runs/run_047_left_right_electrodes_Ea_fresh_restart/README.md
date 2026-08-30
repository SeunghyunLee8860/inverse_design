# Run 047 — fresh left/right-electrode E||a optimization

This is a clean optimization restart of the Run046 physical contract.

- Lumerical axes: `x=b`, `y=a`, `z=c`
- source: `E||a` (90 degrees)
- terminals: left `psi=0`, right `psi=1`
- initial latent/physical density: uniform `rho=0.5`
- first stage: `beta=1`
- optimizer: native NLopt `LD_MMA`
- no warm-start checkpoint and no reused MMA state
- no symmetry, volume, or connectivity constraint
- 500 nm solid/void minimum-feature constraints

The validated uniform forward FSP, component-Yee Jacobian, and combined AD-FD
certificate are reused unchanged. Raw FSP/NPZ files remain outside Git.
