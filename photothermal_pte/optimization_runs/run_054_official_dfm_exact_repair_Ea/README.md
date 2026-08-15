# Run054 — official Ansys DFM + exact repair, E || a

Fresh, uniform-density (`rho=0.5`) TaIrTe4-flake topology optimization.

- Lumerical coordinates: `x=b`, `y=a`, `z=c`
- contacts: fixed TaIrTe4 at the top/bottom design edges
- polarization: `E || a`
- substrate/interface scenario: evaporated SiO2
- optimizer: NLopt `LD_MMA`
- continuation: one bounded stage at each beta `1,2,4,8,16,32,64,128`
- minimum feature: official Ansys v261 DFM indicator/gradient after beta 12
- final gate: independent exact 500 nm solid/void audit
- initialization: uniform latent density 0.5

The driver does not use the old same-beta recovery loop, exact-bad-cell veto,
manual move limit, Adam update, or empirical gradient rescaling.

