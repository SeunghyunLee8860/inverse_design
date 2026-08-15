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

## Diagnostic stop

This first unbounded version was intentionally stopped after evaluation 6.
NLopt expanded its low-beta asymptotes until the latent field reached both
endpoints while beta was still 1.  The physical solve passed, but this violates
the intended gray-exploration contract.  The result is retained as a diagnostic
and is not a completed optimization.
