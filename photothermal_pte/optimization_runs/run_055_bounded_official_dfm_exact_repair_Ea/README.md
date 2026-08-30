# Run055 — bounded official Ansys DFM + exact repair, E || a

Fresh replacement for the stopped Run054 diagnostic.

- uniform latent density 0.5
- Lumerical `x=b`, `y=a`, `z=c`
- top/bottom contact anchoring
- `E || a`
- evaporated-SiO2 interface scenario
- NLopt LD_MMA, no Adam and no gradient normalization
- one bounded stage per beta
- beta-stage trust region prevents beta-1 endpoint collapse
- official Ansys v261 DFM indicator/gradient activates only for beta > 12
- independent exact 500 nm solid/void repair and fresh physics evaluation
