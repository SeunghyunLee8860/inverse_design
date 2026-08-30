# Run056 — bounded official Ansys DFM + exact repair, E || b

Parallel E || b counterpart of Run055.

- fresh uniform latent density 0.5
- Lumerical `x=b`, `y=a`, `z=c`
- top/bottom contact anchoring
- `E || b`
- evaporated-SiO2 interface scenario
- NLopt LD_MMA, no Adam and no gradient normalization
- one bounded stage per beta
- beta-stage trust region prevents beta-1 endpoint collapse
- official Ansys v261 DFM indicator/gradient activates only for beta > 12
- independent exact 500 nm solid/void repair and fresh physics evaluation
- raw artifacts are isolated from Run055
- a run-local FDTD engine lock permits true GPU-2 execution alongside Run055
