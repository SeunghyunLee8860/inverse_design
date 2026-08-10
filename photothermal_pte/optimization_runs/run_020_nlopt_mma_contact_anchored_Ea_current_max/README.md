# Run020 — production NLopt LD_MMA, E parallel a

Fresh contact-anchored TaIrTe4 topology optimization from exact uniform
physical density `rho=0.5`, using NLopt 2.11 `LD_MMA`.

- Lumerical coordinates: `x=b`, `y=a`, `z=c`
- Manual move limit: none
- Custom MMA update, Adam and gradient normalization: none
- `ftol_rel=1e-3`, corrected `xtol_rel=1e-7`
- Strict NLopt inequalities and beta continuation
- Raw FSP/NPZ artifacts remain under `/data`

Run018 is preserved only as the failed loose-xtol diagnostic.
