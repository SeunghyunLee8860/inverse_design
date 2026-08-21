# Discrete passive-Drude adjoint control

Status: `VALIDATED_DISCRETE_PASSIVE_DRUDE_ADJOINT_CONTROL`

This is a one-dimensional fixed-grid algorithmic control, not a 3-D
production result.  The exact 10-um Au endpoint
`epsilon=-4642.230000+1674.640000i` is represented by a passive
one-pole Drude model with positive `omega_p` and `gamma`.  Density changes the
pole strength through `s(rho)=rho^3`; it does not move a conformal CAD
boundary.

The discrete adjoint includes both terms:

```text
-2 Re[lambda^H (dA/drho) E] + E^H (dW_loss/drho) E
```

The baseline linear residual is `3.943e-14` and the largest
residual over all central-FD solves is `4.067e-14`.  At the finest
step `h=0.000625`, the largest relative AD--FD error over the strong
directions is `0.000435134%`.  The
central-localized direction is near-null; using the common gradient scale, the
largest normalized error over all five directions is
`6.552e-07`.  A cancellation-dominated
near-null FD is not mislabeled as a 100% physical-gradient error.

This pass proves the required mathematical repair: optimize causal dispersive
state parameters on a fixed discrete Maxwell operator and differentiate the
same operator.  It does **not** make the failed v261 moving-Au boundary
gradient valid.  A production implementation still requires 3-D Yee/PML and
Drude/CCPR auxiliary-state AD--FD certification, followed by exact-binary
Lumerical endpoint cross-validation.
