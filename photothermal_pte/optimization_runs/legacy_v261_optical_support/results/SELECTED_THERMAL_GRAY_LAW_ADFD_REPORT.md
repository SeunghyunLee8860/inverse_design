# Selected-grid thermal gray-law AD–FD

Status: `VALIDATED_SELECTED_THERMAL_GRAY_LAW_ADFD`

This checkpoint isolates the thermal-material branch by freezing the selected
rho=0.5 Maxwell Q. On the exact 373-node/186-cell production support,
`phi_p(rho)=rho^p` is applied consistently to both gray bulk thermal
conductivity and TaIrTe4/design interface conductance. The chain derivative
`p rho^(p-1)` is included analytically. These are numerical relaxation
scenarios, not measured mixture laws or a confidence interval.

For grown/grown interfaces, the finest-step directional AD–FD errors are:

- p=1: `2.289752e-06`;
- p=2: `2.079124e-06`;
- p=3: `1.450776e-06`.

The selected evaporated/evaporated p=1 endpoint gives
`2.706638e-06`.
Every trajectory decreases under h→h/2. Worst residual is
`9.996e-11`, worst energy-balance error is `8.809e-12`,
and worst 373→186 mapping transpose error is `6.171e-16`.

The choice of p is materially consequential even in this fixed-Q isolation:
p=2 changes the grown/grown objective by
`50.903%` and
rotates the thermal gradient by
`5.363°`; p=3 changes it by
`83.367%` and
`9.800°`.

This is not a coupled optical gray-law or full latent certificate. It does
not run Maxwell, exact-binary DRC, or optimization.
