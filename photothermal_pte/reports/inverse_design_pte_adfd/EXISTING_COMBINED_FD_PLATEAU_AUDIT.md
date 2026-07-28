# Existing combined FD plateau audit

- New Lumerical solves: `0`
- Original certificate: `FAILED_CORRECTED_COMBINED_PHYSICAL_RHO_PTE_ADFD`
- Original certificate remains failed; this is a diagnostic only.
- No empirical normalization or gradient rescaling was used.

## Fast prerequisite checks

- Component-wise Yee mapping: worst mapping-only FD error `2.62931e-10`, worst JVP/VJP dot error `8.81466e-15`, maximum coordinate mismatch `4.23516e-22 m`
- Thermal-only PTE AD-FD: worst selected error `1.91862e-06`, energy-balance error `3.46728e-12`, linear residual `1.01805e-11`

## Evidence from the immutable 30-solve sweep

- 4um central_localized: strength/||g|| `0.0753555`, response/aligned-response `0.00316914`, PTE plateau `0.00139509`, P_Q plateau `8.57377e-05`, fine/coarse derivative-difference ratio `3.81226`
- 4um fixed_seed_random: strength/||g|| `0.089459`, response/aligned-response `0.00376227`, PTE plateau `0.00509411`, P_Q plateau `0.000694249`, fine/coarse derivative-difference ratio `5.38293`
- 6um fixed_seed_random: strength/||g|| `0.181094`, response/aligned-response `0.00761604`, PTE plateau `0.00242508`, P_Q plateau `0.000694249`, fine/coarse derivative-difference ratio `3.26086`

For a smooth centered finite difference dominated by the usual O(h^2) truncation term, the derivative-difference ratio after halving h is approximately `0.25`. It is greater than one for every failed direction, while all failed responses are below 1% of the tested adjoint-aligned response. This localizes the unresolved issue to weak-direction numerical resolution/cancellation rather than a global gradient scale or sign error. It does not establish run-to-run FDTD stochasticity; identical-solve repeats would still be required for that claim.
