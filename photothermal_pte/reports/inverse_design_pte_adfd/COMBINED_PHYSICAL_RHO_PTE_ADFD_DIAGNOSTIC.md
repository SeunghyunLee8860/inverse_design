# Combined physical-density PTE AD–FD diagnostic

Status: `DIAGNOSTIC_FAILED_COMBINED_PHYSICAL_RHO_PTE_ADFD`

This is a preserved failed checkpoint, not a validation.
No empirical normalization or gradient rescaling was used.

The Maxwell source was the spatial native-Yee vector source
`R_Q^T(dI_PTE/dQ_thermal)`; the scalar `P_Q` source was not
reused. The unresolved path is the component-wise
density-to-Yee material Jacobian/collocation.

## Directional results

| scenario | h | adjoint (A) | FD (A) | relative error |
|---|---:|---:|---:|---:|
| 4um | 0.01 | -7.530897986e-16 | -7.696232220e-16 | 2.148249033e-02 |
| 4um | 0.005 | -7.530897986e-16 | -7.696321029e-16 | 2.149378152e-02 |
| 6um | 0.01 | -9.343018042e-17 | -9.620416471e-17 | 2.883434728e-02 |
| 6um | 0.005 | -9.343018042e-17 | -9.620591656e-17 | 2.885203152e-02 |

## Preserved gates

- Worst selected AD–FD error: `2.885203152e-02`
- Q mapping error: `2.386782948e-16`
- Six-face closure: `2.027067760e-04`
- Thermal energy balance: `3.167349507e-12`
- Worst linear residual: `1.020495014e-11`
- CPU/GPU adjoint field NRMSE: `4.002375889e-05`

The workflow stops here until uniform rho=0.5 representation
equivalence and the nonuniform component-specific Yee
material Jacobian/JVP/VJP are validated.
