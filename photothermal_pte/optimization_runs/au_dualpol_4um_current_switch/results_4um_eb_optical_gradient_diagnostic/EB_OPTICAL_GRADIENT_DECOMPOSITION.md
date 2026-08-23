# Eb optical-gradient decomposition diagnostic

Status: `DIAGNOSTIC_ONLY_NOT_AN_OPTIMIZATION_GATE`

The fixed baseline thermal-source adjoint weights are held constant while
the Maxwell field-mediated and explicit Au-loss derivatives are separated.
No empirical normalization or gradient rescaling is applied.

| h | field AD/FD error | explicit-loss AD/FD error | optical-total AD/FD error | FD decomposition closure |
|---:|---:|---:|---:|---:|
| 0.005 | 0.003520% | 0.000003% | 0.010950% | 0.005541% |
| 0.0025 | 0.009366% | 0.000003% | 0.015780% | 0.001392% |

Actual float32-ADE susceptibility differs from the requested value by 0.013574%.
