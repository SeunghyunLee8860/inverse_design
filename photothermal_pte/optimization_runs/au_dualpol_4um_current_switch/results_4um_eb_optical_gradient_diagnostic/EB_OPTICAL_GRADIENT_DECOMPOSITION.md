# Eb optical-gradient decomposition diagnostic

Status: `DIAGNOSTIC_ONLY_NOT_AN_OPTIMIZATION_GATE`

The fixed baseline thermal-source adjoint weights are held constant while
the Maxwell field-mediated and explicit Au-loss derivatives are separated.
No empirical normalization or gradient rescaling is applied.

| h | field AD/FD error | explicit-loss AD/FD error | optical-total AD/FD error | FD decomposition closure |
|---:|---:|---:|---:|---:|
| 0.005 | 0.013921% | 0.001783% | 0.022795% | 0.004737% |
| 0.0025 | 0.014154% | 0.000447% | 0.019175% | 0.001175% |

Actual float32-ADE susceptibility differs from the requested value by 0.013574%.
