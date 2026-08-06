# Uniform complex-material representation equivalence

Status: `VALIDATED_UNIFORM_COMPLEX_MATERIAL_REPRESENTATION_EQUIVALENCE`

At 10 µm the design law is

```text
epsilon(rho) = 1 + rho * (epsilon_SiO2 - 1)
epsilon_SiO2 = 7.3490019303043495 + 1.9899687286880576 i
```

The scalar `(n,k) Material` and uniform `importnk2` representations were run
with the same calibrated Gaussian source, local mesh, conformal variant 1,
and matched Q/six-face control volume.  No Q clipping, smoothing, gain, or
rescaling was applied.

| rho | rel. P_Q diff | rel. P_six diff | worst closure (%) | worst spatial component NRMSE | pass |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.000e+00 | 0.000e+00 | N/A (lossless) | 0.000e+00 | True |
| 0.5 | 0.000e+00 | 0.000e+00 | 0.01571 | 1.641e-16 | True |
| 1 | 1.247e-16 | 0.000e+00 | 0.00891 | 2.169e-16 | True |

For rho=0 both representations read back exactly epsilon=1+0i and all three
Q components are exactly zero.  Therefore a relative absorption closure is
ill-conditioned; this endpoint is judged from exact-zero loss and epsilon
readback instead.

## Scope boundary

This result validates only uniform rho=0, 0.5, and 1 representation
equivalence.  It does **not** validate nonuniform interpolation onto the
component-specific Yee grids, JVP/VJP transpose behavior, Maxwell adjoint
sources, combined PTE gradients, or optimization.  Those remain fail-closed.
