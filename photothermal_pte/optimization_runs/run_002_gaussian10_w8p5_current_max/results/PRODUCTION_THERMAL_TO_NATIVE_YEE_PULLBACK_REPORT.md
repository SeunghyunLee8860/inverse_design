# Production thermal/PTE to native-Yee pullback

Status: `VALIDATED_PRODUCTION_THERMAL_PTE_TO_NATIVE_YEE_PULLBACK`

The uniform-rho production thermal adjoint was pulled through the exact
material-intersection deposition used by the forward source. The transpose is
applied as three memory-bounded 1-D overlap contractions; no full 3-D
Kronecker matrix, nearest-material relocation, Q rescaling, or index pairing
is used.

| native Q component | transpose dot error | actual-Q objective contribution (A) |
|---|---:|---:|
| x | 4.096795e-15 | 3.039095272752e-26 |
| y | 0.000000e+00 | -1.997174324457e-28 |
| z | 1.768388e-16 | -1.741598888483e-30 |

- worst transpose dot error: `4.096795e-15`;
- forward residual: `8.217297e-11`;
- adjoint residual: `9.217972e-11`;
- thermal energy balance: `9.699736e-13`;
- Cauchy-normalized objective identity error:
  `5.103599e-14`;
- Cauchy-normalized reciprocity error:
  `1.008309e-15`.

The raw relative identity is cancellation-dominated because the centered
rho=0.5 PTE value is near zero. It remains in the JSON as a diagnostic and is
not used to rescale the gradient. The stored native absorption weights are now
the spatial weights needed to construct a thermal-weighted Maxwell adjoint
source. No Maxwell solve or optimization was run in this checkpoint.
