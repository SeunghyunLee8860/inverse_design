# Component-wise native-Yee material Jacobian

Status: `VALIDATED_COMPONENT_WISE_YEE_MATERIAL_JACOBIAN`

The production material chain is

`rho(81x81) -> epsilon=1+rho*(1.38^2-1) -> n=sqrt(epsilon) ->
importnk2(81x81x13) -> v261 component-specific conformal index_detail ->
epsilon_Yee,c=index_c^2`.

Separate sparse operators `J_x`, `J_y`, and `J_z` were constructed at the
nonuniform physical-density baseline using 25 local colors and centered
layout-only material perturbations. The completed solver `index_detail` and
the layout-mode `index_detail` were identical. This construction ran zero
Maxwell solves and does not use per-pixel Maxwell solves, empirical
normalization, or gradient rescaling.

The component coordinates come directly from `index_detail`: `index_x` uses
`x_offset,y,z`; `index_y` uses `x,y_offset,z`; and `index_z` uses
`x,y,z_offset`. Forward and adjoint fields are read on the same native
`PABS_FIELD` component coordinates. The discarded path that multiplied
separate design-field and design-index arrays by common array index is not
used.

| component | shape | J nnz | max nnz/row | fwd-adj coord mismatch (m) | field-index coord mismatch (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| x | [47, 47, 88] | 53960 | 4 | 4.235165e-22 | 0.000000e+00 |
| y | [47, 47, 88] | 53960 | 4 | 4.235165e-22 | 0.000000e+00 |
| z | [47, 47, 88] | 48749 | 1 | 4.235165e-22 | 0.000000e+00 |

| direction | mapping-only FD error | JVP-VJP dot error |
| --- | ---: | ---: |
| uniform | 5.785642e-11 | 2.120836e-15 |
| smooth_asymmetric | 7.994461e-11 | 8.814659e-15 |
| central_localized | 2.629314e-10 | 0.000000e+00 |
| design_edge_localized | 2.238858e-10 | 8.855235e-16 |
| fixed_seed_random | 1.595087e-10 | 1.043061e-15 |

- Worst mapping-only FD error:
  `2.629313673e-10`
  (limit `1.0e-08`).
- Worst transpose error:
  `8.814659421e-15`
  (limit `1.0e-12`).
- Maximum coordinate mismatch:
  `4.235164736e-22 m`
  (limit `2.0e-18 m`).
- Completed-solver versus layout `index_detail` epsilon error: `0`.
- Baseline layout round-trip epsilon error: `0`.

Raw sparse matrices and coordinate arrays remain outside Git. Paths, sizes,
and SHA-256 values are recorded in the manifest. This checkpoint does not
run downstream thermal/PTE convergence, combined AD-FD, gray-law
sensitivity, latent/filter/projection AD-FD, transient, or optimization.
