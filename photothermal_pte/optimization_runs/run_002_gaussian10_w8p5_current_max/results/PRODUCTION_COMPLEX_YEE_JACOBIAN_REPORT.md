# Production complex component-Yee Jacobian

Status: `VALIDATED_PRODUCTION_COMPLEX_COMPONENT_YEE_JACOBIAN`

This is the layout-only mapping certificate for the actual 20×20×1 µm coarse production-candidate design at 10 µm. It uses 201×201 physical-density nodes and the actual v261 component-specific `index_detail` coordinates. It performed zero Maxwell solves and no per-pixel solves.

The differentiated material chain is:

```text
rho -> epsilon=1+rho*(epsilon_SiO2-1) -> passive complex sqrt
    -> importnk2 -> index_detail_c -> epsilon_Yee,c=index_c^2
```

with `epsilon_SiO2=7.349001930304349 +
1.989968728688058 i`.

| direction | mapping centered-FD relative error | JVP/VJP dot relative error |
|:--|--:|--:|
| uniform | 1.337003e-09 | 2.912542e-16 |
| smooth_asymmetric | 1.199907e-09 | 5.343477e-15 |
| central_localized | 1.239428e-09 | 7.333851e-16 |
| design_edge_localized | 8.215347e-10 | 1.043654e-15 |
| fixed_seed_random | 1.163420e-09 | 5.543527e-16 |

| component | Yee shape | J nonzeros | max nonzeros/Yee sample | E/index coordinate mismatch (m) | active rows outside exact support |
|:--:|:--|--:|--:|--:|--:|
| x | [311, 311, 52] | 1,929,600 | 2 | 6.776264e-21 | 0 |
| y | [311, 311, 52] | 1,929,600 | 2 | 6.776264e-21 | 0 |
| z | [311, 311, 52] | 929,223 | 1 | 6.776264e-21 | 0 |

Worst mapping FD error: `1.337003e-09`
(gate `1.0e-07`).

Worst transpose error: `5.343477e-15`
(gate `1.0e-12`).

Maximum component coordinate mismatch:
`6.776264e-21 m`.

## Scope boundary

This validates the full production-geometry complex interpolation, component coordinates, exact design-support intersection, and sparse transpose. It is not yet a Maxwell adjoint, thermal/PTE gradient, or optimization certificate.
