# Nonuniform complex component-Yee Jacobian smoke

Status: `VALIDATED_NONUNIFORM_COMPLEX_COMPONENT_YEE_JACOBIAN_SMOKE`

This is a layout-only mapping control for an isolated 10×10×1 µm imported
complex-SiO2 block at 10 µm.  It uses 101×101 physical-density nodes and the
actual v261 component-specific `index_detail` coordinates.  It performed zero
Maxwell solves and no per-pixel solves.

The differentiated material chain is:

```text
rho -> epsilon=1+rho*(epsilon_SiO2-1) -> passive complex sqrt
    -> importnk2 -> index_detail_c -> epsilon_Yee,c=index_c^2
```

with `epsilon_SiO2=7.349001930304349 +
1.989968728688058 i`.

| direction | mapping centered-FD relative error | JVP/VJP dot relative error |
|:--|--:|--:|
| uniform | 1.339796e-09 | 1.947244e-16 |
| smooth_asymmetric | 1.202034e-09 | 5.449440e-16 |
| central_localized | 1.237475e-09 | 7.314868e-15 |
| design_edge_localized | 8.192424e-10 | 1.607256e-15 |
| fixed_seed_random | 1.150351e-09 | 4.598157e-16 |

| component | Yee shape | J nonzeros | max nonzeros/Yee sample | E/index coordinate mismatch (m) |
|:--:|:--|--:|--:|--:|
| x | [111, 111, 65] | 828,200 | 2 | 8.470329e-22 |
| y | [111, 111, 65] | 828,200 | 2 | 8.470329e-22 |
| z | [111, 111, 65] | 408,040 | 1 | 8.470329e-22 |

Worst mapping FD error: `1.339796e-09`
(gate `1.0e-07`).

Worst transpose error: `7.314868e-15`
(gate `1.0e-12`).

Maximum component coordinate mismatch:
`8.470329e-22 m`.

## Scope boundary

This validates the complex interpolation and sparse-Jacobian construction
method on the isolated control only.  It is not the final production-geometry
Jacobian, a Maxwell adjoint certificate, a thermal/PTE gradient certificate,
or permission to start optimization.
