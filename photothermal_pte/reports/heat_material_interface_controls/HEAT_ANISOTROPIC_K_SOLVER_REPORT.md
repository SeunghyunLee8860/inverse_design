# HEAT anisotropic-kappa solver report

**Status: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`.**

- v261 DEVICE version: `7.17.4413`
- Requested tensor: `diag(14.4, 3.8, 1.0) W/(m K)`
- Isotropic fallback: `false`
- Full-device HEAT: `not run`

| Encoding | requested shape | returned shape/value | round trip |
|---|---:|---:|---:|
| vector_1d | `[3]` | `[] / 0.0` | `False` |
| row_1x3 | `[1, 3]` | `[] / 0.0` | `False` |
| column_3x1 | `[3, 1]` | `[] / 0.0` | `False` |
| diagonal_3x3 | `[3, 3]` | `[] / 0.0` | `False` |

| Axis | write/readback before | readback after reload | effective k | flux error | profile error |
|---|---:|---:|---:|---:|---:|
| x | `[14.4, 3.8, 1.0] -> [0.0]` | `[0.0]` | `0` | `100%` | `97.3632%` |
| y | `[14.4, 3.8, 1.0] -> [0.0]` | `[0.0]` | `0` | `100%` | `97.304%` |
| z | `[14.4, 3.8, 1.0] -> [0.0]` | `[0.0]` | `0` | `100%` | `96.8013%` |

The scalar material route passed in the license probe, but v261 did
not retain the requested three-component constant conductivity. The
three directional solves also failed their analytic flux and
temperature-profile controls. A matching readback alone would not
have been accepted; here both readback and solver behavior fail.

No scalar average, coordinate remapping, or isotropic replacement was
used. The production finite-Q source was not imported.

Official Ansys scripting documentation describes the constant thermal
conductivity material field:
https://optics.ansys.com/hc/en-us/articles/360034919233-Creating-and-modifying-thermal-materials-from-a-script
