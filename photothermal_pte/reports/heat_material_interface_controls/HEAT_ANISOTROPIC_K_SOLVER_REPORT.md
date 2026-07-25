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

## Exhaustive native v261 probe

A fresh DEVICE session tested LSF-native 3x1, 1x3, and 3x3
matrix expressions, eleven plausible hidden property names, and
every material returned by `addmaterialproperties("HT")`.

- Native probe status: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- HT database entries: `64`
- Readable scalar conductivity entries: `59`
- Non-scalar conductivity entries: `0`
- Hidden property writes accepted: `0`

| Native LSF encoding | requested | returned | exact round trip |
|---|---:|---:|---:|
| column_3x1 | `[[14.4], [3.8], [1.0]]` | `0.0` | `False` |
| row_1x3 | `[[14.4, 3.8, 1.0]]` | `0.0` | `False` |
| diagonal_3x3 | `[[14.4, 0.0, 0.0], [0.0, 3.8, 0.0], [0.0, 0.0, 1.0]]` | `0.0` | `False` |

This closes the known native v261 material routes: the installed
HEAT material API exposes scalar conductivity only. Consequently
`BLOCKED_ANISOTROPIC_K_UNSUPPORTED` remains correct specifically
for a v261 HEAT-backed result.

## Working anisotropic path

**Fallback status: `VALIDATED_DIAGONAL_KAPPA_FVM_CONTROLS`.**

A repository-native, cell-centered conservative finite-volume
solver now accepts cellwise `diag(kx, ky, kz)`. Conductances use
the exact series resistance of adjacent half cells; unspecified
outer faces are adiabatic. This is an independently validated
solver path, not a relabeled Lumerical result.
The present implementation is intentionally limited to diagonal
tensors aligned with the Cartesian grid; that exactly matches the
requested `diag(14.4, 3.8, 1.0)` tensor.

| Axis | expected k (W/m K) | recovered k (W/m K) | flux error | profile error | energy error |
|---|---:|---:|---:|---:|---:|
| x | `14.4` | `14.4000000032` | `2.19785e-08%` | `5.00734e-09%` | `1.10444e-09%` |
| y | `3.8` | `3.80000000087` | `2.29702e-08%` | `3.24576e-09%` | `1.48983e-09%` |
| z | `1` | `1.0000000013` | `1.29624e-07%` | `2.76259e-09%` | `4.97335e-09%` |

All three controls satisfy the requested `<1%` heat-flux and
temperature-profile criteria without an isotropic average.

Reproduce the controls with:

```bash
python photothermal_pte/validation/photothermal_stage1/31_resolve_anisotropic_kappa.py \
  --phase fvm-controls --output-dir /tmp/anisotropic-kappa-controls
```

Official Ansys scripting documentation describes the constant thermal
conductivity field as scalar and lists only Solid, Solid Alloy,
and Fluid thermal property types:
https://optics.ansys.com/hc/en-us/articles/360034919233-Creating-and-modifying-thermal-materials-from-a-script
https://optics.ansys.com/hc/en-us/articles/360034924973-addhtmaterialproperty-Script-command
