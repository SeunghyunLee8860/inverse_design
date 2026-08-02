# Registered Device-A single-position Maxwell–thermal–PTE audit

Status: `PARTIAL_REGISTERED_DEVICE_A_SINGLE_POSITION_CURRENT_TREND_OPPOSITE_PAPER_SCAN_PEAK`

This is one approximately registered point on the Figure-3H dashed-line
scenario, not a scan maximum and not a certified reproduction of Figure 3I.
Both polarization-specific GPU Maxwell calculations, literal material-overlap
mapping operations, and the identical explicit 3D thermal/PTE operators passed
their numerical gates.

| metric | E parallel a | E parallel b | b/a |
|---|---:|---:|---:|
| TaIrTe4 mapped power at 284.40 uW | 1.18561708e-05 W | 1.32428782e-05 W | 1.116961 |
| Tmax rise | 0.27858970 K | 0.15283275 K | 0.548594 |
| flake volume-average rise | 0.00775779 K | 0.00873465 K | 1.125920 |
| integrated PTE current | 1.36996117e-08 A | 1.14627972e-08 A | 0.836724 |

The total/mapped absorption has the paper-like `b>a` trend, but the `a`
polarization produces a stronger localized hotspot and the single-position
integrated current remains `a>b`. The current was evaluated as a full
flake-cell volume integral; it is not a one-point gradient sample.

Absolute current is not certified. The digitized geometry predicts about
`14.107 ohm`, versus the measured `213 ohm`.
No resistance or current rescaling was applied. The next physically meaningful
comparison is the registered sparse scan and the separate maxima of `|Ia|` and
`|Ib|`, matching the interpretation of Figure 3I.
