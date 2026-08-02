# Device-A source, mapping, and current correction audit

Status: `BLOCKED_DEVICE_A_FIG3H_REGISTRATION_AND_EXACT_LOSS_ATTRIBUTION`

## What was corrected

The old source position was not registered from Figure 3H: the crop was only
stored as provenance. The simulated chord-normal scan differs from the actual
near-vertical black dashed line by
`53.643 deg`,
and vertices 4--7 are not one polygon edge.

The old remap also preserved a complete optical-cell power whenever that cell
had any TaIrTe4 overlap. The new `intersection-density` diagnostic deposits
only `Q * literal intersection volume`, with no nearest-cell relocation,
gain, clipping, smoothing, or global rescaling.

| quantity | E||a | E||b |
|---|---:|---:|
| old attributed P_Q (W) | 4.059899363e-05 | 5.382311210e-05 |
| intersection P_Q (W) | 3.930443310e-05 | 5.272711080e-05 |
| relative change | -3.1887% | -2.0363% |

The legacy-current ratio falls from `1.410409918` to `1.371788234`,
but does not cross one. Strict four-neighbour and common-face quadratures also
retain `abs(Ia)/abs(Ib)>1`. Therefore neither the old cut-cell power rule nor
the one-sided gradient is the sole cause of the reversed paper trend.

## Read-only FSP audit

Component-specific E/index coordinates agree to
`5.082e-21 m`.
The effective-loss ratio leaves [0,1] in cells containing other lossy media,
so it is not an occupancy and was not used as a heat source. This also means
the full-power and literal-intersection results are two named attribution
scenarios, not two claims that one is exact conformal material decomposition.

## Blocking item before another GPU pair

An absolute Figure-3H source coordinate is not published and cannot be
recovered from the current crops without an explicit affine-registration
assumption. No new FDTD was launched. The old 3-um chord-normal point must not
be reused as the experimental position.
