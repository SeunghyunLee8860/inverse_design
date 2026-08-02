# Registered Device-A sparse Maxwell–thermal–PTE scan

Status: `PARTIAL_REGISTERED_DEVICE_A_SPARSE_SCAN_CURRENT_TREND_OPPOSITE_PAPER`

This is a **sparse registered diagnostic**, not a paper-certified beam, exact
Device-A reproduction, or continuous scan-peak fit.  The optical source is the
explicitly assumed scalar Gaussian `w0=8.75 um` scenario.  The flake/electrode
geometry is digitized from the published figures because exact CAD is not
available.

## Result

| sampled maximum | coordinate d | integrated current | paper visual reference |
|---|---:|---:|---:|
| `|Ia|` | 1.0 um | 13.722020 nA | about 122 pA |
| `|Ib|` | 3.0 um | 11.462797 nA | about 143 pA |

The sampled-maximum ratio is

`max_sampled(|Ib|) / max_sampled(|Ia|) = 0.835358`.

Figure 3I visually gives the opposite trend, about `143/122 = 1.172131`.
Because the `a` and `b` sampled maxima occur at different coordinates, this
result rules out the earlier concern that the reversal came only from comparing
one common position.  It does **not** prove sub-micrometre continuous peak
convergence: the present spacing is 2 um.

## Numerical gates

- All seven finite optical cases pass matched-volume closure `<0.5%`, final
  auto-shutoff `<1e-5`, and the independently recomputed position-matched
  empty-stack reference audit: `True`.
- Every source uses the literal optical-cell/TaIrTe4 intersection-density
  mapping.  Non-overlapping air/SiO2 power is not forced into TaIrTe4.
- Every thermal case has mapping error `<1e-12`, zero mapped power outside the
  TaIrTe4 support, linear residual `<1e-8`, and energy error `<1%`: `True`.
- Current is the full flake-volume Shockley–Ramo integral.  It is not sampled
  from one temperature-gradient point.
- No Q clipping, smoothing, gain, global rescaling, nearest-cell relocation,
  current fit, or resistance fit was used.

## Physical interpretation and limits

The mapped absorbed power remains generally larger for `E || b`, but the
spatial heat-source/temperature/weighting-field overlap gives a larger
integrated current for `E || a` throughout the paired sparse points.  The
current discrepancy therefore remains a physical-model/geometry problem, not
a failed optical closure or conservative-mapping problem.

Absolute current is blocked: the digitized conductivity geometry predicts
`14.106622 ohm`, whereas Device A measured
`213 ohm`.  Exact CAD, contact
resistance/geometry, metal thermalization/interface data, and absolute scan
metrology are not published.  The nA values above are consequently not called
experimental predictions and were not rescaled to the paper's pA values.

One attempted parallel optical launch is preserved as a fail-closed diagnostic:
concurrent Lumerical sessions raced through the shared user-level GPU resource
configuration.  Production optical sessions were therefore run sequentially.

The next useful step, if a tighter comparison is required, is a denser local
scan around each sampled maximum together with sensitivity to the digitized
contact/weighting geometry.  AD–FD or optimization would not resolve this
paper-reproduction discrepancy.
