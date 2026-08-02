# Device-A 11-µm measured-current comparison

Status: `FAILED_DEVICE_A_PAPER_LIKE_CURRENT_POLARIZATION_RATIO`

This result uses Kitamura-2007 SiO2, Palik Si as an explicit unpublished
closure, the digitized Device-A geometry, exact 284.40-µW input power, and the
paper-reduced TaIrTe4 Robin thermal model.  It is not fitted to the measured
current or the 213-ohm resistance.

## Result

| Quantity | Simulation | Paper target |
|---|---:|---:|
| `|Ia|` | 5554.95 pA | about 122 pA from Fig. 3I, with SI fit 110.6 pA at 3675 Hz |
| `|Ib|` | 3938.54 pA | about 143 pA from Fig. 3I |
| `|Ia|/|Ib|` | 1.41041 | 0.83659 ± 0.00852575 |

## Optical and beam gates

| Quantity | E || a | E || b |
|---|---:|---:|
| full matched-volume `P_Q` at central 1 W/m2 | 2.033929251e-11 W | 2.723903316e-11 W |
| TaIrTe4-support `P_Q` at central 1 W/m2 | 1.931340918e-11 W | 2.615363137e-11 W |
| six-face closure | 0.03583% | 0.00773% |
| auto-shutoff | 9.6957e-06 | 9.9966e-06 |
| realized effective waist at target plane | 8.79210 µm | 8.78988 µm |
| Gaussian fit RMS / peak | 0.05902% | 0.06313% |

The requested physical waist is 8.75 µm.  The target-plane profile is read
from the matching empty SiO2/Si stack and is therefore evidence for the
realized downward field in that layered reference, not a claim of an
independently measured experimental waist.

The Figure-3I axis prints nA, but the Figure-3H/SI maps and the independent
SI frequency fit are in pA.  Both interpretations remain recorded; pA is the
physically consistent comparison.

## Certification limits

- All optical closure, auto-shutoff, source-mapping, thermal energy-balance,
  residual, and weighting-potential gates are reported independently.
- The polarization-ratio gate fails: the simulation gives
  `1.41041`, versus the digitized paper
  value `0.83659`.  The trend is reversed and is
  not called a reproduction.
- The digitized geometry predicts a two-terminal resistance far from the
  measured 213 ohm.  Therefore absolute-current agreement is not certified
  and no conductivity/current rescaling is applied.
- The exact beam definition, objective transmission, CAD/contact resistance,
  and scan coordinates are unpublished.  The result is a named paper-like
  scenario, not a unique reconstruction of the authors' hidden model.
