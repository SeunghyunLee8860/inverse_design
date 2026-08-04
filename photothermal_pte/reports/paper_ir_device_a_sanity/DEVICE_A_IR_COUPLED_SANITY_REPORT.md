# Device A paper-like 11 µm coupled sanity check

Status: `FAILED_COUPLED_DEVICE_A_IR_PTE_SANITY_GEOMETRY_UNRESOLVED`

## Outcome

This was an actual coupled run:

`Lumerical v261 GPU Gaussian Q → conservative support remap → current expanded
thermal-cell FVM → solved approximate electrode weighting potential → PTE`.

The optical sanity check passed, but the full coupled experimental sanity
check did **not**. No empirical gain, current rescaling, or parameter fitting
was used.

## Published inputs held fixed

- Device-A TaIrTe4 thickness: 130 nm; substrate: 285 nm SiO2/Si.
- `kappa_a,b,c = 14.4, 3.8, 1.0 W/(m K)`.
- `sigma_a,b = 4.91e5, 1.10e5 S/m`.
- `S_a,b = -6, +27 µV/K`.
- `G_TaIrTe4/air = 1 W/(m² K)`;
  `G_TaIrTe4/thermally-grown-SiO2 = 7.37e6 W/(m² K)`.
- Normal-incidence 11 µm Gaussian illumination, 285 µW incident power,
  both `E||a` and `E||b`.

Density and heat capacity were not used because this is steady state.

## Optical result

| metric | E||a | E||b | a/b |
|---|---:|---:|---:|
| TMM absorption | 17.673% | 26.329% | 0.671 |
| finite central Lumerical absorption | 17.350% | 25.114% | 0.691 |
| off-axis-edge absorbed power at 285 µW | 28.597 µW | 35.578 µW | 0.804 |
| six-face closure, central | 0.010% | 0.036% | — |
| six-face closure, edge | 0.058% | 0.064% | — |

The central values agree with the paper Fig. 3D (approximately 18% and 26%)
and the independent TMM. This validates the 130-nm material-axis optical
contract at 11 µm.

## Thermal/PTE result

| model | Tmax E||a | Tmax E||b | |Ia|/|Ib| |
|---|---:|---:|---:|
| current expanded FVM | 0.2814 K | 0.2413 K | 1.189 |
| paper Eq. S4 reduced Robin reference | 0.1664 K | 0.1104 K | 1.227 |
| paper experiment | — | — | approximately 0.80 |

Both numerical models conserve energy below 1%, have residual below 1e-8,
and preserve Q mapping power. Nevertheless both predict `|Ia|/|Ib| > 1` at
the chosen edge point, opposite to the paper.

The immediate cause is visible in the raw Lumerical Q: `E||a` creates a
strong hotspot at the upper concave corner of the approximate polygon. The
same reversal in the expanded and reduced thermal models shows that the
production boundary expansion is not the primary cause.

## What remains unresolved

The exact Device-A CAD, electrode mask, beam location, and wavelength-specific
beam radius are not published numerically. We used a named polygon digitized
from Fig. 2A, `w0=6.5 µm`, and edge centre `(-8.5, 3.5) µm`. The off-axis
optical model excludes the 5-nm Ti/50-nm Au contacts because the selected
spot is away from the contacts. These approximations are sufficient for an
optical material check, but not for promotion as a quantitative experimental
PTE reproduction.

The calculated absolute current is also much larger than the paper's
order-100-pA map. Therefore it is not called an experimental prediction.
The next discriminating checks are exact CAD/spot metrology if available,
or a separate local half-plane edge geometry matching the paper's Fig. 3F
thermal idealization, plus optical/thermal mesh refinement of the `E||a`
corner hotspot.

## Model separation

The expanded case uses explicit bulk Si, SiO2 and air, `G_SiO2/Si=1.1e9`,
top `h=10`, far-x/y fixed DeltaT=0, and bottom fixed DeltaT=0. Those are the
current inverse-design production assumptions and are **not** claimed as
paper-supplied values.

The paper-reduced reference instead solves the flake with top/bottom Eq. S4
Robin conductances. The previous 2-D analytic/FEM work remains only
`PASSED_PAPER_EQUATION_MECHANISM_CONTROL`; this report supersedes any wording
that implied it was an actual Lumerical experimental reproduction.

## Files

- `DEVICE_A_IR_COUPLED_SANITY.png`
- `DEVICE_A_WEIGHTING_AND_PTE_DIAGNOSTICS.png`
- `device_a_ir_sanity_summary.json`
- `device_a_ir_sanity_cases.csv`
- `RAW_ARTIFACT_MANIFEST.json`
