# Paper SPCM minimal reproduction sanity check

Status: `VALIDATED_PAPER_SPCM_MECHANISM_SANITY_CHECK`

## Scope

This is a separate sanity check of the simplified two-dimensional mechanisms
in Blevins *et al.*, Fig. 1F/I and Supplementary Note S5. It does **not**
replace or modify the non-periodic inverse-design optical/thermal contract,
and it is not a pixel-for-pixel reproduction of Device A or Device B.

The paper does not publish the exact simplified-Fig.-1 rectangle dimensions,
mesh, numerical 635-nm beam radius, or absorbed fraction. We therefore use an
explicit 6 µm × 8 µm × 130 nm canonical rectangle, report current per absorbed
power, and compare the predicted symmetry, sign, and mechanism. The assumed
beam radius is 0.50 µm; 0.40 and 0.75 µm are included as assumption
sensitivity cases. No fitted gain or map rescaling is used for numerical
metrics (normalization is used only for visualization).

## Paper equations and parameters used

- Gaussian source: Supplement Eq. S1.
- Anisotropic steady heat equation: Supplement Eq. S3.
- Explicit Robin boundaries: Supplement Eq. S4, with
  `G_top(air)=1 W/(m² K)` and
  `G_bottom(thermally-grown SiO2)=7.37e6 W/(m² K)`.
- Local PTE source and continuity: Supplement Eq. S5.
- Shockley–Ramo collection: Supplement Eq. S6.
- Weighting potential: Supplement Eq. S7; full-width top electrode is 1,
  bottom electrode is 0, and lateral sample-air edges are electrically
  insulating.
- `kappa_a=14.4`, `kappa_b=3.8 W/(m K)`;
  `sigma_a=4.91e5`, `sigma_b=1.10e5 S/m`;
  `S_a=-6`, `S_b=27 µV/K`; `T_bath=300 K`.

The lateral edges are thermal zero-flux exactly as stated for the paper's 2-D
IR edge calculation. Top and bottom are **not** adiabatic: both use the
explicit paper Robin conductances.

## Main sanity result

At 45°, the analytic lab-frame PTE coupling is

- `|(sigma S)_yy| = 0.012 A/(m K)`
- `|(sigma S)_yx| = 2.958 A/(m K)`
- transverse/electrode coupling ratio =
  `246.5`.

Thus the electrode-direction term is nearly cancelled by the paper's p×n
Seebeck/conductivity values, while the transverse term remains. The simulated
50-nm-grid maps show:

| case | side/electrode peak | expected odd-symmetry residual | peak response |
|---|---:|---:|---:|
| a-axis aligned | 0.104136 | 6.462e-04 | 8.068275e-03 A/W_abs |
| a-axis at 45° | 3.59848 | 1.831e-01 | 7.952125e-03 A/W_abs |

This reproduces the paper's central Fig. 1 sanity claim: aligned axes give the
longitudinal, opposite-sign electrode response, whereas the 45° p×n geometry
suppresses electrode response and leaves opposite-sign side-edge response.

## Numerical checks

| case | map NRMSE, 100→50 nm | peak difference | edge-ratio difference |
|---|---:|---:|---:|
| α=0° | 0.221% | 0.633% | 1.233% |
| α=45° | 0.175% | 1.087% | 0.260% |

The direct forward and thermal-adjoint currents agree to
`2.536e-16` relative error in
the 45° check. Its linear residual is
`2.025e-14` and energy-balance error is
`1.466e-14`.

## What is and is not reproduced

Reproduced:

- the paper's published material tensor values and explicit thermal-interface
  laws;
- Gaussian local heating, insulating crystal edges, solved weighting
  potential, and Shockley–Ramo collection;
- the longitudinal-electrode versus transverse-edge sign/symmetry change;
- numerical mesh convergence and conservation checks.

Not claimed:

- exact Fig. 2H/5H magnitude or pixelwise agreement, because device CAD,
  exact electrode masks, local 635-nm absorption, objective transmission, and
  exact beam-radius input are not supplied as numerical data;
- 3-D COMSOL reproduction, transient response, optical FDTD/RCWA,
  inverse design, or optimization;
- an experimental current prediction. The reported scale is A per absorbed W.

## Artifacts

- `PAPER_SPCM_SANITY_MAPS.png`
- `PAPER_SPCM_NUMERICAL_CONVERGENCE.png`
- `PAPER_SPCM_BEAM_RADIUS_SENSITIVITY.png`
- `paper_spcm_sanity_summary.json`
- `paper_spcm_sanity_cases.csv`
- `RAW_ARTIFACT_MANIFEST.json`
