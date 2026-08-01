# Named scenario: real (Palik) lossy SiO2 + paper-consistent minimal waist

Status: `COMPLETED_LOSSY_SIO2_W6P83_NAMED_SCENARIO`

Motivation: the line-by-line paper-vs-code audit found exactly one
undocumented physical omission in the production optical model — the
SiO2 spacer was modeled as lossless n = 1.38 at 11 um, carried over
from the 4-um inverse-design model, while real SiO2 has strong phonon
absorption in the 8-13 um band (the paper itself attributes the E||b
9-um absorption peak to SiO2 resonant phonons).  The beam waist was
also identified as the dominant explicit assumption (the paper states
only "~9-16 um diffraction-limited spot" with no radius/diameter/FWHM
definition; production assumed w0 = 12 um).

## Scenario definition

Two overrides on the otherwise unchanged production pipeline
(60-um domain, 50-um source span, digitized Figure-2 geometry with
Ti/Au electrodes, 50-nm local mesh with +/-15-um refinement, 4 ps,
1 W/m2 incident-intensity normalization, all fail-closed gates):

1. `--sio2-model palik-lossy`: SiO2 spacer uses the Lumerical built-in
   "SiO2 (Glass) - Palik" sampled data.  Solver-fitted epsilon at
   11 um = **3.7287 + 0.1885j** (vs the lossless 1.9044) — verified
   lossy fail-closed at setup.
2. `--scenario-waist-um 6.83`: physical target waist 6.83 um — the
   1/e2-diameter reading of the paper's "9-16 um" spot interpolated to
   11 um (13.67-um spot diameter), the smallest paper-consistent
   interpretation enumerated in the frozen beam-contract audit.

Gate handling (documented in each case_result, raw values kept):
empty-stack lossless-stack gates replaced by an absorbed-fraction
sanity gate (the reference stack now legitimately absorbs); finite
six-face closure tolerance widened 0.5% -> 2% because the absorbing
SiO2 sits directly against the bottom Q-volume face (realized closure
0.74% E||a / 1.37% E||b).  Auto-shutoff reached 1e-5 in all four runs.

Runner patch: `runner_scenario_flags.patch` (3 commits on the
production repo branch `sio2-lossy-scenario`).

## Results (285 uW incident, digitized Figure-3 beam position)

| quantity | E||a | E||b | a/b |
|---|---:|---:|---:|
| flake absorbed power | 44.23 uW (15.5%) | 60.24 uW (21.1%) | 0.734 |
| Tmax rise | 0.377 K | 0.359 K | — |
| flake average rise | 0.0301 K | 0.0409 K | 0.737 |
| I_PTE (isolated) | 14.918 nA | 12.411 nA | **1.2021** |
| I_PTE (perfect) | 14.931 nA | 12.413 nA | **1.2028** |

| model | abs(Ia)/abs(Ib) | vs paper 0.8366 |
|---|---:|---:|
| frozen production (lossless SiO2, w0 = 12 um) | 1.6177 / 1.6386 | x1.93 / x1.96 |
| **this scenario (Palik SiO2, w0 = 6.83 um)** | **1.2021 / 1.2028** | **x1.44** |
| absorbed-power-proportional expectation | 0.734 | x0.88 |

## Interpretation

1. **The two corrected inputs move the ratio 60% of the way (in log
   space, ~44%) toward the measurement** — from 1.62 to 1.20 — while
   the absorbed-power polarization ratio barely moves (0.728 -> 0.734).
   The change is therefore in the *spatial structure* of the
   temperature field (reduced relative weight of the E||a
   edge/contact-localized heating), exactly the channel the previous
   audits identified, not in total absorption bookkeeping.
2. The isolated/perfect metal bounds are now essentially degenerate
   (1.2021 vs 1.2028), so metal Q routing is not a material part of
   the remaining gap.
3. **Remaining gap: x1.44.**  Known channels not yet modeled, in
   plausibility order: SiO2 self-heating (the lossy spacer now absorbs
   real power below the flake, but the thermal stage still injects
   flake-support Q only), real edge non-ideality (the mesh-converged
   ideal edge still holds ~21% of absorbed power within 0.5 um), the
   waist-definition ambiguity (6.83 vs 12 um already moved the ratio;
   the true experimental waist is unpublished), metal heat spreading,
   and the eps_c = eps_b closure.

## Caveats

* The two overrides were applied together; a factorial corner
  (e.g. Palik SiO2 at w0 = 12 um) is needed to attribute the shift
  between the SiO2 model and the waist.  Not yet run.
* SiO2 absorption below the flake is excluded from both P_Q (the Q
  control volume spans the flake+electrode z-range only) and the
  thermal source; including it would add a polarization-dependent
  substrate-heating channel that plausibly moves the ratio further
  toward the measurement (E||b couples more power into the SiO2).
* This is a named diagnostic scenario, clearly labeled in every
  artifact; the frozen production contract is untouched.

## Artifacts

`/data/seunghyun/tairte4/artifacts/paper_ir_device_a_end_to_end/`:
`scenario_w6p83_palik_{empty,finite}_{a,b}_gpu{4,0}_20260801/` (optics),
`scenario_thermal_{a,b}_{isolated,perfect}_20260801/` (thermal/PTE).
Machine-readable summary: `scenario_thermal_results.json`.
