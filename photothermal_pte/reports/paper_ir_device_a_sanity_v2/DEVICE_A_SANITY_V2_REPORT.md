# Device A offline sensitivity sanity check v2

Status: `COMPLETED_DEVICE_A_OFFLINE_SENSITIVITY_V2`

This is a pure post-processing sensitivity check on the four completed
Device-A thermal/PTE artifacts (isolated/perfect x E||a/E||b).  No new FDTD,
thermal solve, clipping, gain, rescaling, or polarization-dependent
treatment was used.  The production `pte_current` integrator, contact
discretization, and frozen Figure-2/3 geometry digitization were reused
unchanged; only the weighting-potential input varies between variants.

## Reproduction gates (all pass)

| gate | value |
|---|---:|
| G0 shared grid/mask across all four artifacts | `true` |
| G1 stored-integrand volume-sum relative error (worst) | `3.31e-16` |
| G2 production-integrator current reproduction (worst) | `0.0` (bit-exact) |
| G3 Laplace weighting-potential re-solve vs stored, max abs diff | `0.0` (bit-exact) |
| G4 unit-conductivity identity vs Laplace, max abs diff | `0.0` |

G3 also certifies that the vendored coordinate translation and the frozen
digitized contact segments reproduce the production weighting contract
exactly.

## Terminal-current ratio matrix

| psi variant | scenario | I_a (A) | I_b (A) | abs(Ia)/abs(Ib) |
|---|---|---:|---:|---:|
| stored Laplace (paper Eq. S7) | isolated | 8.072622430e-09 | 4.990319469e-09 | **1.617656** |
| stored Laplace (paper Eq. S7) | perfect | 8.196197440e-09 | 5.001980470e-09 | **1.638590** |
| re-solved Laplace | isolated | 8.072622430e-09 | 4.990319469e-09 | 1.617656 |
| re-solved Laplace | perfect | 8.196197440e-09 | 5.001980470e-09 | 1.638590 |
| sigma-weighted div(sigma grad psi)=0 | isolated | 1.101780924e-08 | 3.075756488e-09 | **3.582146** |
| sigma-weighted div(sigma grad psi)=0 | perfect | 1.128493178e-08 | 3.098277476e-09 | **3.642324** |
| absorbed-power-proportional expectation | isolated | - | - | **0.728066** |
| absorbed-power-proportional expectation | perfect | - | - | **0.732007** |
| digitized paper Figure 3I/J | - | - | - | **0.836590 +/- 0.008526** |

## Findings

1. **Baseline fidelity.** The published end-to-end ratios 1.617656 /
   1.638590 are reproduced bit-exactly from the stored temperature fields
   with the unmodified production integrator and a bit-exact re-solve of the
   production Laplace weighting potential.  The disagreement with the paper
   is therefore not a bookkeeping or solver-reproduction issue.
2. **The weighting model is a huge sensitivity, but its physical correction
   moves the ratio away from the paper.**  Replacing the paper's isotropic
   Laplace Eq. S7 with the physically motivated anisotropic operator
   `div(sigma grad psi)=0` (`sigma_b/sigma_a = 1.10e5/4.91e5 S/m`, contact
   attachment through y-normal faces) changes the ratio from 1.62 to 3.58
   (2.2x).  This quantifies the structural hypersensitivity of the terminal
   ratio to the weighting/contact model in the near-cancelling
   `|sigma_a S_a| ~ |sigma_b S_b|` regime, and it rules out the Laplace
   choice as the missing correction toward the measurement.
3. **The absorbed-power-proportional expectation is close to the paper.**
   `P_abs,a/P_abs,b = 0.728 / 0.732` versus the digitized `0.8366`.  The
   measured device behaves approximately like a total-absorbed-power
   detector at this position, whereas the simulated spatially structured
   `grad T . grad psi` integral - dominated by edge/contact-localized E||a
   heating - drives the ratio above 1.6.
4. **At this Device-A position the a-dominance is metric-robust.**  On the
   digitized off-axis edge band (+/-0.5 um), the declared paper comparator
   |dT/da| gives a/b = 2.71 (max), 3.54 (p99), 2.93 (rms); the edge-normal
   |dT/dn| gives 3.89 / 3.51 / 2.30; |grad T| gives 2.57 / 3.51 / 2.23.
   Unlike the W12 straight-edge control (where the declared comparator sat
   at about 1.00), the Device-A disagreement at this position is present in
   every gradient metric and therefore lives in the temperature field
   itself, not in the metric choice.
5. **Edge-band integrand decomposition (+/-0.5 um, Laplace psi).**
   E||a: +7.147e-10 A from the band, +7.358e-09 A remainder.
   E||b: -4.916e-10 A from the band, +5.482e-09 A remainder.
   The band contributes with opposite signs, exactly the edge-hotspot
   versus interior-heating signature.  Note the remainder alone still gives
   a/b = 1.34; because the temperature response to the edge-localized E||a
   Q is nonlocal, this decomposition localizes the integrand, not the
   causal Q attribution.

## What this check does and does not test

Tested offline: reproduction fidelity, weighting-operator sensitivity,
comparator-metric robustness, edge-band integrand structure, and the
absorbed-power-proportional context ratio.

Not tested (each requires new solves and remains open):
edge-mesh convergence of the E||a edge-localized optical Q, metal (Au/Ti)
heat spreading/sinking in the thermal domain, lossy/dispersive SiO2 at
11 um, the eps_c = eps_b closure, beam waist/position sensitivity, and the
paper's exact Figure-3 scan geometry.

## Inputs and provenance

- Artifacts (external, SHA-256 recorded in
  `device_a_sanity_v2_summary.json`):
  `thermal_{a,b}_{isolated,perfect}_100nm_20260731*/thermal_pte_fields.npz`
- Geometry contract:
  `photothermal_pte/reports/paper_ir_device_a_end_to_end/device_a_geometry_digitization.json`
- Runner:
  `photothermal_pte/validation/paper_ir_sanity_v2/run_device_a_offline_sensitivity_v2.py`
- Machine-readable results: `device_a_sanity_v2_summary.json`,
  `device_a_sanity_v2_ratio_matrix.csv`
- Figures: `V2_RATIO_MATRIX.png`, `V2_WEIGHTING_POTENTIALS.png`,
  `V2_EDGE_COMPARATOR_METRICS.png`
