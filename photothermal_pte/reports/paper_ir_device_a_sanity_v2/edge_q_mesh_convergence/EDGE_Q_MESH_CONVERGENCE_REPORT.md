# Edge optical-Q mesh-convergence verification (straight 45-deg edge)

Status: `COMPLETED_EDGE_Q_MESH_CONVERGENCE_ANALYSIS`

Purpose: the v2 offline sensitivity check reduced the Device-A
`|Ia|/|Ib| = 1.62` vs paper `0.8366` disagreement to a single remaining
suspect - the E||a edge-localized optical absorption Q.  The production
runs could not be mesh-refined at the tilted edge (the 60-um domain with
edge-local 12.5-nm overrides needs ~94 GiB > 48 GiB GPU), so this check
isolates the edge physics on a small domain and refines the mesh
uniformly until the edge absorption converges.

## Contract

Six new GPU FDTD runs (frozen production runner, `edge-isolation-smoke`
execution contract, no clipping/gain/rescaling): straight 45-deg TaIrTe4
edge (flake y <= x) on SiO2/Si, 12-um domain, nominal w0 = 2 um Gaussian
at the edge, lambda = 11 um, flake dz = 5 nm, polarization E||a and E||b,
uniform local x/y mesh 50 / 25 / 12.5 nm.  Every run passed all
acceptance gates except the documented auto-shutoff floor
(`final_value` 4.13e-5 - 4.23e-5 > 1e-5, time-independent: identical at
4 ps and 10 ps, the same kind of floor the accepted w2 diagnostics
recorded); the six-face energy closures passed at 0.03 - 0.25 %.

Artifacts: `/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/`
`w2edge_conv_{a,b}_xy{50,25,12p5}_dz5_t4_gpu{4,0}_20260801/`.
Analysis runner: `photothermal_pte/validation/paper_ir_sanity_v2/`
`analyze_edge_q_mesh_convergence.py` (pure post-processing).

## Results

Edge-normal areal absorption profiles (z- and tangent-integrated Q per
source watt, 50-nm bins) and integral metrics versus mesh:

| metric | pol | 50 nm | 25 nm | 12.5 nm | rel. change 50->25 | 25->12.5 |
|---|---|---:|---:|---:|---:|---:|
| edge peak (1/m^2) | a | 3.089e9 | 2.309e9 | 2.295e9 | 25.3 % | **0.60 %** |
| edge peak (1/m^2) | b | 3.052e9 | 2.349e9 | 2.328e9 | 23.0 % | **0.88 %** |
| peak / interior plateau | a | 2.659 | 1.952 | 1.966 | 26.6 % | 0.71 % |
| peak / interior plateau | b | 1.910 | 1.448 | 1.455 | 24.2 % | 0.47 % |
| edge band (+/-0.5 um) fraction | a | 0.2213 | 0.2168 | 0.2146 | 2.0 % | 1.0 % |
| edge band (+/-0.5 um) fraction | b | 0.1789 | 0.1767 | 0.1756 | 1.2 % | 0.64 % |
| absorbed fraction of source | a | 0.04009 | 0.03983 | 0.03972 | 0.64 % | 0.30 % |
| absorbed fraction of source | b | 0.04459 | 0.04419 | 0.04399 | 0.91 % | 0.46 % |

Polarization contrast versus mesh (the quantity that would have to move
by ~2x to explain the paper disagreement):

| a/b contrast | 50 nm | 25 nm | 12.5 nm |
|---|---:|---:|---:|
| edge-band (+/-0.5 um) fraction | 1.2370 | 1.2270 | 1.2223 |
| edge peak / plateau | 1.3922 | 1.3478 | 1.3511 |
| total absorbed fraction | 0.8990 | 0.9015 | 0.9029 |

## Findings

1. **The edge hotspot is real physics, not a staircase artifact.**  After
   a 4x lateral refinement the edge enhancement survives and converges:
   peak-over-plateau -> 1.97 (E||a) and 1.46 (E||b); the 25 -> 12.5 nm
   step changes every metric by < 1 %.
2. **The 50-nm production-style mesh does overestimate the edge peak by
   ~25 %, but symmetrically in polarization and only in the peak.**  The
   band-integrated edge power - the quantity the thermal solve and the
   Shockley-Ramo integral actually consume - is already within ~2-3 % of
   its converged value at 50 nm (first-order convergence, ratio ~0.5).
3. **Mesh refinement cannot reconcile the sanity check with the paper.**
   The a/b edge-band contrast moves from 1.237 to 1.222 (-1.2 %) and the
   total-absorption contrast from 0.899 to 0.903 (+0.4 %) over the full
   4x refinement.  Nothing in the numerics trends toward the ~2x change
   required to bring `|Ia|/|Ib| = 1.62` down to 0.837.

## Verdict and remaining suspects

The FDTD edge optical Q is now numerically certified at the integrated
level.  Combined with the v2 offline result (post-processing bit-exact,
weighting-model correction moves the ratio the wrong way, all gradient
metrics a-dominant at the Device-A position), the paper-vs-simulation
disagreement is attributable to *physical modeling assumptions*, not to
solver numerics.  Remaining candidates, in rough order of plausibility:

* ideal sharp vertical edge vs the real device edge (roughness,
  oxidation, non-vertical sidewall, contamination) - the converged
  simulation still concentrates ~21 % of the absorbed power within
  0.5 um of an atomically ideal edge;
* Au/Ti electrodes excluded from the thermal domain (isolated vs perfect
  bounds did not bracket the measurement, but partial metal heat
  spreading changes the T field shape);
* lossless SiO2 closure at lambda = 11 um (real SiO2 is strongly lossy
  near the 9-um phonon band tail);
* the epsilon_c = epsilon_b closure and the scalar-Gaussian beam
  realization.

Caveats: the smoke geometry (12-um domain, w0 = 2 um realized larger,
45-deg edge) is a controlled proxy, not the Device-A production geometry
(60-um domain, w0 = 12 um, 53.6-deg digitized edge); conclusions
transfer at the mechanism level (edge-Q convergence behavior), not as
absolute numbers.
