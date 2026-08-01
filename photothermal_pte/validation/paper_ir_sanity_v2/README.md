# paper_ir_sanity_v2 - offline Device-A sensitivity sanity check

Purpose: quantify, without any new FDTD or thermal solve, which modeling
choices the Device-A `|Ia|/|Ib|` paper comparison (measured
`0.8366 +/- 0.0085`, simulated `1.62-1.64`) is sensitive to.

The runner re-uses the four completed Device-A thermal/PTE artifacts and
the frozen Figure-2/3 geometry digitization.  The production integrator
(`run_device_a_explicit_thermal_pte.pte_current`), contact discretization,
and coordinate translation are imported/reproduced unchanged and are gated
by bit-exact reproduction checks (G0-G4) before any variant is evaluated.

Variants evaluated per scenario (isolated / perfect metal thermalization):

1. stored production Laplace weighting potential (paper SI Eq. S7);
2. re-solved Laplace weighting potential (reproduction gate);
3. anisotropic `div(sigma grad psi) = 0` weighting potential with the
   published `sigma_b, sigma_a = 1.10e5, 4.91e5 S/m`;
4. context row: absorbed-power-proportional ratio `P_abs,a / P_abs,b`.

Additionally, robust (max/p99/rms) polarization ratios of `|dT/da|`
(the declared paper Figure-3G comparator), `|dT/db|`, edge-normal
`|dT/dn|`, and `|grad T|` are evaluated on the digitized off-axis edge
band (0.3 / 0.5 / 1.0 um), and the Shockley-Ramo integrand is decomposed
into edge-band and remainder contributions.

Run (any host with NumPy/SciPy/Matplotlib; no Lumerical needed):

```bash
python photothermal_pte/validation/paper_ir_sanity_v2/run_device_a_offline_sensitivity_v2.py \
  --a-isolated  <...>/thermal_a_isolated_100nm_20260731_v3 \
  --b-isolated  <...>/thermal_b_isolated_100nm_20260731 \
  --a-perfect   <...>/thermal_a_perfect_100nm_20260731 \
  --b-perfect   <...>/thermal_b_perfect_100nm_20260731 \
  --output-dir  <new empty external dir for small NPZ artifacts> \
  --report-dir  photothermal_pte/reports/paper_ir_device_a_sanity_v2
```

Results: `photothermal_pte/reports/paper_ir_device_a_sanity_v2/`
(`DEVICE_A_SANITY_V2_REPORT.md` plus JSON/CSV/figures).

This check is diagnostic post-processing only.  It does not test metal
heat sinking, SiO2 IR loss, the `eps_c = eps_b` closure, or beam
assumptions; those require new solves.

## Edge optical-Q mesh convergence (follow-up, GPU)

`analyze_edge_q_mesh_convergence.py` post-processes the six
`w2edge_conv_{a,b}_xy{50,25,12p5}_dz5_t4_*_20260801` edge-isolation-smoke
runs (straight 45-deg edge, 12-um domain, uniform local mesh
50/25/12.5 nm, both polarizations) and answers the one question the v2
offline check left open: whether the E||a edge-localized Q is a mesh
artifact.  Result (see
`../../reports/paper_ir_device_a_sanity_v2/edge_q_mesh_convergence/`):
the edge hotspot is real and converged by 25 nm (25 -> 12.5 nm moves
every metric < 1 %); the 50-nm mesh overestimates only the edge *peak*
(~25 %, both polarizations alike) while band-integrated edge power is
within ~2-3 % at 50 nm; the a/b contrast is mesh-stable (1.237 -> 1.222),
so mesh refinement cannot reconcile |Ia|/|Ib| = 1.62 with the paper's
0.837.  The remaining suspects are physical-model assumptions (real edge
non-ideality, metal heat sinking, SiO2 loss, eps_c closure, beam).
