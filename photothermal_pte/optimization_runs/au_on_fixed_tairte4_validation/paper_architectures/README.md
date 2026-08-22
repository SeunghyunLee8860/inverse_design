# Paper-like Au architectures with a fixed TaIrTe4 active layer

This folder separates three structures that must not be mixed:

1. `A_DIRECT_AU_TAIRTE4`: the project's simple floating-Au-on-TaIrTe4
   control. It has no opaque backplane, so its SiO2/Si optical substrate
   cannot be silently removed.
2. `B_T_2024_TAIRTE4_SUBSTITUTION`: the 2024 inverse-T
   metamaterial-perfect-absorber stack. The top Ti/Au T touches the active
   layer; the Al2O3 spacer is **below** the active layer and above the Au
   mirror.
3. `B_Z_2022_TAIRTE4_SUBSTITUTION`: the 2022 chiral Z stack. The Cr/Au
   antenna chip is fabricated first and the 2-D thermoelectric material is
   dry-transferred over the patterned topography.

Only the active 2-D material is replaced by the fixed 100-nm TaIrTe4 flake.
Unknown TaIrTe4 contact/topography properties and all extrapolated 10-um
dimensions remain named scenarios.

Run the offline contract audit:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/paper_architectures/01_audit_and_plot_contracts.py
```

Run tests:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python -m pytest -q \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/paper_architectures/tests
```

The architecture contract is offline, and the substrate discriminator has now
also been run with v261 GPU FDTD.  See
`results/SUBSTRATE_REDUCTION_DECISION.md`.

The GPU-only discriminator is `02_run_v261_backplane_truncation_control.py`.
Run a `full` and an `au_truncated` case into raw directories outside Git, then
compare them with `03_summarize_backplane_truncation_control.py`.  A failed or
missing run is never interpreted as evidence that the substrate is removable.

The substrate rules are deliberately asymmetric:

- the 2022 paper's published FDTD stack explicitly contains Si, 285-nm
  thermal SiO2, the 200-nm Au backplate, Al2O3, and the Au antenna.  That is
  the paper-reference model;
- the 2024 main Methods state 1.5-um thermal SiO2, whereas Supplementary
  Fig. 17's RF cross-section states 1.0 um.  These are separate provenance
  scenarios, not a value to average;
- omitting everything below an opaque Au backplane is an accelerated optical
  candidate only.  It must pass the numerical backplane-truncation gate;
- a one-substrate thermal model is not implied by optical opacity.  It needs
  a separate explicit-3D versus reduced-impedance thermal comparison.

Published numerical outcome:

- 2022 Z, full 285-nm SiO2/Si versus Au-truncated: absorbed-flux difference
  `0.005404%`, top-field NRMSE `0.000359%`, full-stack transmission
  `1.181e-9`;
- 2024 T main 1.5-um SiO2 scenario versus Au-truncated: absorbed-flux
  difference `0.001259%`, top-field NRMSE `0.000399%`, full-stack
  transmission `5.382e-10`.

The strict periodic `pabs_adv` volume-Q versus flux closure remains
`2.54--2.56%` and is kept fail-closed.  The field/flux result therefore
certifies optical insensitivity below the opaque backplane, not an absolute-Q
closure and not a thermal substrate reduction.

Reproduce the published decision after the paired raw cases exist:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/paper_architectures/04_publish_substrate_reduction_decision.py
```

## Actual inverse-T optical smoke

`05_actual_metasurface_geometry.py` and
`07_run_v261_t2024_tairte4_optical_smoke.py` implement the 2024 MIR inverse-T
scenario with only the active 2-D material replaced by fixed 100-nm TaIrTe4.
The T vertices are a figure-digitized approximation to Supplementary Fig. 14,
not author CAD. Both normal-incidence polarizations have been run on the v261
GPU solver and pass closure, shutoff, finite-Q, and nonnegative-Q gates.

The production default is now the explicit paper substrate:

`air / Au inverse-T / 100-nm TaIrTe4 / 35-nm Al2O3 / 200-nm Au mirror /
1.5-um thermally grown SiO2 / intrinsic Si`.

At 4.75 um, SiO2 and Si are read directly from the installed Lumerical v261
Palik database and their complex-index readbacks are stored in every raw case
JSON. The legacy `Au-to-bottom-PML` model remains available only through the
explicit `--substrate-mode au_truncated` diagnostic option.

Publish the paired comparison without rerunning FDTD:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/paper_architectures/09_compare_t2024_tairte4_polarizations.py
```

See `results_actual_metasurfaces/T2024_TAIRTE4_TWO_POLARIZATION_REPORT.md`.
The 2022 M5 scalar dimensions are audited, but the Z Maxwell case remains
fail-closed because the PDFs do not disclose a unique polygon/junction CAD.

The inverse-T contribution is isolated with matched no-top-T controls using
the same solver entry point plus `--omit-top-t-control`. Publish the four-case
comparison with `10_summarize_t2024_top_t_enhancement.py`.

The explicit-substrate four-case results and direct comparison against the
legacy optical truncation are in `results_actual_metasurfaces_sio2_si/`.
Reproduce the offline equivalence check with:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/paper_architectures/14_compare_t2024_explicit_substrate.py
```

This comparison modifies no Q array. Components below a documented `1e-8`
native-power fraction are reported as numerical zero and are not normalized
for a spatial-shape metric, because doing so would amplify roundoff. All
active-component lateral-Q, total-power, and TaIrTe4-power equivalence gates
pass by a wide margin. Optical equivalence below an opaque Au mirror does not
remove SiO2 or Si from the later thermal model.

For presentations, run `11_publish_meeting_plot_package.py`. It creates one
folder per case with separate structure, Qx, Qy, Qz, conservative common-grid
Qtotal, power breakdown, overview, and machine-readable metrics.

`12_publish_detailed_meeting_materials.py` adds xy/xz/yz setup and Q sections,
top-monitor total fields, absorption-depth profiles, geometric material maps,
T-vs-bare/polarization comparisons, a CSV table, and a meeting Q&A guide.

`13_publish_periodic_and_finite_scope_materials.py` preserves the raw Yee
endpoint seam as a diagnostic while publishing a display-only periodic
canonical cell and 3x3 tiling. It also separates the paper's periodic optical
screening problem from the finite all-PML Maxwell plus thermal/electrical PTE
device that is required for terminal-current inverse design.
