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

The current checkpoint is offline only. The next numerical gate must compare
explicit and optically truncated backplane domains before the shortened model
is used in Maxwell AD--FD or optimization.

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
