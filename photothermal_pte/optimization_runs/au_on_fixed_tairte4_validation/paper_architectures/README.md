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
