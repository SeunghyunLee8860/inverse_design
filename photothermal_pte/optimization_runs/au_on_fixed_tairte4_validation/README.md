# Au topology validation on a fixed TaIrTe4 flake

This folder is intentionally separate from the completed TaIrTe4/void
optimization runs.  It validates a new physical contract:

- fixed TaIrTe4 flake;
- fixed terminal/electrode locations;
- a separate Au/air topology layer above the flake;
- direct Au/TaIrTe4 electrical and thermal coupling will be added only after
  the optical metal endpoint and adjoint are certified.

No previous Run 040–058 artifact is modified by this work.

## Current checkpoint

The first checkpoint freezes the Au endpoint at 10 um and audits two density
paths.  The production candidate is the nonlinear plasmonic interpolation

```text
n(rho) = (1-rho) n_air + rho n_Au
epsilon(rho) = n(rho)^2
```

with `n_Au = 12.1 + 69.2i` from the exact 10-um row of Ordal et al.  The
linear-complex-epsilon law is retained only as a failure/diagnostic control.
Gray density is not interpreted as a physical Au/air effective medium.

Run the offline checkpoint with the environment that contains NumPy and
Matplotlib:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/01_audit_au_material_and_density_path.py
```

Run tests:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python -m pytest -q \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/tests
```

The next checkpoint opens a v261 design session but performs no Maxwell solve
and acquires no GPU engine:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/02_probe_lumerical_au_readback.py
```

The retained first attempt is currently fail-closed at
`BLOCKED_LUMERICAL_LICENSE_SESSION_STARTUP`: both installed v261 API roots
failed before material import because ANSYSLI did not create/read its shared
port file.  This is not an Au readback failure, and no GPU solve was launched.
Regenerate the consolidated report and manifest with:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/03_summarize_checkpoint.py
```

## Fail-closed sequence

1. Au material source and nonlinear-density-path audit.
2. Lumerical requested/fitted material readback at 10 um.
3. Binary air/uniform-Au/stripe/island optical controls and mesh convergence.
4. Nonuniform Au density-to-component-Yee Jacobian and optical AD-FD.
5. Explicit Au/TaIrTe4 thermal contact controls and thermal-only AD-FD.
6. Two-layer TaIrTe4/Au weighting-potential controls and electrical AD-FD.
7. Combined physical-density and latent/filter/projection AD-FD.
8. Optimization followed by exact-binary Au/air reevaluation.

If the density route fails material readback, binary equivalence, or AD-FD,
the approved fallback is sharp-interface level-set/shape optimization.

## Material provenance

- Au optical `n,k`: [Ordal et al., Applied Optics 26, 744–752 (1987)](https://doi.org/10.1364/AO.26.000744).
- CC0 data transcription: [refractiveindex.info database](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml).
- Nonlinear interpolation: [Zeng, Venuthurumilli, and Xu, ACS Photonics (2021)](https://doi.org/10.1021/acsphotonics.1c00260).
- Bulk Au thermal reference: [NIST resistivity compilation](https://srd.nist.gov/JPCRD/jpcrd155.pdf).

Bulk transport values are references rather than certified thin-film values.
Film thickness, deposition, grain size, and Au/TaIrTe4 electrical/thermal
contacts remain explicit physical uncertainties.
