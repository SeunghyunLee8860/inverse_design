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

The initial sandboxed attempt failed before material import, but the normal
host session subsequently passed as `VALIDATED_LUMERICAL_AU_MATERIAL_READBACK`.
The exact 10-um `(n,k)` material passed the complex-epsilon fit gate; the
global full-table Ordal fit did not and remains diagnostic only. No GPU solve
was launched. Regenerate the consolidated report and manifest with:

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

The first binary representation control is deliberately smaller than the
device. It compares a 50-nm finite Au film represented by an exact scalar
`(n,k)` material and by uniform `importnk2` under the same finite Gaussian,
six-PML, native-Yee-Q contract. Runsetup is audited before any GPU forward:

```bash
python 04_run_au_binary_representation_control.py \
  --output-dir /path/to/raw_case --rho 1 --representation scalar \
  --gpu-device 'GPU 6' --contract-only
```

The completed binary checkpoint found that exact scalar Au is stable and
closes its 20-um control volume, while the identical uniform `rho=1`
`importnk2` representation diverges. Therefore no gray-density or density
AD-FD test is promoted. The workflow now follows the approved fallback:
sharp-interface binary Au with level-set/shape derivatives.

The first sharp-interface control moves the two x-normal faces of the exact
scalar-Au film while keeping the source, mesh, monitors, material and all
other faces fixed. It is a forward central-FD geometry control, not yet an
adjoint certificate:

```bash
python 06_run_au_sharp_interface_width_control.py \
  --au-half-x-um 10.0 --output-dir /path/to/raw_case \
  --gpu-device 'GPU 6'
```

No gray Au/air cell is introduced by this route. The next gate compares a
mesh-aware central-FD plateau with the bundled v261 polygon boundary
perturbation adjoint before this representation is allowed into the coupled
thermal/electrical model.

Audit the exact bundled v261 source path without opening Lumerical:

```bash
python 07_audit_v261_sharp_interface_adjoint_path.py
```

This audit only establishes which boundary formula and polygon contract are
installed. It intentionally leaves the numerical AD--FD status pending.

Summarize the sharp-interface forward-FD controls:

```bash
python 08_summarize_au_sharp_interface_width_controls.py
```

At the current 100 nm lateral edge mesh, the `h=0.20` and `0.10 um` central
differences agree to about 1.16%, while `h=0.05 um` changes by about 30%.
Therefore this checkpoint remains fail-closed: the exact-binary Au route is
retained, but a numerical shape-adjoint is not promoted until an edge-local
50 nm mesh produces a forward-FD plateau.

Edge-local refinement is controlled independently from the 100 nm interior
mesh:

```bash
python 06_run_au_sharp_interface_width_control.py \
  --au-half-x-um 8.0 --edge-dxy-nm 25 --edge-band-um 0.5 \
  --output-dir /path/to/raw_case --gpu-device 'GPU 6'
python 09_summarize_au_sharp_interface_mesh_refinement.py
```

The 25 nm edge mesh gives a 0.716% difference between `h=0.10` and `0.05 um`,
so the within-mesh FD-step plateau passes. The `h=0.10 um` derivative still
changes by about 3.04% from edge-50 to edge-25 nm; mesh-independent shape
sensitivity and the numerical boundary adjoint therefore remain unvalidated.

## Material provenance

- Au optical `n,k`: [Ordal et al., Applied Optics 26, 744–752 (1987)](https://doi.org/10.1364/AO.26.000744).
- CC0 data transcription: [refractiveindex.info database](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml).
- Nonlinear interpolation: [Zeng, Venuthurumilli, and Xu, ACS Photonics (2021)](https://doi.org/10.1021/acsphotonics.1c00260).
- Bulk Au thermal reference: [NIST resistivity compilation](https://srd.nist.gov/JPCRD/jpcrd155.pdf).

Bulk transport values are references rather than certified thin-film values.
Film thickness, deposition, grain size, and Au/TaIrTe4 electrical/thermal
contacts remain explicit physical uncertainties.
