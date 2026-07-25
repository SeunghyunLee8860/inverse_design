# Latest photothermal validation status

## Finite 2 um TaIrTe4 optical Q

- Branch: `agent/validate-finite-2um-optical-q`
- Baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Status: `SOURCE_CONTROL_PASSED_FLAT_CLOSURE_REFINEMENT`
- HEAT Draft PR #2: unchanged and still blocked
- Periodic production optical path: unchanged
- New finite Q artifact validated: `false`

The actual v261 GPU solver rejected TFSF as unsupported, matching Ansys's
documented GPU limitation. The finite source was therefore changed to the
allowed Gaussian alternative. Its pre-run contract now passes with all-PML
boundaries, a 3–6 µm broadband Gaussian beam, 2 µm waist focused at the flake
center, 4 µm monitors, the 600-sample 2.7–13.2 µm material table, auto
non-uniform/CV1/accuracy-5 mesh, requested 5 nm flake dz, and GPU 0.
The zero-amplitude control passed with exactly zero recorded field. The 6.8 µm
aperture empty-stack control passed: zero volume Q, 0.0330% closed-box residual,
opposite lateral-face asymmetry below 2.1e-8, and 4.316% edge/central
intensity. The first flat-x run generated component Q but failed the closure
gate at 2.431%. The pabs volume monitor is being given an explicit 50 nm zero
padding on all finite sidewalls so conformal boundary loss is not cut at the
monitor edge; the physical flake bounds remain exact metadata.
