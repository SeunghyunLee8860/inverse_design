# Latest photothermal validation status

## Finite 2 um TaIrTe4 optical Q

- Branch: `agent/validate-finite-2um-optical-q`
- Baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Status: `GAUSSIAN_SOURCE_APERTURE_CONTROL_REFINEMENT`
- HEAT Draft PR #2: unchanged and still blocked
- Periodic production optical path: unchanged
- New finite Q artifact validated: `false`

The actual v261 GPU solver rejected TFSF as unsupported, matching Ansys's
documented GPU limitation. The finite source was therefore changed to the
allowed Gaussian alternative. Its pre-run contract now passes with all-PML
boundaries, a 3–6 µm broadband Gaussian beam, 2 µm waist focused at the flake
center, 4 µm monitors, the 600-sample 2.7–13.2 µm material table, auto
non-uniform/CV1/accuracy-5 mesh, requested 5 nm flake dz, and GPU 0.
The zero-amplitude control passed with exactly zero recorded field. The first
empty-stack run had zero volume Q but failed the source-aperture gate
(edge/central intensity 7.34%). The aperture is being increased from 6.0 to
6.8 µm. Finite Gaussian diffraction is reported explicitly; its lateral flux
gate tests opposite-face symmetry and empty-box energy residual rather than
incorrectly forcing physical diffraction to zero.
