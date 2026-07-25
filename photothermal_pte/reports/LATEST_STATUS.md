# Latest photothermal validation status

## Finite 2 um TaIrTe4 optical Q

- Branch: `agent/validate-finite-2um-optical-q`
- Baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Status: `GAUSSIAN_GPU_CONTRACT_PASSED_SOURCE_CONTROL_PENDING`
- HEAT Draft PR #2: unchanged and still blocked
- Periodic production optical path: unchanged
- New finite Q artifact validated: `false`

The actual v261 GPU solver rejected TFSF as unsupported, matching Ansys's
documented GPU limitation. The finite source was therefore changed to the
allowed Gaussian alternative. Its pre-run contract now passes with all-PML
boundaries, a 3–6 µm broadband Gaussian beam, 2 µm waist focused at the flake
center, 4 µm monitors, the 600-sample 2.7–13.2 µm material table, auto
non-uniform/CV1/accuracy-5 mesh, requested 5 nm flake dz, and GPU 0. The next
gate is the zero-amplitude and empty-layered-stack source control.
