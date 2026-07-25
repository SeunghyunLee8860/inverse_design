# Latest photothermal validation status

## Finite 2 um TaIrTe4 optical Q

- Branch: `agent/validate-finite-2um-optical-q`
- Baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Status: `FIXED_DESIGN_CLOSURE_PASSED_CONVERGENCE_PENDING`
- HEAT Draft PR #2: unchanged and still blocked
- Periodic production optical path: unchanged
- New finite Q artifact validated: `false`

The actual v261 GPU solver rejected TFSF as unsupported, matching Ansys's
documented GPU limitation. The finite source was therefore changed to the
allowed Gaussian alternative. Its pre-run contract now passes with all-PML
boundaries, a 3–6 µm broadband Gaussian beam, 2 µm waist focused at the flake
center, 4 µm monitors, the 600-sample 2.7–13.2 µm material table, auto
non-uniform/CV1/accuracy-5 mesh, requested 5 nm flake dz, and GPU 0.
The zero-amplitude and matching x/y/45-degree empty-stack controls pass. With
50 nm zero padding around the finite pabs monitor, all flat polarization cases
also pass the 0.5% volume-Q versus six-face gate: x 0.3339%, y 0.3223%, and
45 degrees 0.3277%. Their unit-central-intensity absorption cross sections are
1.56568e-12, 1.78061e-12, and 1.67315e-12 m2, respectively. The next gate is
the fixed-design x closure; domain/PML/mesh/waist convergence and final
artifact validation remain pending.

The finite fixed-design x case also passes: `P_Q=2.55865e-12 W`,
`P_six=2.56276e-12 W`, closure 0.1604%, and
`sigma_abs/A_geo=0.639663`. Its design disk is the same optical SiO2 model as
the bottom spacer and is not periodically repeated.
