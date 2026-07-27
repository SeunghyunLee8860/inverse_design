# Imported-permittivity endpoint equivalence

Status: `VALIDATED_IMPORTED_PERMITTIVITY_ENDPOINT_EQUIVALENCE`

Every case uses the matched rho=0.5 checkpoint environment except for the
requested endpoint and representation: CPU TFSF, six PML, PML 32,
stabilized x/y and standard z, 7.2 µm outer x-y, identical source/Q bounds,
and central incident intensity 1 W/m².

The imported object uses exact 81×81×13 samples on x,y=[-1,1] µm and
z=[0,600] nm.  Scalar and imported endpoints are compared on common native
Yee coordinates.  Complex fields are compared without phase fitting.
Qx/Qy/Qz powers and spatial fields are retained separately; a component
below `1.0e-08` of total P_Q is reported as a low-power
diagnostic and is not allowed to dominate the major-component gate.

| kind | endpoint | representation | flake dz (nm) | promoted | P_Q diff | P_six diff | field NRMSE | spatial-Q NRMSE | worst gate |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| representation_equivalence | rho0 | scalar_vs_imported | 5.0 | False | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| representation_equivalence | rho0 | scalar_vs_imported | 2.5 | False | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| representation_equivalence | rho1 | scalar_vs_imported | 5.0 | False | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 6.026017e-17 |
| representation_equivalence | rho1 | scalar_vs_imported | 2.5 | False | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 6.138495e-17 |
| mesh_convergence_dz5_to_dz2.5 | rho0 | scalar | 5_to_2.5 | True | 2.736253e-04 | 2.695750e-04 | 4.958467e-05 | 2.361796e-04 | 2.736253e-04 |
| mesh_convergence_dz5_to_dz2.5 | rho0 | imported | 5_to_2.5 | False | 2.736253e-04 | 2.695750e-04 | 4.958467e-05 | 2.361796e-04 | 2.736253e-04 |
| mesh_convergence_dz5_to_dz2.5 | rho1 | scalar | 5_to_2.5 | False | 2.277825e-04 | 1.718300e-04 | 1.382244e-03 | 1.040646e-02 | 1.040646e-02 |
| mesh_convergence_dz2.5_to_dz1.25 | rho1 | scalar | 2.5_to_1.25 | False | 2.285453e-05 | 1.057748e-05 | 8.518324e-04 | 5.303949e-03 | 5.303949e-03 |
| mesh_convergence_dz1.25_to_dz0.625 | rho1 | scalar | 1.25_to_0.625 | True | 2.309076e-07 | 1.437005e-05 | 1.622467e-04 | 1.070271e-03 | 1.070271e-03 |
| mesh_convergence_dz5_to_dz2.5 | rho1 | imported | 5_to_2.5 | False | 2.277825e-04 | 1.718300e-04 | 1.382244e-03 | 1.040646e-02 | 1.040646e-02 |

Gate limit: `5.000000e-03`.

- Worst scalar/imported endpoint metric:
  `6.138495e-17`.
- Worst promoted finest-pair mesh metric:
  `1.070271e-03`.
- Worst recorded mesh metric including the preserved coarse failures:
  `1.040646e-02`.
- Worst imported-object bounds error:
  `0.000000e+00 m`.

The rho1 raw spatial-Q trace is explicitly retained through
5→2.5→1.25→0.625 nm. The gate uses the finest scalar pair, while endpoint
representation equivalence is independently checked at both 5 and 2.5 nm.
There is no bitwise-equality requirement. No thermal solve, adjoint,
gradient, transient, or optimization is run by this checkpoint.
