# Run065: +45-degree rotated-Q dual optimization

- One shared structure maximizes the signed-current soft minimum of `E||a` and `E||b`.
- Maxwell forward/adjoint uses the validated Run58-style GPU setup with no optical Au.
- Each native `Qx/Qy/Qz` field is explicitly rotated +45 degrees before thermal deposition.
- The optical adjoint uses the exact transpose of the same sparse rotation operator.
- The thermal interface is thermally grown SiO2 (`G = 7.37 MW m^-2 K^-1`).
- Optimization starts from a fresh uniform `rho=0.5` latent field. No Run062
  structure, objective, gradient, or cache is reused.
