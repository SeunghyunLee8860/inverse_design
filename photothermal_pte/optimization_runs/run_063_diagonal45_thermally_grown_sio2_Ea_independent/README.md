# Run 063: +45-degree contacts, thermally grown SiO2, independent E||a

Run063 independently maximizes the signed `E||a` terminal current. It is not
a dual-polarization or soft-min optimization.

- The TaIrTe4 flake is exactly 24 um x 24 um and is physically rotated +45
  degrees in the fixed crystal frame `x=b`, `y=a`.
- Ideal low/high terminals occupy opposite diagonal edges with 2 um overlap.
  They enter only the electrical solve; Au is absent from optical and thermal
  models.
- Maxwell uses the same Run058 no-Au `E||a` optical proxy and certified
  component-Yee Jacobian as Run059 and Run062.
- Thermal and electrical tensors remain in the fixed crystal axes while the
  finite device and terminal masks are rotated.
- The TaIrTe4/SiO2 interface is `thermally_grown`, with
  `G=7.37e6 W/m2/K`.
- The initial latent density is uniform 0.5 on the 100 nm grid. NLopt LD_MMA
  uses beta continuation 1 through 128, official Ansys 500 nm DFM constraints,
  independent exact-binary repair, and fresh exact-candidate physics checks.
- The objective is signed current, not current magnitude.
- One physical GPU is exposed and nine Lumerical licenses remain reserved by
  `runres` for the entire job.

Raw solver artifacts are written to
`/data/seunghyun/tairte4/artifacts/tairte4_rotated45_edge_contact_anchored/run063_diagonal45_single_Ea_thermally_grown_v1/`.
Git-trackable checkpoints and final results are written to `results/`.
