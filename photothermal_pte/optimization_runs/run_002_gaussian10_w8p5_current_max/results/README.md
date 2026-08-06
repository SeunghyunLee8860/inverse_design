# Results

The homogeneous-air source-only result, uniform complex-material equivalence,
nonuniform component-Yee Jacobian smoke, production-candidate geometry and
GPU forward gates, the 201×201 production component-Yee Jacobian, and small
literal-FVM CUDA forward/adjoint control are recorded here. The literal
material-intersection Q attribution is also recorded without nearest-material
relocation or global rescaling, together with its exact 3D thermal-grid
deposition. The first production combined physical-rho PTE gradient smoke is
now recorded separately; it is one direction and one FD step, not a full
latent or optimization certificate. The
offline production-window certificate freezes a centered 18.6×18.6 µm,
373×373-node window from the SHA-pinned combined gradient; it runs no solver
or optimizer. The
finite production filter/projection certificate validates the 373×373 mapping
and exact transpose without a solver; exact-binary DRC and full latent AD-FD
remain pending. The
analytic setup audit is produced in memory by `run_optimization.py
--setup-audit`.

All referenced raw FSP/NPZ/sparse-matrix paths, byte sizes, and SHA-256 values
are recorded in the manifest. Raw solver artifacts remain outside Git.
