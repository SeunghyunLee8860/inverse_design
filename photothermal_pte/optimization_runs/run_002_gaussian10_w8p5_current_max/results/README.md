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
selected production optical-chain certificate records the actual 18.6 µm,
373×373 GPU forward and component-specific density-to-Yee Jacobians. It does
not promote the earlier 201×201 thermal/combined smoke to the selected grid.
The
selected production thermal-mapping certificate records exact bilinear
373-node-to-186-cell density transfer, its transpose, selected-support literal
material-Q attribution, and conservative deposition to the explicit 3D grid.
The
selected thermal gray-law report validates the fixed-Q thermal material
derivative for p=1,2,3 and the evaporated endpoint using CUDA-only linear
solves. It explicitly does not certify the optical epsilon gray-law branch.
The
analytic setup audit is produced in memory by `run_optimization.py
--setup-audit`.

The selected-grid optical-gradient certificate is published in
`SELECTED_OPTICAL_GRADIENT_ADFD_REPORT.md`. It preserves the earlier `5.326%`
failure as a diagnostic and records the corrected scalar-PQ, spatially
weighted optical-PTE, and one-direction combined AD-FD errors. The correction
uses the FieldRegion-only CW spectrum reconstructed from the two official
normalization states; it does not fit finite differences or rescale a
gradient. Broader directions and full latent AD-FD remain pending.

All referenced raw FSP/NPZ/sparse-matrix paths, byte sizes, and SHA-256 values
are recorded in the manifest. Raw solver artifacts remain outside Git.
