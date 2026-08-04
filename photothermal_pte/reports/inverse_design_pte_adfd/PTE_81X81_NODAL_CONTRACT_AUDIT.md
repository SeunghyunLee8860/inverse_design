# PTE discrete operator and 81x81 nodal contract audit

Status: `AUDITED_PTE_DISCRETE_OPERATOR_AND_81X81_NODAL_CONTRACT`

- physical-density coordinates: 81x81 nodes on exact
  `[-1,1] um x [-1,1] um` support;
- spacing: `25 nm`; opposite-edge wrap: absent;
- design is a 2D density extruded from `z=0` to `600 nm`;
- PTE weighting:
  `dpsi/dx=dpsi/dy=1/(4 um)`;
- periodic derivatives: absent;
- forward functional and `c_T` use the same sparse `D_x,D_y` matrices.

Errors:

- affine analytic control:
  `0.00000000e+00`;
- affine forward/source identity:
  `0.00000000e+00`;
- random forward/source identity:
  `0.00000000e+00`;
- temperature-source centered FD:
  `5.42830565e-11`.

This is a uniform-45-degree PTE surrogate, not a solved finite-contact
terminal current. No Maxwell solve, thermal solve, or optimization was run.
