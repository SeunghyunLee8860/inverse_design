# Run012 E||a continuation convergence assessment

Status: `COMPLETED_BOUNDED_CONTINUATION_NOT_CERTIFIED_OPTIMUM`.

The beta schedule completed, but this is **not** a certified local or global optimum. The rapid binarization is mainly the imposed tanh projection. The final beta=64 stage ended because its three-update budget was exhausted, not because a strict gradient or KKT convergence test passed.

The feature audit also needs a discretization qualification: 500 nm was requested, while the 100 nm nodal grid rounds the 250 nm opening radius to three offsets, giving a conservative ~600 nm nominal discrete opening. Reported bad counts are design nodes.

| beta | accepted updates | gray first→last | binarization first→last | bad nodes first→last | termination |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 1.0000→1.0000 | 1.0000→0.8588 | 0→33 | bounded_stage_budget |
| 4 | 6 | 1.0000→1.0000 | 0.7026→0.4483 | 33→48 | bounded_stage_budget |
| 8 | 5 | 1.0000→0.3753 | 0.1689→0.1415 | 48→82 | bounded_stage_budget |
| 16 | 4 | 0.1293→0.1608 | 0.0674→0.0567 | 82→76 | bounded_stage_budget |
| 32 | 4 | 0.0647→0.0561 | 0.0247→0.0241 | 76→73 | bounded_stage_budget |
| 64 | 3 | 0.0270→0.0284 | 0.0117→0.0125 | 73→89 | bounded_stage_budget |
