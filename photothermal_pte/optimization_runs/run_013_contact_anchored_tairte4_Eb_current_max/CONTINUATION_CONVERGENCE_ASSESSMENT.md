# Run013 E||b continuation convergence assessment

Status: `COMPLETED_BOUNDED_CONTINUATION_NOT_CERTIFIED_OPTIMUM`.

The beta schedule completed, but this is **not** a certified local or global optimum. The rapid binarization is mainly the imposed tanh projection. The final beta=64 stage ended because its three-update budget was exhausted, not because a strict gradient or KKT convergence test passed.

The feature audit also needs a discretization qualification: 500 nm was requested, while the 100 nm nodal grid rounds the 250 nm opening radius to three offsets, giving a conservative ~600 nm nominal discrete opening. Reported bad counts are design nodes.

| beta | accepted updates | gray first→last | binarization first→last | bad nodes first→last | termination |
|---:|---:|---:|---:|---:|---|
| 2 | 8 | 1.0000→1.0000 | 1.0000→0.8598 | 0→32 | bounded_stage_budget |
| 4 | 6 | 1.0000→1.0000 | 0.7045→0.4138 | 32→17 | bounded_stage_budget |
| 8 | 5 | 1.0000→0.3433 | 0.1268→0.1033 | 17→24 | bounded_stage_budget |
| 16 | 4 | 0.0766→0.1621 | 0.0336→0.0362 | 24→19 | bounded_stage_budget |
| 32 | 4 | 0.0351→0.0363 | 0.0152→0.0173 | 19→20 | bounded_stage_budget |
| 64 | 3 | 0.0194→0.0191 | 0.0088→0.0094 | 20→25 | bounded_stage_budget |
