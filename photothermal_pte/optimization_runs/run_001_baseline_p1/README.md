# run_001_baseline_p1

First planned inverse-design run using the certified finite nonperiodic AD--FD
contract and the linear (`p=1`) gray-material relaxation.

This is a **prepared baseline, not an optimization result**.  The run is held
at `PLANNED` because the iterative update rule, constraint and stopping policy
have not yet been reviewed.  No solver was launched by this commit.

Before execution, add the production forward FSP and component-wise Yee
Jacobian directory to `external_inputs`, pin every SHA-256, select the optimizer
and volume constraint, then run the fail-closed preflight.
