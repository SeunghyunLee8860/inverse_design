# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=23, global iteration=23.

Actual FOM: `8.110875746278e-07 A/W`. Fixed-cap solid/void constraints: `2.710009e-02` / `1.646264e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `40`, void `5`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
