# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=41, global iteration=41.

Actual FOM: `8.756432067811e-07 A/W`. Fixed-cap solid/void constraints: `1.450982e-02` / `7.127759e-03` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `47`, void `124`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
