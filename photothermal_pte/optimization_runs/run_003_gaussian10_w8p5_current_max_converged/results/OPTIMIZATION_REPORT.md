# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=22, global iteration=22.

Actual FOM: `7.903895280073e-07 A/W`. Fixed-cap solid/void constraints: `2.885527e-02` / `1.915854e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `39`, void `8`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
