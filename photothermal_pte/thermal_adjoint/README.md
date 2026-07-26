# Thermal/PTE adjoint

This package implements and certifies the conservative

`Q -> temperature rise -> local PTE surrogate`

part of the inverse-design chain. It reuses the validated Cartesian FVM
assembly under `photothermal_pte/validation/photothermal_stage1` and must not
contain a second thermal face-conductance implementation.

The first executable certificate is deliberately solver-free and uses named
numerical material/interface scenarios. It verifies matrix assembly,
functional transposes, and AD–FD identities. It is not a final experimental
temperature or electrode-current prediction.

The governing contract and the later Maxwell chain are documented in
`photothermal_pte/reports/inverse_design_pte_adfd/INVERSE_DESIGN_PTE_ADFD_CONTRACT.md`.

