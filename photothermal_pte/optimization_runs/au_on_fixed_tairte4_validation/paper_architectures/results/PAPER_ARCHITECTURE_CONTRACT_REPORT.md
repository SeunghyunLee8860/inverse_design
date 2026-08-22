# Paper-derived Au/TaIrTe4 architecture contract

Status: `VALIDATED_PAPER_ARCHITECTURE_CONTRACT_OFFLINE`

This checkpoint reads the official 2024 and 2022 supplements and changes only
the active 2-D thermoelectric material to the project's fixed 100-nm TaIrTe4
flake. It does **not** claim a Maxwell, thermal, PTE, adjoint, or optimization
result.

## Substrate decision

- `A_DIRECT_AU_TAIRTE4`: SiO2/Si cannot be removed optically without an
  endpoint-equivalence test because there is no opaque Au mirror.
- `B_T_2024_TAIRTE4_SUBSTITUTION`: SiO2/Si below the Au mirror may be omitted
  from the optical domain after a thickness/PML convergence test.
- `B_Z_2022_TAIRTE4_SUBSTITUTION`: the published 200-nm Au backplate likewise
  permits optical truncation below the metal.
- Thermal SiO2/Si is retained in the explicit reference. The reported reduced
  Robin values are screening candidates only and omit semi-infinite lateral
  spreading.

At 10 um, Ordal Au with k=69.2 has an intensity skin depth of
`11.499635 nm`. The 200-nm bulk
propagation factor is `2.797773e-08`.
This is not a replacement for the pending numerical backplane convergence.

## Important geometry corrections

- In the 2024 T architecture, the Ti/Au resonator touches the active 2-D layer;
  the Al2O3 cavity spacer is below that layer and above the Au mirror.
- In the 2022 Z architecture, the Au/Cr antenna chip is fabricated first and
  the active 2-D material is dry-transferred over it.
- T and Z are not interchangeable plan-view masks in a common stack.
- Published Z dimensions end at 8 um. The stored 10-um sweep is explicitly a
  numerical initialization, not a paper value.
- The 2022 paper's `10 W/m2` interface statement is a heat flux, not `G=10
  W/(m2 K)`; it is not promoted as an interface conductance.
