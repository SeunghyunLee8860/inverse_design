# Device-A material-overlap Q remap validation

Status: `VALIDATED_DEVICE_A_MATERIAL_OVERLAP_REMAP`

The nearest-material-cell projection has been removed from the Device-A
thermal source path.  For every optical cell, its already computed absorbed
power is distributed only through exact overlap with the discrete TaIrTe4
material volume used by the thermal FVM.  A zero-overlap cell is reported as
non-TaIrTe4/unattributed and is not moved to a nearby flake cell.

The current FVM assigns one material per Cartesian cell, so
`Omega_TaIrTe4,h` is the union of cells that the same thermal operator solves
with TaIrTe4 conductivity.  Using an analytic sub-cell polygon only for Q
while solving that partial cell as air would be inconsistent.  A truly
analytic polygon cut-cell overlap would therefore require a matching cut-cell
conductivity/interface operator and is not claimed here.

## 100 nm thermal-grid result

- full common-grid optical power: `4.218824235332772e-05 W`
- material-overlap-attributed TaIrTe4 power: `4.199561714195624e-05 W`
  (`99.543415%` of full)
- zero-overlap, non-TaIrTe4/unattributed power: `1.473989031424926e-07 W`
  (`0.349384%`)
- explicitly excluded metal power: `4.522630822899162e-08 W`
  (`0.107201%`)
- signed partition residual: `0.000000000000000e+00 W`
- source-to-target mapping error: `0.000e+00`
- power outside TaIrTe4 after mapping: `0 W`
- change from the old cell-centre mask diagnostic: `0.218052%`

The 500 nm case is only a coarse operator control.  Its larger zero-overlap
term demonstrates why coarse thermal geometry must not be used to promote a
physical source partition.

The immutable input artifact predates the new 11-um Palik substrate contract.
Therefore this checkpoint validates the mapping mathematics and real Device-A
grid execution, but it does not promote the quoted absorbed power as the final
Palik optical/thermal prediction.  No new FDTD, thermal solve, weighting
potential, PTE current, adjoint, or optimization was run.
