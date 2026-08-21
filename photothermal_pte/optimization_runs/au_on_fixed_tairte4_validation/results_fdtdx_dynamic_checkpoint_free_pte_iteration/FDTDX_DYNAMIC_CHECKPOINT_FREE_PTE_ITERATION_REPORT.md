# Dynamic checkpoint-free production PTE iteration

Status: `VALIDATED_FDTDX_PRODUCTION_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION`

Unlike the earlier fixed-weight equivalence, this run recomputed all
iteration-dependent quantities from the current density: native-Yee Maxwell
`Q`, explicit 3-D temperature, electrical weighting potential, thermal and
electrical adjoints, and the native-Yee `dI/dp` source weights. It then ran one
reciprocal Maxwell adjoint. No FDTD checkpoint or field time history was kept.

| metric | result |
|---|---:|
| dynamic combined vector error vs frozen AD | 0.153988% |
| dynamic combined norm error | 0.107068% |
| dynamic combined angle | 0.063378 deg |
| native vs explicit source-adjoint contraction | 2.869e-16 |
| PTE objective vs weighted-Q contraction | 4.437e-10 |
| forward + Maxwell adjoint execution | 358.501 s |
| first compile + forward + Maxwell adjoint | 408.499 s |
| full measured pipeline after runsetup audit | 426.619 s |
| speedup of the two Maxwell solves vs frozen checkpoint AD | 8.145x |

The full measured pipeline includes the current thermal/electrical solves,
their adjoints, remap pullback, plotting, and JSON/NPZ output. A persistent
optimizer process can reuse JIT compilation; this run does not yet measure
multi-iteration steady-state timing.

The inherited substrate status remains
`BLOCKED_LUMERICAL_10UM_SI_PALIK_READBACK`. This is a numerical
contract equivalence, not a paper-certified substrate material claim.
