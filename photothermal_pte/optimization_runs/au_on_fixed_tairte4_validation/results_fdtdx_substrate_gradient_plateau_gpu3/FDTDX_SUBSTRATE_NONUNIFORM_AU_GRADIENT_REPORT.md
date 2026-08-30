# FDTDX substrate-bearing nonuniform-Au gradient smoke

Status: **VALIDATED_FDTDX_DIAGNOSTIC_SUBSTRATE_NONUNIFORM_AU_GRADIENT_STABLE_STEP_PLATEAU**

The 20x20, 500-nm-pitch nonuniform Au density uses the passive causal Drude
strength law `s(rho)=rho^3` over a fixed 50-nm Au layer.  The objective is
total native-Yee material loss in Au, TaIrTe4, and SiO2 on the same 32-period
matched-interface optical contract as the validated binary endpoints.

| h | AD (W) | central FD (W) | strong relative error |
|---:|---:|---:|---:|
| 0.02 | 8.936886346e-16 | 8.945664000e-16 | 0.098122% |
| 0.01 | 8.936886346e-16 | 8.929280000e-16 | 0.085184% |
| 0.005 | 8.936886346e-16 | 9.027584000e-16 | 1.004673% |


The `h=0.02` and `h=0.01` forward derivatives differ by
`0.183150%` and both agree with AD below 1%.  The `h=0.005`
result is retained as a fail-closed small-step diagnostic: its objective
difference is too small for stable subtraction in the float32 time-domain
contract and its AD-FD error rises to
`1.004673%`. No value is removed,
fitted, normalized, or rescaled.

Baseline total Q is `2.477973832e-13 W`; matched-volume closure is
`0.137738%`; late-window change is
`0.014438%`.

The strict AD used 16 checkpoints and required
`5773.955 s` with about 36.2 GB observed GPU
memory.  It is an accuracy reference, not an approved per-iteration
production runtime.  A faster optical-period contract must be compared
against this gradient before combined PTE AD-FD or optimization.

This validates one strong optical-total-Q direction only.  It does not yet
validate the spatially weighted Maxwell source required by PTE, combined
thermal/electrical chain rule, Au thermopower, or an Au inverse design.
