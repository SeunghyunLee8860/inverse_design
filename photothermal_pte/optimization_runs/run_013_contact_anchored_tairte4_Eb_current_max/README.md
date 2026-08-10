# Run 013 — contact-anchored TaIrTe4, E parallel b

Status: `AUTO_CHAIN_ARMED_AFTER_RUN012`

This is the independent `E || b` counterpart of Run 012.  It uses the same
geometry, uniform-0.5 initial density, optical/thermal/electrical operators,
feature constraints, and terminal-conductance safeguard.  It does not warm
start from the `E || a` result.

The fail-closed supervisor starts this run only after Run012 emits
`continuous_continuation_complete`.  It reuses the validated geometry/material
layout and component-wise density-to-Yee Jacobian, while the forward source is
explicitly changed and read back at 0 degrees (`E || b`).  Run013 starts from a
new uniform physical density of 0.5; no Run012 optimized density is reused.

Each completed Run013 evaluation is rendered and pushed independently to this
directory.  The figure reports both the raw 1 V/m-source current and the
linearly equivalent current at the explicit 285 uW reference power.
