# Latest photothermal validation status

## Isolated 2 um TaIrTe4 steady-state HEAT

- Branch: `agent/validate-isolated-2um-heat-steady`
- Optical baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Optical code and validated production `Q_on`: unchanged
- Scope: steady state only; no transient, PTE current, adjoint, gradient, or
  optimization
- Current state: execution-path audit complete; API/analytic controls in
  progress

### Fail-closed findings

1. Existing v261 evidence does not round-trip diagonal TaIrTe4 conductivity:
   requested `[14.4, 3.8, 1.0]`, returned `[0.0]`.
2. The immutable validated source belongs to a 6 um periodic TaIrTe4 volume.
   Only 32.5525744323% of its power lies inside the requested 2 um by 2 um
   footprint, so importing it into that finite solid would violate the 0.5%
   conservation criterion.
3. A live v261 re-probe is currently unavailable because DEVICE cannot start
   its Ansys license-sharing client on this host.

Full-device domain/depth and interface-G sweeps remain prohibited until all
mandatory controls pass.
