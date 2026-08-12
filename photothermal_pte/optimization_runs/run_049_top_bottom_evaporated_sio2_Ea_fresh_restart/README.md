# Run049: top/bottom electrodes, evaporated-SiO2 interface, E||a

This is a fresh optimization from exact uniform physical density `rho=0.5`.

- Lumerical axes: `x=b`, `y=a`, `z=c`
- source polarization: `E||a` (Lumerical `y`, 90 degrees)
- electrical terminals: bottom `psi=0`, top `psi=1`
- geometry mode: `contact_anchored`
- fixed TaIrTe4 contact strips: top and bottom
- TaIrTe4/SiO2 interface scenario: `evaporated`
- `G_TaIrTe4/SiO2 = 7.37e4 W/(m2 K)`
- `G_SiO2/Si = 1.1e9 W/(m2 K)`
- bulk SiO2 geometry, bulk thermal conductivity, optical material, source,
  FDTD domain, and mesh remain unchanged
- optimizer: native NLopt `LD_MMA`
- no warm start or reused MMA state
- no connectivity, symmetry, or volume constraint
- 500 nm solid/void minimum-feature contract

The evaporated value is a named paper-derived interface scenario, not a claim
that the bulk SiO2 material itself changes. A new combined physical-density
AD-FD precheck is required because the thermal operator and gradient change.
Raw FSP/NPZ artifacts remain outside Git.
