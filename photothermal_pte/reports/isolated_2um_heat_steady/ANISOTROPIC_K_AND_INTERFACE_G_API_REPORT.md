# Anisotropic kappa and interface-G API report

Optical baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`.

## TaIrTe4 conductivity tensor

- Requested diagonal: `[14.4, 3.8, 1.0]` W/(m K)
- v261 round-trip value: `[0.0]`
- Scalar control passed: `True`
- Diagonal round trip passed: `False`
- Gate: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- No isotropic average or isotropic override was used.
- `kz = 1.0 W/(m K)` is an estimated value.

The installed Solid material property exposes the constant thermal
conductivity field used by the prior probe. The official scripting
example documents a scalar `thermal conductivity.constant`:
https://optics.ansys.com/hc/en-us/articles/360034919233-Creating-and-modifying-thermal-materials-from-a-script

## Fresh v261 probe

- Attempted: `True`
- Status: `BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE`
- Exception summary: `DEVICE startup failed: Ansys license-sharing client did not publish its server port`

A fresh DEVICE session could not be established on the current host
because the Ansys license-sharing client did not publish its server
port. The prior v261 result remains direct solver-API evidence, but a
new round trip could not be recorded.

## Interface conductance

- Internal domain-to-domain G verified: `False`
- Gate: `BLOCKED_INTERFACE_G_UNVERIFIED`

The requested finite conductances cannot be represented by silently
assuming perfect contact. The official HEAT boundary documentation
defines thermal impedance as a boundary thermal insulance in m2 K/W:
https://optics.ansys.com/hc/en-us/articles/360034398314-Boundary-Conditions-in-HEAT-Simulation-Object

A live two-domain analytic solve must still demonstrate that the
chosen internal-interface API realizes `DeltaT = q''/G` and that the
configured value survives save/load. No interface-G full-device case
was started.
