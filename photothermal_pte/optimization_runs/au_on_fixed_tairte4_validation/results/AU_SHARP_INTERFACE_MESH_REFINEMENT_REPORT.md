# Au sharp-interface edge-mesh refinement checkpoint

Status: `VALIDATED_AU_SHARP_INTERFACE_FD_STEP_PLATEAU_BLOCKED_EDGE_MESH_CONVERGENCE`

This is an isolated exact-binary Au optical control. It does not certify a
numerical shape adjoint or a coupled Au/TaIrTe4 device.

All twelve forward cases used here pass six-face closure `<0.5%` and
auto-shutoff `<1e-5`. No gray Au/air material, CPU FDTD fallback, Q clipping,
smoothing, gain, or rescaling was used.

## Central finite differences

| edge mesh | h=0.10 um | h=0.05 um | step change |
|---:|---:|---:|---:|
| 100 nm | -3.095343724579e-17 | -4.423843196808e-17 | 30.030438% |
| 50 nm | -2.973806976340e-17 | -3.004177660606e-17 | 1.010948% |
| 25 nm | -2.883317312476e-17 | -2.904123473676e-17 | 0.716435% |

The 25 nm mesh passes the 1% FD-step plateau gate. However, the h=0.10 um
derivative changes by `3.042890%` from edge-50 to edge-25 nm,
so the derivative is not yet mesh-independent. The result is not rescaled or
promoted to a production Au optimization gradient.

## Decision

The density/imported Au route remains rejected because the uniform binary
endpoint diverged. The sharp-interface route is retained and now has a stable
within-mesh central difference at 25 nm. A numerical boundary-adjoint can be
implemented as a diagnostic at this mesh, but final certification additionally
requires an edge-mesh convergence resolution and the explicit Au/TaIrTe4
thermal/electrical contact scenarios.
