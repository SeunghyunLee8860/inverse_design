# Au sharp-interface width forward-FD checkpoint

Status: `FAILED_SHARP_INTERFACE_FORWARD_FD_PLATEAU_AT_100NM_EDGE_MESH`

This checkpoint is an isolated optical shape control. It is not a numerical
shape-adjoint certificate and contains no thermal, electrical, PTE, or
optimization result.

## Contract

- exact scalar Au at 10 um: `n + ik = 12.1 + 69.2i`
- binary sharp Au/air boundary; no gray material
- symmetric x-normal faces moved about an 8.0 um half-width baseline
- fixed 20 um y span and 50 nm Au thickness
- 100 nm lateral edge mesh and 5 nm Au z mesh
- GPU FDTD only; six PML; no Q clipping, smoothing, gain, or rescaling

All six cases used in the central differences passed the individual optical
closure and auto-shutoff gates. The central derivatives were:

- h=0.20 um: `-3.059351455931e-17 W/um`
- h=0.10 um: `-3.095343724579e-17 W/um`
- h=0.05 um: `-4.423843196808e-17 W/um`

The 0.20 -> 0.10 um change is `1.162787%`; the
0.10 -> 0.05 um change is `30.030438%`. The latter
fails the 1% plateau gate. A 50 nm boundary motion is below the present
100 nm lateral edge mesh, so this result is preserved as a mesh-resolution
failure rather than normalized or promoted.

The earlier 5.8 um strong-direction case also remains fail-closed because its
six-face closure was `1.054387%`.

## Decision

The density route remains rejected because uniform binary imported Au
diverged. The exact-binary sharp-interface route remains physically viable,
but its numerical AD-FD gate is not yet passed. The minimum next calculation
is an edge-local 50 nm lateral-mesh repeat of h=0.10 and 0.05 um, followed by
the numerical boundary-adjoint comparison only if the forward FD plateaus.
