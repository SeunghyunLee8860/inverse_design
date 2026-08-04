# Device A geometry digitization audit

Status: `DEVICE_A_FIGURE_DIGITIZATION_AUDITED`

This is a `FIGURE_DIGITIZED_APPROXIMATION`, not unpublished Device-A CAD.
The overlay was calibrated from the 10-µm scale bar in Figure 2D using a
600-dpi render of the local paper PDF.  Code axes are fixed as
`x=b`, `y=a`.

## Frozen geometry

- Scale: 18.1 rendered pixels per µm.
- Visible flake bounds: approximately `x=[-15.03,15.03] µm`,
  `y=[-11.77,11.77] µm`.
- The top and bottom metal footprints and their flake-contact boundary
  segments are stored separately.
- The mean off-axis side is defined by digitized vertices 4 and 7.  Its
  code-coordinate unit tangent is `(0.592817,0.805337)` and its inward unit
  normal is `(0.805337,-0.592817)`.
- The first optical smoke position is frozen before simulated current is
  inspected: `(-9.683437,0.597238) µm`, three micrometres inward from the
  mean off-axis side.

The 3-µm offset registers the approximate Figure-3I edge-current minimum.
It is a figure-derived scan-coordinate assumption, not stage metrology and
not a parameter that may be tuned after seeing the simulation.

## Uncertainty contract

| item | estimated uncertainty |
|---|---:|
| visible flake edge | 0.4 µm |
| flake beneath metal | 0.8 µm |
| contact endpoint | 0.6 µm |
| beam-position registration | 1.0 µm |
| scale bar | 0.15 µm |

The comparison target is the existing Figure-3J digitization
`|Ia|/|Ib| = 0.836590 ± 0.008526`.  A single-position result remains a
smoke diagnostic; promotion to a Figure-3J comparison requires the approved
minimal edge-position check.

## Files

- `DEVICE_A_FIG2D_DIGITIZATION_OVERLAY.png`
- `DEVICE_A_GEOMETRY_CONTRACT.png`
- `device_a_geometry_digitization.json`
- `device_a_geometry_vertices.csv`

No FDTD, thermal, PTE, adjoint, AD-FD, or optimization solve was run for
this geometry checkpoint.
