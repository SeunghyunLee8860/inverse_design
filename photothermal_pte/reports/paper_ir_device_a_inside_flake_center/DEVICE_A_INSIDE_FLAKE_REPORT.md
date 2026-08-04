# Device-A inside-flake beam-centre diagnostic

Status: `COMPLETED_DEVICE_A_INSIDE_FLAKE_BEAM_CENTER_DIAGNOSTIC`

The Gaussian centre was deliberately moved from the earlier outside-flake
registration to digitized `(x=b, y=a)=(0, 3) um`, which becomes `(0, 0) um`
in the realized simulation frame. This is a named inside-flake diagnostic,
not a claim about the unpublished experimental stage coordinate.

## Fixed contract

- 11-um scalar Gaussian, `w0=8.75 um`, 50-um square source aperture.
- 64-um lateral FDTD domain, six PML boundaries, GPU-only solve.
- Device-A digitized flake and Au/Ti electrode polygons.
- TaIrTe4-only volumetric Maxwell Q enters thermal through literal
  optical-cell/thermal-material intersection density.
- SiO2 and metal optical loss are not thermal sources in this diagnostic.
- No clipping, smoothing, gain, global rescaling, tiling, or nearest-cell
  relocation was used.

## Result

| metric | E parallel a | E parallel b | b/a |
|---|---:|---:|---:|
| TaIrTe4 optical power at unit central intensity (W) | 1.986512425e-11 | 2.769596918e-11 | 1.394201 |
| mapped thermal source at 285 uW (W) | 4.726756142e-05 | 6.561667364e-05 | 1.388197 |
| Tmax rise (K) | 0.169580018 | 0.232144085 | 1.368935 |
| TaIrTe4 average rise (K) | 0.0327018775 | 0.0451833195 | 1.381674 |
| integrated PTE current (pA) | 340.440601 | -3472.285920 | abs=10.199388 |

Both optical closure errors are below 0.5%, both auto-shutoff values are below
`1e-5`, mapping power errors are zero to recorded precision, and both thermal
energy errors are far below 1%.

The current changes sign and `|I_b|>|I_a|` at this inside-flake position.
However, the digitized geometry predicts about 14.1 ohm whereas the measured
Device-A resistance is 213 ohm. Therefore the absolute pA/nA magnitudes are
not certified experimental predictions.

## Corrected failure found during this task

The first thermal attempt silently rebuilt the old outside-flake coordinate
translation from the geometry JSON. It displaced the thermal flake by 10.67
um relative to the new optical Q. That raw result is preserved as a failed
diagnostic. Production thermal now loads the realized translated polygon from
the actual optical `case_result.json` and fail-closes on a geometry-path or
coordinate mismatch.
