# FDTDX 4 um source-only incident-power calibration

Status: **VALIDATED_FDTDX_4UM_SOURCE_POWER_CALIBRATION**

This is an all-air run on the identical nonuniform grid and source aperture.
The full-device plane detector is not used as a pure-incident calibration because it contains reflection.

| polarization | incident power (W) | runtime (s) |
|---|---:|---:|
| Ea | 1.856360118e-12 | 44.98 |
| Eb | 1.856359901e-12 | 62.32 |

Ea/Eb mismatch: `0.000012%`.
All later 285 µW values use the same source-only scale factor; the two polarizations are never matched to one another.
