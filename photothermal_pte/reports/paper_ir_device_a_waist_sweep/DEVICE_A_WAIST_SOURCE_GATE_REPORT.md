# Device-A explicit-waist source-only gates

Status: `VALIDATED_AT_LEAST_ONE_DEVICE_A_WAIST_SOURCE_GATE`

The paper SI defines `w0` as the 1/e^2 intensity radius, but the main-text
9--16 um diffraction-limited spot does not identify its radius/diameter
convention.  These are explicit sensitivity scenarios, not paper-certified
beam measurements.

| target w0 (um) | source-object w0 (um) | fitted wx (um) | fitted wy (um) | fit NRMSE | ellipticity | Au/Ti incident fraction | gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4.5 | 4.468824 | 5.7050 | 5.7494 | 1.8182% | 0.7757% | 0.2565% | False |
| 6.5 | 6.454968 | 6.8652 | 6.9200 | 1.4536% | 0.7952% | 0.0302% | False |
| 8 | 7.813360 | 7.9948 | 8.0364 | 0.5773% | 0.5184% | 0.3196% | False |
| 8.75 | 8.610603 | 8.7391 | 8.7710 | 0.2570% | 0.3646% | 0.7238% | True |

Power fractions are exact polygon--bounded-dual-cell overlaps of the stored
target-plane downward Poynting field.  They are not single-point intensity
estimates.  Failed source fields are retained as diagnostics and are not
used for material Q, thermal, or terminal-current calculations.

No fit threshold was relaxed and no field, power, Q, or current was rescaled.
The pre-existing 12-um large-beam scenario remains unchanged.

The preserved 12-um large-beam source passes its historical gate, has fitted waist `12.0011 um`, and sends `4.2042%` of the stored target-plane downward power through the digitized Au/Ti polygons.
