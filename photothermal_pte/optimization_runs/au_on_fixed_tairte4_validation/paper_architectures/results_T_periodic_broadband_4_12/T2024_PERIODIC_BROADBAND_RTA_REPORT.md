# T2024 periodic broadband R/T/A screening

Status: `VALIDATED_T2024_PERIODIC_BROADBAND_RTA_SCREENING`

This is the infinite periodic, normal-incidence plane-wave resonance-screening problem. It is not the finite Gaussian-beam PTE device.

The plotted absorption is the flux quantity `A=1-R-T`. No broadband 3-D Q monitor was retained. A selected resonance must therefore be rerun at one wavelength with component-resolved volumetric Q and six/control-volume closure before thermal use.

| case | peak wavelength (µm) | peak total A | runtime (s) |
|---|---:|---:|---:|
| T_Ea | 4.000000 | 0.198598 | 37.79 |
| T_Eb | 4.000000 | 0.310514 | 32.68 |
| bare_Ea | 4.000000 | 0.208996 | 30.33 |
| bare_Eb | 4.000000 | 0.309239 | 27.10 |

## Interpretation

- Ea: maximum signed T-minus-bare absorption enhancement is `0.009361` at `12.000000 µm`.
- Eb: maximum signed T-minus-bare absorption enhancement is `0.056920` at `11.825000 µm`.

The next optical calculation is a single-frequency Q certificate at the physically selected resonance, followed by a finite multi-T array under a Gaussian beam.
