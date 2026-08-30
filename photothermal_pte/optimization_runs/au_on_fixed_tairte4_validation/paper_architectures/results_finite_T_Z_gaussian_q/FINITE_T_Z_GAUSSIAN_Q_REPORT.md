# Finite T/Z Gaussian Maxwell Q — Au-on/off and E∥a/E∥b

Status: **VALIDATED_FINITE_T_Z_AU_ON_OFF_EA_EB_VOLUMETRIC_Q**

All eight finite nonperiodic v261 GPU cases passed matched-volume six-face closure, auto-shutoff, native/pabs agreement, finite-array, and nonnegative-Q gates.
The scalar Gaussian source is the separately validated w0=4 µm source. Raw Q is never matched between polarizations or Au states.

## Signed top-Au effect on total absorption

| Architecture/polarization | Au-on P_Q (fW) | Au-off P_Q (fW) | (on-off)/off |
|---|---:|---:|---:|
| T_Ea | 5.985805 | 6.006510 | -0.345% |
| T_Eb | 5.935973 | 5.893651 | 0.718% |
| Z_Ea | 6.553511 | 7.849117 | -16.506% |
| Z_Eb | 20.166082 | 23.759495 | -15.124% |

This total-power comparison does not by itself identify whether Au changes top-Au, TaIrTe4, mirror, or substrate heating. That decomposition is the next material-overlap thermal gate.

Raw FSP/NPZ files remain outside Git; exact paths, sizes, and SHA-256 values are in the manifest.
