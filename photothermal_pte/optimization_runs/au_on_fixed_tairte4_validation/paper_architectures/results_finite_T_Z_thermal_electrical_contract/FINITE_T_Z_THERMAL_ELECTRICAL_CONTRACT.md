# Finite T/Z thermal-electrical contract

Status: `FROZEN_FINITE_T_Z_THERMAL_ELECTRICAL_CONTRACT`

The preceding periodic calculations certify optical per-cell Q only. They are
not tiled, cropped, or interpreted as periodic temperature/PTE. Each T/Z
antenna is now placed once at the center of a finite 20 x 20 um TaIrTe4 flake
and is illuminated by a finite scalar Gaussian.

The Maxwell domain uses six PML boundaries. The thermal domain is a separate
explicit 3-D 32 x 32 um domain with 20-um Si depth, fixed far x/y and bottom
bath boundaries, and top convection. Optical PML is never used as a thermal or
electrical boundary.

Both top-bottom and left-right TaIrTe4 terminal pairs are solved. The top Au T/Z
structure is electrically floating and may alter the weighting field through a
finite vertical contact, but it is not a readout electrode.

Au/TaIrTe4 thermal and electrical contacts and TaIrTe4/Al2O3 thermal contact are
explicitly named numerical scenarios because device-specific measured values
are unavailable. No single scenario is promoted as experimental truth.
