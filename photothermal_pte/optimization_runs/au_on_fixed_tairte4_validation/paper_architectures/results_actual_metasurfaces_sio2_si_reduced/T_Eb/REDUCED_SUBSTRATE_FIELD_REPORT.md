# Reduced optical SiO2/Si closure and near fields — T_Eb

The 1.5-um physical thermal oxide is **not** retained in this optical solve.
The optical closure uses 200-nm Au / 285-nm SiO2 / Si because the opaque Au
mirror reduces the measured bottom transmission to `3.180e-10`.
The already preserved 1.5-um control established the same optical observables.

- P_Q: `4.236429882e-16 W/cell`
- matched-volume closure: `0.145507%`
- auto-shutoff: `5.669e-07`
- runtime: `32.87 s`
- mesh: `[151, 101, 132]`; structure dx=dy=10 nm, dz=5 nm

The plots show the actual collocated Lumerical electric fields at the TaIrTe4
midplane and through xz/yz cross sections, plus component-resolved absorption.
No clipping, smoothing, gain, or source rescaling was used.
