# Floating-Au weighting/electrical AD–FD control

Status: **VALIDATED_FLOATING_AU_WEIGHTING_ELECTRICAL_CONTROL**

## Physical question

The Au pattern is a floating optical nanostructure, not a measurement
electrode.  Nevertheless, direct Au/TaIrTe4 electrical contact creates a
parallel conducting sheet.  It changes current crowding and the TaIrTe4
weighting solution even when the Au Seebeck coefficient is set to zero.

The high/low terminals are applied only to fixed TaIrTe4 (`y=a` max/min).  The
Au potential is solved as a floating unknown.  The objective uses a fixed
asymmetric TaIrTe4 temperature field and the paper TaIrTe4 conductivity and
Seebeck tensors.  Thus this checkpoint isolates the electrical contribution
to `dI/drho`; it is not a coupled Maxwell/thermal PTE prediction.

## Material and gray laws

- `sigma_Ta(x=b,y=a) = (1.10e5, 4.91e5) S/m`
- `S_Ta(x=b,y=a) = (27, -6) uV/K`
- `sigma_Au = 41152263.4 S/m`
  (bulk reference, not certified 50-nm film transport)
- `S_Au=0` in this isolation control
- Au sheet: `sigma_floor + rho (sigma_Au-sigma_floor)`
- vertical contact: `A (G_floor + rho G_contact)`

The fixed-shape numerical floors are reported in JSON and are not interpreted
as physical air conduction.

No device-specific Au/TaIrTe4 electrical contact resistivity was identified.
The four values below are numerical scenarios, not a confidence interval.

| contact resistivity | G contact (S/m2) | fixed-T current (nA) | ||dI/drho|| (nA) | worst h=0.0025 AD–FD | terminal imbalance |
|---|---:|---:|---:|---:|---:|
| $\rho_c=10^{-8}$ | 1.000e+08 | 0.370961 | 0.149187 | 3.246e-05 | 2.200e-15 |
| $10^{-10}$ | 1.000e+10 | 22.2866 | 6.18355 | 2.337e-07 | 1.828e-12 |
| $10^{-12}$ | 1.000e+12 | 29.1845 | 2.63027 | 1.441e-06 | 1.889e-13 |
| $10^{-14}$ Ωm² | 1.000e+14 | 25.3139 | 1.16776 | 1.337e-06 | 1.790e-12 |

The large change in both current and gradient proves that a directly touching
Au nanoantenna cannot automatically be treated as optically active but
electrically invisible.  Conversely, the numerical sweep does not identify
which contact value belongs to the fabricated device.

## Numerical gates

- worst fine-step AD–FD error: `3.245839e-05` (< 1%)
- worst linear residual: `9.513460e-12` (< 1e-8)
- worst terminal-current imbalance: `2.333601e-12` (< 1e-8)
- CPU linear-solve fallback: `False`

## Next coupled gate

The optical substrate must first be added to the FDTDX Au/TaIrTe4 model.
After that, `Q_Au + Q_Ta` is conservatively mapped into the validated thermal
operator, and the resulting temperature is passed to this electrical
operator.  The combined gradient is then the sum of Maxwell-Q, thermal
material/contact, and electrical weighting/contact terms.  Nonzero Au
thermopower remains a separate sensitivity case.

Raw NPZ is outside Git; path, size and SHA-256 are in the manifest.
