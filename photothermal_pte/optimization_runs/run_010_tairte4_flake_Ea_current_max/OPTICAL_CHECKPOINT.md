# Run 010 optical checkpoint

Status: `VALIDATED_COMPACT_TAIRTE4_FLAKE_OPTICAL_CONTRACT`

This checkpoint validates the fast optical contract only. It does not claim
that the thermal, electrical, combined-adjoint, or optimization stages have
run.

## Frozen contract

- Lumerical coordinates: `x=b`, `y=a`, `z=c`
- wavelength: 10 um
- finite scalar Gaussian target waist: 8.5 um
- optical domain: 40 x 40 um with six PML boundaries (24 layers)
- source span: 34 x 34 um
- finite TaIrTe4 flake: 24 x 24 x 0.1 um
- central design: 16 x 16 um, TaIrTe4-to-void physical density
- production flake mesh: 100 nm laterally and 10 nm vertically
- no periodic/Bloch boundary and no CPU FDTD fallback
- no Q clipping, smoothing, gain, or rescaling

## Source-only gate

The 40 um domain / 34 um aperture source-only run passed. The realized
waists were 8.476997 um and 8.501440 um, the Gaussian-fit NRMSE was 0.2090%,
ellipticity was 0.2879%, boundary maximum/mean intensity fractions were
2.3516e-4 and 6.664e-5, and auto-shutoff reached 9.72536e-9. Runtime on GPU 5
was 3.67 s.

## Uniform-density forward controls

At uniform physical density rho=0.5:

| polarization | P_Q (W) | P_six (W) | six-face closure | GPU runtime (s) |
|---|---:|---:|---:|---:|
| E parallel a | 3.25424260645e-14 | 3.25422181535e-14 | 6.38896e-6 | 44.24 |
| E parallel b | 2.50907169061e-14 | 2.50887042589e-14 | 8.02213e-5 | 41.91 |

Both passed the 0.5% closure gate and the 1e-5 auto-shutoff gate.

## Mesh and domain convergence

- 100 nm versus 50 nm, E parallel a:
  - total-power change: 0.001943%
  - conservative common-grid lateral-Q NRMSE: 0.350608%
  - equal-power shape NRMSE (diagnostic only): 0.350603%
- 40 um versus 48 um, E parallel a:
  - total-power change: 8.93989e-8
  - conservative common-grid lateral-Q NRMSE: 3.47317e-6
  - equal-power shape NRMSE (diagnostic only): 3.47356e-6

The 40 um domain and 100 nm lateral flake mesh are therefore promoted for
the next validation stage. Equal-power normalization was used only to
compare shape; no raw source or Q artifact was modified.

## Next gate

The next stage is the explicit TaIrTe4-to-void thermal/interface model and a
fixed-Q thermal AD-FD test. Run 009's upper-SiO2 interpolation
`G_air + rho*(G_SiO2-G_air)` is not applicable to this in-plane TaIrTe4
topology and must not be copied.
