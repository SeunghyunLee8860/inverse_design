# Selected production optical chain

Status: `VALIDATED_SELECTED_PRODUCTION_OPTICAL_CHAIN`

This checkpoint replaces the coarse 20×20 µm / 201×201 optical-layout
assumption for future production work. It uses the frozen centered
18.6×18.6 µm window with 373×373 nodes at 50 nm and the unchanged 10 µm,
w0=8.5 µm scalar-Gaussian, six-PML contract.

## Runsetup and forward

- Realized global mesh: `[629, 629, 104]`
  (41,146,664 grid points).
- Minimum dx/dy/dz: `5.000000e-08` /
  `5.000000e-08` /
  `1.000000e-08 m`.
- GPU solver wall time: `230.229 s`.
- P_Q: `7.219486641789e-14 W`; P_six: `7.219533008696e-14 W`.
- Six-face closure: `6.422425e-06` (<0.5%).
- Final auto-shutoff: `8.279570e-08` (<1e-5).
- No Q clipping, smoothing, gain, or rescaling.

## Component-specific material Jacobian

- Density shape: `[373, 373]`.
- Worst mapping-only FD error: `1.336698e-09` (<1e-7).
- Worst JVP/VJP dot error: `1.658572e-14` (<1e-12).
- Maximum forward/index coordinate mismatch:
  `6.776264e-21 m` (<2e-18 m).
- Active J rows outside the exact selected support: zero for x, y, and z.
- Maxwell solves used to build J: zero; per-pixel Maxwell solves: false.

The first geometry attempt remains a failed diagnostic: the final audit used
the old coarse object name and raised a KeyError. No field solve occurred in
that failed attempt. The corrected run used a new directory.

This checkpoint does not certify thermal gray laws, the Maxwell adjoint, full
latent AD-FD, exact-binary DRC, or optimization.
