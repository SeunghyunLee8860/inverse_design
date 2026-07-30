# Paper-like IR beam audit and proposed source contract

Status: `PROPOSED_PAPER_IR_SOURCE_CONTRACT_READY_FOR_GPU_PROBE`

No FDTD, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization solve was run by this audit.

## Paper-reported facts

- Main paper PDF p.9, Methods 2.3: 40x reflective objective with NA=0.4,
  Block LaserTune QCL covering 7–13 µm, and an approximately 9–16 µm
  diffraction-limited spot size.
- Main paper PDF p.6, Figure 3: the 11 µm maps use 285 µW time-averaged
  incident power; Device A is 130 nm TaIrTe4 on 285 nm SiO2/Si.
- Main paper PDF pp.5–6, Figure 3F: a Gaussian heat source is focused near
  a crystal edge 45 degrees to the a axis at 11 µm for E parallel a/b.
  Its exact beam-center coordinate and incident power are not published.
- Main paper PDF p.6, Figure 3H/I: the 11 µm, 285 µW E-parallel-a/b SPCM
  maps include the off-axis edge; Figure 3I is extracted along the plotted
  dashed lines.  Absolute raster coordinates, scan step, and line coordinates
  are not published.
- SI PDF p.5, Figure S5: 7.84, 9.17, 10, 11 and 12.5 µm were measured for
  E parallel a and b.
- SI PDF p.6, Equations S1–S2: the thermal model uses Gaussian intensity and
  calls `w0=2 sigma` the beam radius.

The paper does **not** state whether the 9–16 µm spot is a radius, diameter,
FWHM, or 1/e^2 width.  It does not publish the exact wavelength-specific
11 µm spot, waist-plane location, objective pupil fill, or alignment error.

## Selected first source-only candidate

The 0.4-NA Airy FWHM estimate at 11 µm is
`14.135000 µm`; its same-FWHM Gaussian radius is
`12.005164 µm`.  The first candidate therefore uses a rounded
`w0=12.0 µm`, explicitly as an assumption rather than a paper value.

- eta=lambda/(pi*w0): `0.291784`
- Rayleigh range: `41.126304 µm`
- source-to-focus distance: `5.065000 µm`
- expected source-plane radius: `12.090664 µm`
- source span: `50 µm`; lateral domain: `60 µm`
- fitted-Gaussian square capture: `99.99291408%`
- expected boundary maximum/peak:
  `1.93378964e-04`
- expected boundary mean/peak:
  `5.86048981e-05`

The backward source must use
`distance from waist = -5.065000 µm`.
The legacy positive sign is not reused.

## Scalar/vector status

Since `w0` is only about 1.09 wavelengths, the scalar source is a
source-only calibration candidate, not a production approval.  Before any
planar/edge material case, it must be matched against a fully vectorial
thin-lens source using NA=0.4 by realized flake-plane power, center, fitted
and second-moment widths, focus, and normalized spatial intensity.  Merely
toggling the scalar Boolean with identical numeric parameters is prohibited.
