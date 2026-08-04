# Device-A scalar-Gaussian waist-sensitivity contract

Status: `FROZEN_DEVICE_A_EXPLICIT_WAIST_SENSITIVITY_CONTRACT`

The paper reports an approximately 9--16 um diffraction-limited QCL spot but
does not state whether that number is a radius, a diameter, or an FWHM.  Its
Supporting Information defines

\[
I(x,y)=\frac{2P}{\pi w_0^2}\exp[-2(x^2+y^2)/w_0^2]
\]

and calls `w0` the beam radius.  The new `4.5`, `6.5`, and `8.0 um` values are
therefore explicit diameter-interpretation scenarios.  They are not a
confidence interval and are not paper-certified beam measurements.

The existing `w0=12 um` result is preserved without modification and is
relabeled as a large-beam scenario.  A new `waist-sensitivity` execution
contract retains the complete 60-um/six-PML/50-um-aperture numerical model
while changing only the assumed waist.  It cannot be promoted as the fixed
12-um production contract.

Each waist must first pass an independent GPU-only homogeneous-air source
gate.  Only then may the two s0 Device-A material cases (`E||a`, `E||b`) run.
All terminal currents use the same integrated incident power of 285 uW and
the unchanged explicit-3D thermal/weighting operator.  Raw optical Q is never
matched between polarizations or waists.

The Figure-3I beam-position profile cannot be inferred from one s0 result.
After the six s0 material cases are reported, at most one selected waist may
receive the existing three-position `s0-1/s0/s0+1 um` follow-up.  No dense
scan is authorized automatically.
