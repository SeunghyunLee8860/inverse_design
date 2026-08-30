# Finite inverse-T array Gaussian size audit

This is an offline geometry/cost audit. It does **not** contain an FDTD, thermal,
PTE, adjoint, or optimization result. The periodic resonance search spans
`4-12 um`; that wavelength interval is independent of the beam-waist choices.

| Assumed waist | Array | T count | Maxwell x/y span | Fine Yee-cell proxy |
|---:|---:|---:|---:|---:|
| 4.0 um | 16.5 x 17.0 um | 187 | 28.5 x 29.0 um | 13,464,000 |
| 8.5 um | 34.5 x 35.0 um | 805 | 46.5 x 47.0 um | 57,960,000 |

The finite-device Maxwell model uses six PML boundaries. Thermal and electrical
models use physical boundaries and do not inherit optical PML. The array reaches
`r=2*w0`, where an ideal Gaussian intensity is `exp(-8)=3.35e-4` of its peak.

Decision: run the `w0=4 um` finite-array smoke first. The `w0=8.5 um` scenario
is promoted only after a v261 runsetup audit records the realized mesh, GPU
memory, and source-boundary field level. The fine-cell proxy is not a memory
claim.
