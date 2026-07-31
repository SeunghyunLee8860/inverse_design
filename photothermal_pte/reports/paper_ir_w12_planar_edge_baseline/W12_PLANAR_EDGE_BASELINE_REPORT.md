# Paper-like w0=12 µm scalar-Gaussian planar/edge baseline

Status: `BASELINE_PAPER_LIKE_W12_SCALAR_GAUSSIAN_OPTICAL_GATES_PASSED_REFINEMENT_PENDING`

This is a **paper-like scalar-Gaussian scenario with an explicitly assumed
waist**.  The 12 µm waist is not published by the paper.  This is not an
experimentally reproduced beam, paper-certified beam, paper reproduction, or
promoted production heat source.

No thermal, PTE, adjoint, gradient, or optimization calculation was run.

## Fixed optical contract

- wavelength: 11 µm
- scalar Gaussian, assumed waist radius 12 µm
- source span: 50×50 µm²
- FDTD span: 60×60 µm², six PML, no periodic boundaries
- TaIrTe₄: 130 nm; local baseline mesh 100 nm in x/y and 5 nm in z
- 285 nm SiO₂ on Si
- lab x=b, lab y=a, epsilon_z=epsilon_c=epsilon_b closure
- GPU FDTD only; no CPU fallback
- no Q clipping, smoothing, gain, rescaling, tiling, or deletion

## Baseline gates

- closure <0.5% for all four cases: **True**
- auto-shutoff ≤1e-5 for all four: **True**
- Q reintegration error <0.5%: **True**
- no negative-Q voxels: **True**
- no Q/source modification: **True**

| case | P_Q (W) | P_six (W) | closure | auto-shutoff | Qx/Qy/Qz (W) |
|---|---:|---:|---:|---:|---|
| planar_a | 3.995457041e-11 | 3.993046630e-11 | 0.060365% | 7.584460e-06 | 5.770e-15 / 3.993e-11 / 1.512e-14 |
| planar_b | 5.934078089e-11 | 5.931736740e-11 | 0.039472% | 8.487330e-06 | 5.932e-11 / 1.308e-15 / 2.025e-14 |
| edge_a | 2.256660508e-11 | 2.253087111e-11 | 0.158600% | 9.971810e-06 | 1.993e-12 / 2.051e-11 / 6.486e-14 |
| edge_b | 2.971653440e-11 | 2.970831307e-11 | 0.027674% | 9.594360e-06 | 2.943e-11 / 2.690e-13 / 2.052e-14 |


Raw edge/planar absorbed-power ratios are
`0.564807` for a-polarization
and `0.500778` for
b-polarization.  These raw powers were not equalized.  Equal-power
normalization appears only in the spatial-shape metrics and plots.

## Saved-field readback

The requested 0.6 µm monitor is realized on the Yee plane at approximately
0.5136 µm.  Its downward E/H decomposition remains a total-field diagnostic
that can contain reflection, scattering, and evanescent fields; it is not
called a pure incident waist.  Component-specific E fields inside TaIrTe₄ at
z=-65 nm were independently read on their staggered coordinates and
interpolated only to their exact common support.  Same-index component pairing
and extrapolation were not used.

The planar flake-midplane total-E² fit widths remain near 12 µm.  The
finite-edge fits have larger residuals and shifted centers, which is reported
as edge-induced non-Gaussian field redistribution rather than a shifted
incident beam.

## Remaining gate

The current 100 nm x/y mesh is the four-case baseline.  A 50 nm local-x/y
comparison remains required before promoting a material-Q artifact.  No
refinement run was started in this checkpoint.
