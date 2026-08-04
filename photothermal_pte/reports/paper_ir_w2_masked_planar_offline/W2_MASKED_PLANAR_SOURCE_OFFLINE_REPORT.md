# Offline masked-planar and source-contract audit

Status: `VALIDATED_OFFLINE_MASKED_PLANAR_AND_SOURCE_AUDIT`

No new FDTD, thermal, PTE, weighting-potential, adjoint, gradient, or
optimization solve was executed.

## Answers first

1. **Support removal.**  The exact analytic dual-cell half-plane removes
   `49.950862%` of planar-b power, accounting
   for `98.760752%` of the observed total drop.
2. **Signed EM residual.**  After that support removal,
   `D_EM/P_planar = 0.626783%` and
   `D_EM = 1.225518932e-18 W`, or
   `1.239248%` of the total drop.
3. **Spatial shape.**  Equal-power NRMSE is
   `99.664762%`
   for full→masked,
   `12.183080%`
   for masked→edge, and
   `100.185592%`
   for full→edge.  These are not treated as additive quantities.
4. **Why nominal w0=2 µm is not realized.**
   `lambda/(pi*w0)=1.750704` is greater
   than one.  The scalar/paraxial contract is outside its validity range
   and the 6 µm aperture severely truncates its requested source-plane
   profile.
5. **Paper-like GPU value.**  A new calculation is worth considering only
   after selecting a physically realizable beam definition and certifying
   a source-only/background reference.  This report proposes cases but does
   not execute them.

## Signed power decomposition

- P_planar = `1.955253431e-16 W`
- P_masked = `9.785874965e-17 W`
- P_edge = `9.663323072e-17 W`
- D_total = `9.889211237e-17 W`
- D_support = `9.766659344e-17 W`
- signed D_EM = `1.225518932e-18 W`
- D_support/P_planar = `4.995086156e-01`
- signed D_EM/P_planar = `6.267826525e-03`
- D_total − (D_support + D_EM) =
  `0.000e+00 W`

The primary mask is the exact overlap of every bounded dual cell with
`y<=x`, not a center Boolean mask.  Component-specific native Yee cut-cell
fractions are also evaluated.

## Loss-participation proxy

The component quantity
`Im(epsilon_edge,c)/Im(epsilon_planar,c)` is retained only as a diagnostic
loss-participation proxy, never as geometric occupancy.  No clipping is
applied.  Cells below the recorded denominator floor, proxy ranges, and
counts below zero or above one are listed component-by-component in the
summary JSON.  Native field/index coordinate mismatch is at most
`8.470e-22 m`;
planar/edge component coordinates are identical.  Component-to-common
mapping uses exact bounded dual-cell overlaps and reports its power error.

| component | denominator floor | near-floor cells | active-ratio cells | f<0 | f>1 | f min | f max | excluded-cell direct signed power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x | 2.618187e-11 | 720,920 | 1,009,288 | 0 | 268 | 0 | 1.99951 | 0.000e+00 |
| y | 2.045339e-10 | 720,920 | 1,009,288 | 0 | 268 | 0 | 1.99951 | 0.000e+00 |
| z | 2.618187e-11 | 723,610 | 940,693 | 0 | 0 | 0 | 1 | 1.079e-29 |

The `x` and `y` ratios reach about 2 in 268 cells each.  This is retained
without clipping and is direct evidence that `f_c` is not occupancy.  The
maximum component-to-common conservative-remap power error is
`1.613e-16`.

The analytic-cut residual on the material side is:

| edge-normal band n (µm) | absolute-residual integral / masked reference | absolute residual power (W) |
|---|---:|---:|
| -0.25 to 0 | 16.801959% | 1.328937e-18 |
| -0.5 to -0.25 | 13.198243% | 1.025537e-18 |
| -1 to -0.5 | 14.937066% | 2.334023e-18 |
| -2 to -1 | 8.599014% | 2.318827e-18 |
| -4 to -2 | 8.191344% | 2.552109e-18 |

It is smaller in the 2–4 µm band than in the 0–0.25 µm band, but the trend
is not monotonic and remains nonzero far from the edge.  It is therefore
reported as a diagnostic profile, not fitted to an edge-decay law.

## Spatial comparisons

| comparison | equal-power NRMSE | Pearson correlation | cosine similarity |
|---|---:|---:|---:|
| full planar ↔ analytic masked planar | 99.664762% | 0.575858342 | 0.708116218 |
| analytic masked planar ↔ finite edge | 12.183080% | 0.990032375 | 0.992568344 |
| full planar ↔ finite edge | 100.185592% | 0.567369420 | 0.703337159 |
| loss-participation masked ↔ finite edge | 12.190833% | 0.990019605 | 0.992559080 |

NRMSE values are independent pairwise comparisons and are not decomposed
or added.

## Source-object audit

- requested waist: `2 µm`
- source-to-focus distance: `5.065000 µm`
- paraxial-formula source-plane radius:
  `9.090067 µm`
- native absolute `sourcepower` readback:
  `2.339588230e-15 W`
- saved square-aperture E-only transverse integral:
  `3.915297315e-14`
- saved square-boundary max intensity/peak:
  `0.735030`
- saved square-boundary mean intensity/peak:
  `0.662321`
- fitted infinite-Gaussian waist:
  `7.603443 µm`
- retained-square second-moment waist:
  `3.338921 µm`
- fitted infinite-Gaussian square captured fraction:
  `32.484972%`
- fitted infinite-Gaussian inscribed-circle fraction:
  `26.754415%`
- paraxial diagnostic square/circle fractions:
  `24.087030%` /
  `19.574616%`

The saved source-object `source_profile_E` is the primary evidence.  The
paraxial formula is used only as an aperture-truncation failure diagnostic.
The source-object E array and `sourcepower` use different spectral-amplitude
normalizations: the E-only plane-wave integral is retained as a shape proxy,
not called launched watts; `sourcepower` is the primary absolute power.
The z=50 nm total-field downward decomposition may contain reflection,
edge-scattered, and evanescent fields and is not called a pure incident-beam
waist.  The Gaussian fit extrapolates an infinite profile, whereas the
second moment uses only retained square-aperture power; truncation makes the
two widths differ.

## Proposed next GPU contract — not executed

First verify the paper/SI definitions of wavelength, spot radius versus
diameter versus FWHM, power, location, and polarization.  Unpublished
choices remain named scenarios.  The minimum optical set is:

1. source-only/background reference for the selected beam and domain;
2. planar-a and planar-b;
3. finite-edge-a and finite-edge-b.

The aperture/domain must contain at least 99.9% fitted incident power, have
small aperture-edge intensity and sufficient PML margin, and certify the
flake-plane incident profile in the background reference.  Production
material remains `epsilon_c=epsilon_b`, with `x=b, y=a, z=c=b`.

The current 12 µm diagnostic grid required
`12.57`–
`14.48` minutes of logged solver
wall time per 4 ps case.  Five cases on that same grid therefore give only a
lower bound of about
`68.41`
minutes.  A physically adequate paper-like aperture/domain can be much
slower, so a contract-only grid/memory probe is required before approval;
no reliable paper-like runtime is claimed from the truncated w0=2 µm runs.
