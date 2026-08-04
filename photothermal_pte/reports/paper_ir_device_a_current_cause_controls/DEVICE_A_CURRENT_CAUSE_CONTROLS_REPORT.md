# Device-A current-cause controls

Status: `COMPLETED_DEVICE_A_CURRENT_CAUSE_CONTROLS`

This checkpoint separates the saved Maxwell/TaIrTe4 thermal fields from the
electrical weighting operator, and adds an explicitly named planar-SiO2
background thermal sensitivity.  No new FDTD, adjoint, AD-FD, or optimization
was run.

## Sampled-maximum current ratios

| control | sampled max `|Ib|/|Ia|` |
|---|---:|
| Ta-only actual weighting | 0.835358 |
| Ta-only equal-power efficiency | 0.772182 |
| Ta-only uniform 45deg weighting | 0.558686 |
| Ta+planar-SiO2 actual weighting | 0.842262 |
| Ta+planar-SiO2 uniform 45deg | 0.571060 |


The Figure-3I visual reference is about `143/122=1.172131`.  A ratio
below one retains the simulated `a>b` trend.

## SiO2 diagnostic contract

At 11 um the explicit optical constants are
`n_SiO2=2.019443683+0.162620219i` and
`n_Si=3.421289622+4.389880310e-05i`.  Normal-incidence planar TMM gives
SiO2 absorptance `2.294526%`.  The
ideal infinite Gaussian has `w0=8.75 um` and `Pinc=284.40 uW`; no source or
result was rescaled to match the TaIrTe4 calculation.

This oxide source is an **empty-stack planar-background sensitivity**, not
finite-device Maxwell SiO2 Q.  It neglects TaIrTe4/electrode modification of
the oxide field and polarization-dependent edge redistribution.  It therefore
cannot promote or replace the blocked full-SiO2 optical calculation.

## Interpretation

The controls give three direct conclusions.

1. **The digitized weighting field is not producing the reversal.** Replacing
   it by an ideal uniform 45-degree field moves `|Ib|/|Ia|` from
   `0.835358` down to
   `0.558686`.
   The actual electrode weighting therefore helps `b` relative to `a`; it
   does not explain why the simulated ratio remains below one.
2. **Absorbed-power magnitude is not sufficient.** Equal-power normalization
   moves the ratio to
   `0.772182`.
   Thus `b` already benefits from its larger absorbed power, while the spatial
   current-generation efficiency of the Maxwell/TaIrTe4 temperature field
   still favors `a`.
3. **The planar-background SiO2 control is much too small to reverse the
   trend.** Adding it moves the actual-weighting ratio only from
   `0.835358` to
   `0.842262`,
   whereas the Figure-3I visual reference is about `1.172131`.

The strongest identified remaining cause is therefore the polarization-
dependent **spatial Maxwell TaIrTe4 Q distribution and its downstream thermal
field**, not the current digitized weighting operator or this planar oxide
background. This is a causal diagnosis, not proof that the optical field is
wrong: a matched beam-radius sweep and a stable finite-device SiO2-Q solve are
still required to separate source size, edge scattering, and oxide absorption.

The exact contact CAD remains unresolved, and the 14.11-ohm calculated versus
213-ohm measured resistance mismatch continues to block absolute-current
certification. Chopping/frequency response was not evaluated here. Existing
Au/Ti optical-on/off and metal-thermal limiting controls changed current only
at about the percent level, so they are not presently large enough to explain
the sign of the polarization ratio by themselves.

All four oxide thermal solves pass TMM depth integration, linear residual, and
energy-balance gates: `True`.  External thermal
fields are recorded by path, size, and SHA-256 in the manifest.

The long-running thermal workers were audited at process level. Duplicate
launcher states were stopped before they produced artifacts; the published
raw results were generated sequentially with one worker at a time.
