# Au-on-fixed-TaIrTe4 thermal material/interface AD–FD control

Status: **VALIDATED_AU_THERMAL_MATERIAL_INTERFACE_CONTROL**

## What this checkpoint proves

This is a fixed-heat-source, GPU sparse-FVM control.  It validates the exact
discrete derivative of the Au/air thermal layer and the TaIrTe4-to-Au/air
contact-area relaxation.  It does **not** contain a Maxwell solve, PTE current,
electrical shunting, Au thermopower, or topology optimization.

The design is 20 x 20 physical 500-nm pixels over 10 x 10 um.  A fixed 100-nm
TaIrTe4 sheet (`x=b`, `y=a`, `z=c`) is covered by a 50-nm Au/air design layer.
The bottom is the paper-reduced thermally-grown-SiO2 Robin boundary and the
top is an ambient Robin boundary.  Lateral faces are adiabatic **for this
operator control only**, not as a promoted production boundary.

## Gray material and contact law

The physical density is interpreted as parallel contact-area fraction:

```text
g_face(rho) = A [(1-rho)/R_Ta-air + rho/R_Ta-Au]
```

The Au/air-layer conductivity is `k_air + rho (k_Au-k_air)`, and all harmonic
half-cell face resistances are differentiated.  No clipping, smoothing, gain,
or gradient rescaling was used.

`k_Au=317 W/(m K)` is a bulk reference scenario, not a certified 50-nm-film
value.  No direct Au/TaIrTe4 thermal-boundary-conductance measurement was
identified.  The 17.24 MW/(m2 K) case is derived from the *calculated*
Au/MoS2 resistance `5.8e-8 m2 K/W` in [Mao et al.](https://arxiv.org/abs/1407.2335),
and is explicitly **not** TaIrTe4 data.  The 1 and 100 MW/(m2 K) cases are
numerical sensitivity scenarios, not a confidence interval.

| scenario | G (W/m2K) | Ta Tmax (K) | ||dF/drho||2 | worst h=0.0025 AD–FD | analytic q/G jump (K) | series-control error |
|---|---:|---:|---:|---:|---:|---:|
| 1 MW | 1e+06 | 0.0138958981 | 1.5677404e-05 | 1.214e-07 | 0.2 | 1.388e-16 |
| 17.24 MW Au/MoS2 analogue | 1.72414e+07 | 0.0107254244 | 2.76036179e-05 | 3.700e-07 | 0.0116 | 1.495e-16 |
| 100 MW | 1e+08 | 0.00983128063 | 3.17018251e-05 | 5.500e-07 | 0.002 | 8.674e-16 |
| perfect | ∞ | 0.00958983083 | 3.33048952e-05 | 6.172e-07 | 0 | 0.000e+00 |

## Numerical gates

- worst fine-step AD–FD relative error: `6.171702e-07` (< 1%)
- worst explicit linear residual: `9.271857e-12` (< 1e-8)
- worst energy-balance error: `4.743385e-14` (< 1%)
- CPU linear-solve fallback: `False`
- fixed source power: `1.000000000000e-06 W`

The variation of Tmax and gradient norm across G scenarios is physical-model
sensitivity; the AD–FD error is numerical derivative error.  They are reported
separately.

## Remaining blockers before Au PTE inverse design

1. The currently validated FDTDX optical checkpoint contains Au/TaIrTe4/air,
   but not the SiO2/Si optical substrate.  The optical geometry must be made
   identical to the thermal stack and re-close before coupling.
2. A coupled TaIrTe4 + floating Au electrical operator must validate how Au
   conductivity and Au/TaIrTe4 electrical contact alter the weighting field.
3. The spatial Maxwell sensitivity must be contracted with the thermal/PTE
   adjoint, then combined physical-density and latent AD–FD must pass.
4. Au/TaIrTe4 thermal and electrical contact values remain scenario parameters
   unless device-specific measurements are supplied.

Raw NPZ is not committed to Git.  Its absolute path, byte size, SHA-256, and
generation command are recorded in `RAW_ARTIFACT_MANIFEST.json`.
