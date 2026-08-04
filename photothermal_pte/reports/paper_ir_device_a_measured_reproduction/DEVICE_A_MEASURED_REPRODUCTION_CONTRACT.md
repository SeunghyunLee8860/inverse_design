# Device-A measured 11-µm reproduction contract audit

## Outcome

Status: `READY_DEVICE_A_MEASURED_REPRODUCTION_CONTRACT`

The three supplied documents were audited.  A paper-consistent substrate
scenario is now defined as **Kitamura-2007 fused silica for the 285-nm SiO2
film plus Lumerical Palik Si as an explicit closure**.  The paper cites
Kitamura for the silica phonon but does not identify a numerical Si optical
database, so this is not described as an exact hidden author input.

The requested calculation can test the optical/thermal/current trend and the
published polarization ratio.  It cannot certify an exact absolute current
without unpublished beam, CAD, contact-resistance, and scan-position data.
No empirical optical-power or current rescaling is allowed.

## Audited documents

| Document | Pages | SHA-256 |
|---|---:|---|
| Main paper | 10 | `ad160823ce0805e709be2ea54c663a51280e56c498d76c8d33651599b8733155` |
| Supporting Information | 14 | `c2ebe6bfbf00f954e0ccaa88431eec059db7d25beeaa4622ceb3c02d83b7f99c` |
| Blevins thesis | 164 | `06e159965f9ab0ca5b7dc9d601d9206d18f897d287e48f073cff1c6e7f487f30` |

## Fixed published Device-A inputs

- TaIrTe4 thickness: 130 nm.
- Substrate: 285 nm thermally grown SiO2 on Si (main prose also says nominal
  300 nm; AFM/figure value 285 nm is used).
- Electrodes: 5 nm Ti / 50 nm Au.
- Wavelength: 11 µm; normal incidence.
- Time-averaged incident power: 284.40 µW from SI; 285 µW is the rounded
  main-figure caption.
- Objective: 40x reflective, NA=0.4; QCL stated spot range 9–16 µm.
- The documents do not state whether spot size means FWHM, 1/e2 diameter, or
  radius and do not tabulate the 11-µm realized beam profile.
- TaIrTe4 parameters: kappa(a,b,c)=(14.4,3.8,1.0) W/(m K),
  sigma(a,b)=(4.91e5,1.10e5) S/m, S(a,b)=(-6,27) µV/K,
  G(TaIrTe4/thermal-SiO2)=7.37e6 W/(m2 K), and G(TaIrTe4/air)=1 W/(m2 K).

## 11-µm substrate optical constants

| Scenario | SiO2 n+ik | SiO2 epsilon | Interpretation |
|---|---|---|---|
| Paper-consistent Kitamura | 2.019443683 + 0.162620219i | 4.051707452 + 0.656804749i | Production reproduction scenario |
| Existing Palik fitted readback | 1.988540859 + 0.045056539i | 3.952264656 + 0.179193536i | Preserved comparison only |

Kitamura loss k is 3.609 times the existing fitted Palik value at 11 µm, so the prior Palik Maxwell artifact is not reused as the paper-consistent result.
Si remains Palik (`n=3.421289622+4.38988031e-05i`) and is explicitly marked as an unpublished closure.

## Current equation and absolute-current gate

The implementation uses `x=b`, `y=a`,
`Jx=-sigma_b S_b dT/dx`, `Jy=-sigma_a S_a dT/dy`, followed by the
Shockley–Ramo volume integral.  This matches SI Eq. S5–S7.  The reduced
thermal reference uses the paper's top-air and bottom-SiO2 Robin boundaries;
bulk SiO2/Si thermal cells are not silently claimed to be the paper model.

Using the frozen Figure-2 digitization, published conductivities, 130-nm
thickness, and no fitted contact resistance predicts
`R=14.139 ohm`, versus measured
`213 ohm` (93.36%
difference).  Therefore absolute-current magnitude is fail-closed as
`BLOCKED_DIGITIZED_GEOMETRY_RESISTANCE_MISMATCH`; the computed current is not
renormalized to 213 ohm.

## Experimental comparison targets and paper inconsistency

- Figure 3J digitization: `|Ia|/|Ib|=0.836590 ± 0.008526`
  at 11 µm, or `|Ib|/|Ia|=1.195329`.
- Figure 3H and SI Figure S5 map colorbars are in pA (roughly ±200 pA at
  11 µm), while extracted Figure 3I labels its profile axis as nA.  Those
  differ by 1000x and cannot both be literal.  Absolute comparison therefore
  reports both the plotted value and the likely pA interpretation rather than
  silently choosing the nA label.
- SI Figure S5 independently fits the 11-µm off-axis E||a response to
  `I0=129 pA`, `tau=26±3.3 µs`.  At the reported 3675-Hz chopper frequency
  this fit gives `110.599 pA`.
  This supports interpreting the Figure-3I minima (visually about 122 and
  143 plot units for a and b) as pA, while retaining the printed nA typo as a
  source inconsistency.
- The SI/thesis emphasize simulated current pattern robustness; the exact
  COMSOL CAD, beam profile, objective transmission, and tabulated scan
  coordinates are absent.

## Approved execution matrix

1. Same-substrate empty-stack references for E||a and E||b.
2. Digitized Device-A off-axis edge for E||a and E||b.
3. Scalar Gaussian, explicit assumed w0=8.75 µm, 50-µm source aperture,
   60-µm six-PML domain, existing boundary-aware local mesh, no CPU fallback.
4. Preserve full raw Q; use only material-overlap-attributed TaIrTe4 power for
   the paper-reduced thermal calculation.  No clipping, smoothing, gain,
   polarization matching, or global rescaling.
5. Use exact 284.40 µW and the paper-reduced Robin thermal operator, then the
   digitized weighting field and Shockley–Ramo current.
6. Judge numerical gates, `|Ia|/|Ib|`, signs/maps, and absolute pA separately.

This is a **paper-like measured Device-A reproduction with explicit closures**,
not an exact paper-certified recreation and not an inverse-design result.
