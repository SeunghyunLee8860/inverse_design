# Au topology-density and constitutive-law audit

Status: `O3_REMOVED_NK_RELAXATION_IMPLEMENTED_B200_VALIDATION_PENDING`

## Historical defect

The historical FDTDX optimization used

```text
optical Au oscillator strength: rho**3
thermal/electrical Au coefficients: rho
```

This was not one consistent relaxed device. More importantly, the cubic
optical law had no Au-specific derivation. Its origin is the structural SIMP
penalization used to discourage gray elastic-density elements; it is not a
constitutive law for Au Drude/interband response.

`material_fraction.py` later changed the historical FDTDX path to a shared
linear fraction. That was a useful software-consistency diagnostic but is not
the selected Lumerical optical model.

## Current design-state rule

One filtered/projected topology occupancy `rho_bar` is shared by all physics.
It is a numerical relaxation, not electron/hole density and not a claim that a
fabricated cell contains a homogeneous gray Au alloy.

The physical maps are deliberately separated:

```text
rho_bar -> optical epsilon(omega,rho_bar)
rho_bar -> thermal k and interface conductance
rho_bar -> electrical sheet/contact conductivity
```

Sharing a topology state does not require unrelated physical coefficients to
share one exponent.

## Selected optical law

The solver-free implementation in `au_density_relaxation.py` uses the
Christiansen metal/dielectric interpolation

```text
n = n_bg + rho_bar (n_Au-n_bg)
k = k_bg + rho_bar (k_Au-k_bg)
epsilon = (n+i k)^2
```

At `rho_bar=0` it is exactly the background endpoint. At `rho_bar=1` it is
exactly the frozen Ordal 4-um Au endpoint. Its analytic complex derivative is
`2(n+i k)[(n_Au-n_bg)+i(k_Au-k_bg)]`. No `rho**3` appears.

This nonlinear interpolation was developed specifically to reduce artificial
field amplification in metallic topology optimization:

- Christiansen et al., CMAME 343, 23-39 (2019),
  DOI 10.1016/j.cma.2018.08.034;
- Zeng et al., ACS Photonics 8 (2021),
  DOI 10.1021/acsphotonics.1c00260.

The equation is a mathematically smooth, passive single-frequency relaxation.
It is not yet a production certificate: passivity alone does not rule out an
intermediate-density field resonance in the full device.

## Binarization

Gray removal is assigned to the existing finite filter, tanh projection,
beta continuation, robust eta scenarios, and final 500-nm solid/void audit.
It is not assigned to an arbitrary cubic material law.

The final thresholded mask must be reevaluated in Lumerical using ordinary
sampled-data dispersive Au, not the continuous relaxation.

## Remaining gates

- B200 uniform-density field/Q sweep over `rho_bar in [0,1]`;
- imported full-Au versus ordinary sampled-data Au endpoint and bandwidth
  parity;
- component-Yee material Jacobian FD and transpose tests;
- optical and complete latent AD-FD for both polarizations;
- optical/thermal/electrical mesh convergence;
- thermal/electrical mixture-law and void-floor sensitivity;
- exact-binary ordinary-Au final reevaluation.

All historical O3/TE1, shared-linear FDTDX, and mesh tables remain diagnostics
only and cannot clear these Lumerical gates.
