# Run 010 combined physical-density AD-FD checkpoint

Status: `VALIDATED_TAIRTE4_FLAKE_COMBINED_PHYSICAL_RHO_ADFD`

The uniform `rho=0.5`, `E || a` control validates the complete derivative
chain used by the TaIrTe4-to-void topology optimization:

1. physical density to component-specific Yee permittivity,
2. Maxwell volumetric absorption to the explicit thermal grid,
3. density-dependent anisotropic thermal conduction and bottom contact,
4. temperature to the density-dependent electrical weighting solve, and
5. the direct and implicit electrical terms.

The Maxwell adjoint used the spatial thermal cotangent on the native Yee
component grids. It did not reuse a scalar absorbed-power source. No empirical
normalization, gradient rescaling, Q clipping, smoothing, gain, or rescaling
was used.

## Directional derivative result

- central-difference step: `0.005`
- adjoint directional derivative: `1.47654908052e-17 A`
- finite-difference directional derivative: `1.47654481930e-17 A`
- relative AD-FD error: `2.88593210e-6`
- permitted error: `< 1e-2`

## Numerical gates

- component-Yee transpose error: `4.06904e-16`
- maximum forward/adjoint coordinate mismatch: `0 m`
- worst optical six-face closure: `6.66398e-6`
- worst Q-mapping conservation error: `1.99614e-16`
- worst thermal forward/adjoint residual: `7.08866e-11`
- worst thermal energy-balance error: `1.49431e-12`
- worst Maxwell auto-shutoff: `9.11927e-8`

All Maxwell solves ran through the licensed GPU resource on physical GPU 5;
all thermal and electrical sparse solves ran on CUDA. The completed baseline
forward was reused, followed by one new spatially weighted adjoint and two new
finite-difference forwards. Runtime was 186.0 s.

## Interface interpretation

Run 009's upper design material was `air <-> SiO2`, so its design-face law
interpolated `G_air` and `G_SiO2`. Run 010 instead changes the flake itself
from void to TaIrTe4 above fixed SiO2. Its bottom gray face is therefore an
area-fraction parallel relaxation between an air/SiO2 path and a
TaIrTe4/SiO2 path with `G=7.37e6 W/(m2 K)`. These are different physical
contracts; the Run 009 law is not declared invalid.

Raw FSP/NPZ artifacts remain outside Git and are listed with byte sizes and
SHA-256 values in `RAW_ARTIFACT_MANIFEST.json`.
