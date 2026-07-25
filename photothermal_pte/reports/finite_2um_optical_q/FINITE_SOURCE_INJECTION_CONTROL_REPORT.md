# Finite source injection control report

## Decision

The source is a finite Gaussian beam, not a plane wave. TFSF passed the v261
project-property audit but the actual GPU engine stopped with:

`Error: GPU simulation does not support the use of TFSF sources.`

The required GPU contract therefore rules out TFSF. The Gaussian implementation
uses the same 3–6 µm broadband range and 4 µm evaluation point.

## Implemented measurement

The empty SiO2/Si layered-stack control records complex E/H on an air plane
50 nm above the nominal flake. The transverse fields are decomposed into the
downward-traveling component. Central incident intensity, total source power,
flake-footprint peak/mean/minimum, and aperture-edge intensity are stored.

Material cases must name a matching empty-stack `case_result.json`. Domain,
PML layers, flake dz, source span, waist, and polarization are checked before
using the measured intensity. The unit-response scale is central
`I_inc = 1 W/m²`; it is not adjusted to make volume Q match flux.

## Controls to date

Generation commit: `657e87a13871850852898f65ea655853c2c1a7a0`

- Zero-amplitude Gaussian source, 8 µm domain / 24 PML layers / 5 nm dz /
  2 µm waist: maximum recorded inside and outside `|E|² = 0`; pass.
- First empty-layered-stack run with a 6.0 µm aperture: volume Q was exactly
  zero, central downward incident intensity was measured as
  `6.793222247592325e-4 W/m²`, and the opposite x/y lateral face powers were
  symmetric to better than numerical reporting precision.
- That first empty run is not accepted because aperture-edge intensity was
  7.339% of central intensity, above the 5% clipping gate.

The nonzero lateral power of a finite Gaussian beam is expected diffraction.
It is recorded, not deleted. The empty-stack lateral control therefore checks
opposite-face symmetry and the closed-box energy residual. Treating all
physical Gaussian diffraction as unexpected leakage and requiring it to be
zero would be an invalid plane-wave assumption.

The next control uses a 6.8 µm aperture with the same 2 µm waist. No material
case is accepted until that empty-stack run passes.
