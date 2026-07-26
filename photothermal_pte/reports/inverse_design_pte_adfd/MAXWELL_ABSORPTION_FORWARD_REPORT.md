# Maxwell absorption forward checkpoint

**Status: `COMPLETED_MAXWELL_ABSORPTION_FORWARD_SMOKE`**

This is the first actual v261 run for the periodic inverse-design geometry.
It is not a disk/proxy run and it is not yet an AD–FD validation.

## Realized solver contract

- Lumerical v261 version: `8.35.4522`.
- GPU: local `NVIDIA RTX 6000 Ada Generation`.
- Periodic x/y cell: 6 um x 6 um.
- Analysis wavelength: 4 um.
- Source band: 3–6 um.
- Mesh: auto non-uniform, conformal variant 1, accuracy 5.
- TaIrTe4 z mesh: 5 nm.
- Design density: `241 x 241 x 13`, deterministic periodic test field with
  range `[0.38, 0.62]`.

## Absorption discretization

The native FieldRegion components use the validated shifted coordinate
contract:

- Ex on `x + delta_x`;
- Ey on `y + delta_y`;
- Ez on `z + delta_z`.

Each component is integrated using the shifted-coordinate 3-D trapezoid
quadrature. This is intentionally distinct from the Yee volumes used by the
design-monitor epsilon sensitivity.

The loss tensor is taken from the v261 fitted material response, not from an
independent table interpolation:

`Im(epsilon) = [50.85010970213534, 9.289194655416972, 0]`.

For source amplitude 1 V/m in air,

`I_inc = 0.5 epsilon0 c |E0|^2
       = 1.3272093648958553e-3 W/m2`.

Reported absorption is divided by this value:

- `P_Qx / I_inc = 8.763584966562064e-12 m2`;
- `P_Qy / I_inc = 2.98099632534266e-14 m2`;
- `P_Qz / I_inc = 0 m2`;
- `P_Q / I_inc = 8.79339492981549e-12 m2`.

The nonconstant frozen native-Yee certificate weight gives
`F_weighted = 9.367200868199787e-12 m2`. It is a Maxwell transpose test
weight, not yet the thermal-adjoint pullback.

## Provenance and correction

The EM solve completed successfully after 245567 iterations and
`3.901599e-12 s` simulated time. The first result-extraction attempt used the
monitor-style `getdata` API for FieldRegion half steps. FieldRegion exposes
these values through `getresult`; the extraction path was corrected and the
same completed FSP was postprocessed without another EM solve.

The raw FSP, engine HDF5, log, and postprocessed NPZ remain outside Git.
Their exact paths, sizes, and SHA-256 values are in
`MAXWELL_ABSORPTION_RAW_ARTIFACT_MANIFEST.json`.

## Remaining gate

No Maxwell gradient claim is made at this checkpoint. Next:

1. import the exact Wirtinger source through FieldRegion;
2. verify the periodic source pairing and profile round trip;
3. run the measured 27-color `rho -> epsilon` transpose;
4. compare one predeclared physical-density direction against central FD.
