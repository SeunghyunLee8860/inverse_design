# Robust high-beta Au dual-polarization continuation

Status: `BLOCKED_ROBUST_PROJECTION_EXACT_BINARY_SIGN`

The requested objective is `I_a > 0` and `I_b < 0` for left/right terminals.
The optical wavelength is 4 um, the incident Gaussian waist is 4 um, and the
Au design region is 8 x 8 um on a fixed 16 x 16 um TaIrTe4 flake.  The
eroded/dilated projection pair uses eta=0.65/0.35 and a 500 nm solid/void
filter.  No symmetry, volume-fraction, or connectivity constraint is used.

The beta-80 checkpoint was continued through beta=96, 128, 192, and 256.
The robust continuous design retained the required signs while nominal
grayness decreased from 3.81% to 0.395%.  At beta=256 the returned robust
minimum `min(I_a,-I_b)` was 1.881534 nA, but the independent exact 500 nm
audit still found 28 violating cells before repair.

Six exact-binary, exact-500-nm targeted-repair candidates were then evaluated
with independent Ea and Eb forward Maxwell/thermal/electrical solves.  Every
candidate failed the required Eb sign: Ia was +6.67 to +6.71 nA and Ib was
+8.80 to +8.85 nA.  No candidate is promoted.

This result demonstrates that the remaining sub-percent gray boundary is not
an innocuous visualization artifact: it controls the Eb current sign.  More
beta continuation or empirical rescaling is therefore not accepted as a
binary inverse-design certificate.  The next method must optimize an exact
binary, 500-nm-feasible parameterization directly.

Raw checkpoints and exact designs remain outside Git.  Their paths, sizes,
and SHA-256 values are recorded in `RAW_ARTIFACT_MANIFEST.json`.  No Q
clipping, smoothing, gain, or global rescaling was used.
