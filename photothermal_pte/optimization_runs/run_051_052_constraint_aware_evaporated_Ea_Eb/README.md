# Run 051/052 — constraint-aware evaporated-SiO2 Ea/Eb restart

These are fresh, independent top/bottom-electrode optimizations from a uniform
latent density of 0.5. Run 051 maximizes the signed terminal PTE current under
`E || a`; Run 052 uses `E || b`.

The physical contract is unchanged from Runs 049/050:

- Lumerical coordinates: `x=b`, `y=a`, `z=c`.
- Top/bottom electrical contacts, with bottom weighting potential 0 and top 1.
- Evaporated-SiO2 TaIrTe4 interface conductance: `7.37e4 W/(m^2 K)`.
- No connectivity, symmetry, or volume-fraction constraint.
- 500 nm minimum solid and void feature contract.
- NLopt `LD_MMA`; no Adam, manual move limit, normalized gradient direction,
  clipping, or post-update morphology edit.

The new continuation differs from the superseded runs in one important way:
beta is not promoted by objective stagnation alone. DFM pressure is active from
beta 1, explicit local KS-opening inequalities start at beta 4, and beta
promotion additionally requires the stage-specific exact 500 nm audit target.
The exact thresholded audit remains diagnostic only at beta 1 and 2 because a
gray density has no unique binary topology. From beta 32 onward zero exact
violations are mandatory before beta can increase.

The beta-1 KS weight is fixed from an offline gradient-scale audit of the
uniform Ea checkpoint. A unit KS weight was 0.957% of the physical objective
gradient L2 norm; therefore weight 1 is used at beta 1 and grows as beta^2
until explicit hard inequalities take over from beta 4. This is a recorded
optimizer scaling choice, not an AD/FD gradient rescaling.

Each accepted full-physics evaluation publishes a JSON checkpoint and PNG.
Raw FSP/NPZ artifacts remain outside Git.
