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

The v7 continuation keeps DFM pressure active from beta 1 and activates local
KS-opening inequalities at beta 4, but it does not allow a nearly feasible
low-beta surrogate to cause an unbounded repair loop. At beta 4--16, an exact
target streak plus an objective/design plateau is the preferred promotion
path; after at most three same-beta stage attempts, a non-worsening exact audit
plus the same plateau advances beta. The exact thresholded audit remains
diagnostic at beta 1 and 2 because a gray field has no unique binary topology.
From beta 32 onward, feasible KS constraints and zero exact violations are
strictly mandatory before beta can increase or the run can terminate.

The beta-1 KS weight is fixed from an offline gradient-scale audit of the
uniform Ea checkpoint. A unit KS weight was 0.957% of the physical objective
gradient L2 norm; therefore weight 1 is used at beta 1 and grows as beta^2
until explicit hard inequalities take over from beta 4. This is a recorded
optimizer scaling choice, not an AD/FD gradient rescaling.

Each accepted full-physics evaluation publishes a JSON checkpoint and PNG.
Raw FSP/NPZ artifacts remain outside Git.

Production launch uses two independent `runres` parents invoking `run_one.py`.
Each parent owns one physical GPU claim and nine FDTD licenses for the entire
optimization, including Python-only thermal/electrical intervals. The earlier
single-supervisor launcher is retained as code provenance but is not used for
the production v3 runs because it could only advertise one GPU to the site
scheduler.
