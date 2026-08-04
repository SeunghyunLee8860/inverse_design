# Device-A three-position scan contract

Status: `DEVICE_A_THREE_POSITION_SCAN_CONTRACT_FROZEN_NOT_SOLVED`

This checkpoint freezes the low-cost position-sensitivity test. It does not
run Maxwell, thermal, PTE, adjoint, AD-FD, optimization, or empirical fitting.

The scan origin is the digitized off-axis TaIrTe4/air edge midpoint
`[-12.099447513812155, 2.375690607734806]` um. Increasing `s` follows the inward unit normal
`[0.8053368838936223, -0.5928174284216942]`. Thus `s=0` is the digitized edge, and the pre-registered
single-position beam lies at `s0=3.000000000` um. The exact experimental stage
coordinate is unavailable; this is the Figure-3H black-dotted-line counterpart
on the figure-digitized geometry.

Across all three cases the Device-A polygons, PML, monitors, simulation origin,
and local 50-nm mesh remain fixed. Only the scalar-Gaussian source center moves.
The three signed coordinates are 2, 3, and 4 um from the edge. The minimum
source-aperture/PML clearance remains 1.742447 um.

The `s0` optical and thermal artifacts are immutable and reused. At `s0±1 um`,
only the a/b GPU optical cases are new (four solves maximum). A separate Au/Ti-
off `E||a` diagnostic is conditional on the position cases passing their gates.
