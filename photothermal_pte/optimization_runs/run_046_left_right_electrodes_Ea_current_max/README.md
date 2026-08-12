# Run 046 — left/right electrodes, E||a current maximization

This run changes the electrical terminal pair and fixed TaIrTe4 contact strips
from top/bottom to left/right.  The coordinate contract remains Lumerical
`x=b`, `y=a`, `z=c`; therefore `E||a` is a 90-degree source polarization.

- left terminal: weighting potential `psi=0`
- right terminal: weighting potential `psi=1`
- design region: 20 x 24 um, 201 x 241 nodal density variables
- fixed TaIrTe4 contact strips: x in [-12,-10] um and [10,12] um
- no symmetry, volume, or connectivity constraint
- 500 nm solid and void minimum-feature audit
- beta continuation: multiply by two after an audited FOM plateau
- exact cleanup: only after beta>=16, gray fraction<5%, and both FOM and
  fabrication metrics plateau; solid-first and void-first exact-zero
  candidates are then evaluated with fresh unrescaled full physics

Raw FSP/NPZ artifacts remain under `/data/seunghyun/tairte4/artifacts/` and
are not committed.  `results/` receives per-evaluation JSON and PNG files.

