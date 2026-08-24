# Lumerical z-mesh downstream multiphysics findings

Date: 2026-08-24. These are RTX 6000 Ada development diagnostics for the
linked 1.25/12.5-to-0.625/6.25-nm Ea exact-control pair. They are not B200 or
production evidence. Raw JSON/NPZ inputs and outputs remain outside Git under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.

## Solver and remap definition

Script `28_validate_lumerical_4um_z_multiphysics_pair.py` first calls the
hash-verifying Maxwell comparator. It then scales each raw Lumerical native
Yee Q bundle by its own measured source-only incident power to the common
285-uW reporting power. Thermal and electrical calculations use only the
repository custom CUDA finite-volume and weighting-potential solvers;
Lumerical HEAT and CHARGE are not called or licensed.

A first, deliberately retained raw diagnostic remapped each entire Yee dual
cell to every overlapping thermal cell. It conserved total power to about
1e-15 but spread conformal interface loss into thermal air. That created
mesh-dependent low-conductivity air hotspots: empty Tmax changed from 4.777 K
to 2.412 K and full Tmax from 0.112 K to 0.0747 K. This is a remap artifact,
not an accepted physical result.

The selected fail-closed diagnostic is material-aware. For each Yee
component it uses saved `Q/Im(epsilon_effective)` as the positive field-loss
factor, multiplies by Lumerical's finite-dt fitted material loss and the exact
dual-cell/physical-material overlap, and conservatively maps that power only
to thermal cells of the same material. No local or global closure rescaling,
clipping, smoothing, gain, or tiling is applied. Consequently the difference
between reconstructed physical-material power and Lumerical native total Q is
an explicit gate rather than an error hidden by renormalization.

## Results

Every individual remap conserves its reconstructed power to better than
3e-15. Native integration reproduces the source JSON Q to better than 4e-16.
Both thermal solves pass residual and energy-balance gates; both electrical
solves pass residual and terminal-balance gates.

| exact control | coarse material-Q reconstruction error | fine material-Q reconstruction error | remapped Q volume-L2 NRMSE | TaIrTe4 temperature NRMSE | Tmax change | result |
|---|---:|---:|---:|---:|---:|:---:|
| empty, Ea | 1.5433% | 0.7777% | 0.9730% | 1.0058% | 0.9993% | fail |
| full Au, Ea | 2.8468% | 1.5504% | 1.8576% | 1.7397% | 1.1321% | fail |

Material-aware empty Tmax is 1.02946 K on the coarse member and 1.03985 K on
the fine member. Full-Au Tmax is 0.062581 K and 0.063298 K. These values no
longer contain the rejected thermal-air hotspot, but their relative changes
still exceed the 0.5% contract.

Empty and full are mirror-symmetric controls, so their true x-directed
current is zero. A relative change or sign test on a near-zero number is
ill-conditioned. The validator instead requires the net current to cancel to
one part per million of the integrated absolute local-current scale. The
worst observed cancellation ratio is 2.23e-8, so the symmetry-current gate
passes. Non-symmetric simple-L and design controls must still use the ordinary
relative signed-current and sign-preservation gates.

## Consequence

The 1.25/12.5-to-0.625/6.25-nm pair passes total Q, six-face flux, and common
endpoint-plane Maxwell metrics, but it does not pass the material-resolved Q
or downstream thermal gates. The z axis therefore remains blocked and x/y
convergence or optimization must not start.

The saved effective epsilon and total Q do not uniquely provide exact
physical-material absorption inside Lumerical conformal cut cells. The
decreasing reconstruction error with refinement is useful evidence, but
blindly extending to 0.3125/3.125 nm would approximately double an already
53.6-million-point grid and would not resolve the definition problem. The
next task is to establish a Lumerical-native material-resolved absorption
extraction or an independently converged interface method (including the
required MCM6 CV0/CV1/staircase axis), then repeat this downstream pair. Do
not close the gap by rescaling reconstructed material power to native Q.
