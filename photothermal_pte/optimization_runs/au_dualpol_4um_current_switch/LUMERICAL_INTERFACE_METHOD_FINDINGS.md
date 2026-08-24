# Lumerical metal-interface method findings

Date: 2026-08-24. These are RTX 6000 Ada development diagnostics at one
fixed 5-nm thin-stack / 50-nm bulk-z mesh for Ea exact-empty and exact-full
Au controls. They are not a mesh-convergence or B200 certificate. Raw
FSP/JSON/NPZ inputs and the 119.5-MB comparison output remain outside Git
under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.

## Controlled comparison

Script `30_validate_lumerical_4um_interface_methods.py` hash-verifies six
passed Lumerical v261 solver `8.35.4413` runs: CV0, CV1, and staircase for
both exact-empty and exact-full Au. Every run has the same physical geometry,
Ea source, calibrated source contract, 100-nm flake x/y mesh, 5-nm stack-z
limit, 50-nm bulk/air/PML-z limit, 20 x 20 x 6 um domain, eight PML layers,
1-ps duration, RTX GPU UUID, and solver version. Every full-Au run uses MCM6.

Maxwell quantities are divided by each method's own passed source-only
incident power. The official multi-material `pabs_adv`/`index_x` mask is then
applied without redistributing unidentified conformal samples. Identified
Au, TaIrTe4, and SiO2 power is conservatively mapped only to matching custom
thermal materials and passed to the repository CUDA thermal/electrical PDE
solvers. No Lumerical HEAT/CHARGE, rescaling, clipping, smoothing, gain, or
alternative Maxwell solver is used.

## Material-assignment result

| interface method | empty unassigned Pabs | full unassigned Pabs | empty Tmax | full Tmax | official material-assignment gate |
|---|---:|---:|---:|---:|:---:|
| CV0 | 11.8313% | 7.7844% | 0.90589 K | 0.05560 K | fail |
| CV1 | 6.2624% | 11.2663% | 0.98045 K | 0.06022 K | fail |
| staircase | 0.001012% | 0.195399% | 1.03393 K | 0.06290 K | pass |

The gate is less than 0.5% unassigned absorption in both controls. CV0 and
CV1 produce mixed-index conformal samples that the exact Ansys material mask
correctly refuses to relabel as one bulk material. Staircase assigns one
material to almost every sample and therefore supplies a well-defined
material-resolved heat source. The missing CV0/CV1 power is deliberately not
renormalized into the identified materials; doing so would fabricate its
spatial and material distribution.

## Maxwell comparison

At this fixed mesh, CV0 and staircase agree closely in the Maxwell solution:

| control | normalized Q change | normalized flux change | complex E NRMSE | E2 NRMSE |
|---|---:|---:|---:|---:|
| empty | 0.14491% | 0.14488% | 0.03662% | 0.01298% |
| full Au | 0.08776% | 0.08761% | 0.02349% | 0.01585% |

All eight metrics pass the 0.5% development gate. CV1 does not agree with
staircase: its empty/full normalized-Q changes are 1.7365%/3.2465%, complex-E
NRMSE values are 1.4554%/2.0877%, and E2 NRMSE values are
2.3120%/2.3048%. This is consistent with Ansys' warning that CV1 includes
metal interfaces in conformal meshing and can introduce metal artifacts.

The CV0/staircase downstream fields do not agree at the same level because
the official filter drops the large CV0 unidentified fraction. Empty/full
remapped-Q NRMSE is 27.4892%/20.6631% and TaIrTe4-temperature NRMSE is
11.9996%/8.4407%. These differences are evidence against using the CV0
material partition, not evidence that the 5/50-nm staircase mesh is already
converged.

## Decision and remaining blockers

Staircase is selected only as the next linked-z refinement candidate because
it simultaneously:

1. agrees with CV0 in all tested source-normalized Maxwell observables below
   0.5%; and
2. assigns official material-filtered absorption with less than 0.5%
   omission for both empty and full controls.

This is explicitly **not** a final choice of mesh. A matching staircase
5/50-to-2.5/25-nm linked refinement was therefore run with new source-only,
exact-empty, and exact-full Lumerical FDTD results. Every individual run
passed, but the pair did not converge. Empty normalized Q/flux/complex-E/E2
changes were 0.9522%/0.9541%/0.6656%/1.1752%; full-Au changes were
1.3921%/1.3954%/1.0105%/1.1966%. Every value exceeds the 0.5% gate. The
downstream validator correctly stops before custom PDE comparison when the
Maxwell sub-gate fails.

The next bounded step is a matching staircase 1.25/12.5-nm source, empty, and
MCM6 full set, followed by the 2.5/25-to-1.25/12.5-nm pair. Ea
symmetry-current cancellation at the fixed 5/50-nm mesh is still only about
`5.42e-5` (empty) and `5.68e-4` (full), versus the one-ppm diagnostic gate,
so its cause must remain visible once a staircase Maxwell pair passes. Eb,
simple-L, final topology, x/y, thermal/electrical mesh, and B200 certification
remain open.
