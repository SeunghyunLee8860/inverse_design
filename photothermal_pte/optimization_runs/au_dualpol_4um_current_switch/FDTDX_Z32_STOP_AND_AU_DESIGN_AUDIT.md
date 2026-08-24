# FDTDX z32 stop certificate and Au-design audit

## Decision

The fresh increment-state FDTDX exact-binary optical reference is **not mesh
converged**.  The final practical z-only pair, z16 to z32, fails the unchanged
component-Q and material-region complex-field gates.  No optical mesh is
selected, z64 is forbidden, and no FDTDX adjoint, thermal/electrical
propagation, gray-density optimization, or inverse-design restart is allowed.

This document covers only the historical/fresh FDTDX investigation.  A
different Codex session owns the Lumerical work.  Nothing here modifies,
runs, or reinterprets that session's Lumerical files.

## Exact z32 numerical contract

The z32 case changes only the full-domain z refinement factor relative to
z16.  Source, x/y grid, material geometry, PML policy, time window, and
Courant factor remain fixed.

- case-contract SHA-256:
  `971d106e87fdd66000ac0cbaaf6c430a0710b6363a2583ce84f3cdbed79ff8b0`
- grid: `196 x 196 x 1280`, or `49,172,480` Yee cells
- time: `307,215` steps, `dt=1.0423360591091103e-18 s`, 24 periods total,
  4-period terminal phasor window, Courant `0.5`
- x/y design and outer pitch: `100 nm`; x/y PML pitch: `125 nm`
- bottom PML: 256 cells at `6.25 nm`
- Si: 160 cells at `6.34375 nm`
- SiO2: 96 cells at `2.96875 nm`
- TaIrTe4: 160 cells at `0.625 nm`
- Au: 64 cells at `0.78125 nm`
- near air: 128 cells at `1.5625 nm`
- middle air: 64 cells at `7.8125 nm`
- source air: 96 cells at about `6.7708 nm`
- top PML: 256 cells at `6.25 nm`

The resulting aspect ratio is itself a warning: the Au layer is refined below
1 nm in z while its lateral pixel remains 100 nm.  z32 is a convergence
diagnostic, not a balanced production discretization.

## External z32 artifacts

All raw artifacts remain outside Git under `/home/seunghyun200/fdtdx_results`.
The project worktree and patched FDTDX fork were clean for every provenance-
bound solve.  Ea and Eb ran concurrently on physical GPUs 6 and 7.  An
unrelated Lumerical process on GPU 0 was observed and never touched.

Source root:
`/home/seunghyun200/fdtdx_results/increment_state_source_36bb0a2a_z32_t24/`

- Ea source report:
  `8356dc9e7e201fcbf2132e764524b583dc791ccfca5fbbb2acf69ee318830a9f`
- Ea source NPZ:
  `c8ebc4d10ee8a9cba8d75c324057add8dcea98c7d2db2e26dea6766a052eea8a`
- Eb source report:
  `f865df8deae807924f136eced83f8ebd8c2de3aa68d7b8e0dbbc1cadf1dcd302`
- Eb source NPZ:
  `eb98c3348925209cbcd2d62e00f1f8879e4c8a6788f6afead8b50aa701e9c7db`
- source-pair certificate:
  `e926729fe75cf5fa8fcd3a10e24137037963c2f218968262f663d1c62f2d4f6b`
- unscaled incident power: exactly `1.883802575736171e-12 W` for each
  polarization; mismatch zero
- one common power scale: `151289738.99434492`
- one common field scale: `12299.989390009445`

Material root:
`/home/seunghyun200/fdtdx_results/increment_state_material_36bb0a2a_fullz_z32_t24/`

- Ea report:
  `f13c9ee53fd2b8fc5209324439a0a406c8d51ac879c44c5da3b4d316590eedc1`
- Ea NPZ:
  `a54761e889302086c61044ae1365be5b77d42434a9718ad29b35a1e208df20a6`
- Eb report:
  `ffc23d15c448a2bb7002ae6f27231beef223aec8d38493326d1491536754847d`
- Eb NPZ:
  `71f2f4afa796851b851fbcb560911e12701f5b73b4b1b7469b5024682c3f91ab`
- Ea late total unscaled Q: `4.4796321696812157e-13 W`
- Eb late total unscaled Q: `7.686414094667418e-13 W`

Both material reports are ready.  Every source binding, exact-binary material
readback, stationarity, Q/flux closure, finite-value, GPU, FDTDX-commit, and
clean-worktree gate passes.  The mask has 375 solid design cells, accepts only
integer/bool 0/1 occupancy, uses ordinary-Au or air coefficient endpoints,
and records `gray_density_allowed=false` and `rho_power=null`.

Certificate root:
`/home/seunghyun200/fdtdx_results/increment_state_z32_extension_certificate_1cebc11e/`

- certificate:
  `FDTDX_INCREMENT_STATE_FULL_Z32_EXTENSION_CERTIFICATE.json`
- certificate SHA-256:
  `079a6fbbb78aeab29d5e7460815f22208708a307f02572dc956f244433b9bb97`
- generator project commit:
  `1cebc11e5a10b2f0406683614e66cbd15198d248`
- generator SHA-256:
  `7a1385cd2606566f8c7592a250d22eb5e791629389ef46f819c1a2bfea7351ef`
- status: `BLOCKED_FDTDX_INCREMENT_STATE_Z16_TO_Z32`
- all seven global artifact/provenance checks: true

## Runtime and inverse-design feasibility

The two polarizations were parallelized across GPUs 6 and 7.  Adding more
GPUs cannot shorten one polarization's single-device JAX graph.

| case | Ea cold forward | Eb cold forward | Ea total | Eb total | pair wall | peak memory |
|---|---:|---:|---:|---:|---:|---:|
| z32 source | 1110.469 s | 1108.378 s | 1153.681 s | 1151.012 s | about 19 min 14 s | 33.68 GiB/GPU |
| z32 material | 1108.540 s | 1107.521 s | 1157.439 s | 1158.760 s | about 19 min 19 s | 33.68 GiB/GPU |

One two-polarization forward is therefore about 18.5 minutes of solver time
with two GPUs.  Even if one adjoint cost only one forward, a single nominal
optical forward-plus-adjoint iteration would be about 37 minutes before
thermal/electrical work.  Three robust projections give a lower bound near
111 minutes per iteration.  One hundred such iterations would exceed 185
hours before coupled-PDE costs.  z32 cannot be an optimizer mesh under the
user's practicality requirement.

Measured z2/z4/z8/z16/z32 material forward times are approximately
`15.5/36.6/101.7/320.4/1108.5 s`.  Extrapolating this sequence places z64 far
above the accepted per-forward cutoff.  z64 is not an allowed next action.

## z16-to-z32 result under unchanged gates

The z16 certificate was rehashed and required to remain blocked.  Its z16
reports and raw NPZ hashes were rebound byte-for-byte before z32 comparison.
The comparison uses the same predeclared gates and the same fine-to-coarse
physical-z field interpolation/component-Yee Q restriction as the earlier
ladder.

| metric | limit | z8 to z16 | z16 to z32 | gate |
|---|---:|---:|---:|---|
| source-power relative change | 0.5% | 0.0168% | 0.00337% | pass |
| Q/closed-flux relative error | 2% | 0.0620% | 0.0182% | pass |
| late-field stationarity NRMSE | 0.5% | 0.00878% | 0.00901% | pass |
| total-Q relative change | 1% | 0.6306% | 0.3321% | pass |
| material/component-Q max change | 2% | 4.6970% | 2.2751% | **fail** |
| fixed-probe tangential complex-E NRMSE | 2% | 3.3382% | 1.6796% | pass |
| conservative Q-volume L2 NRMSE | 5% | 3.1909% | 1.5708% | pass |
| material-region complex-E max NRMSE | 5% | 19.7920% | 6.9513% | **fail** |

The errors trend downward, but trend is not a certificate.  Ea's worst
material-field error is `5.7960%`; Eb's is `6.9513%`.  Eb also sets the
component-Q failure at `2.2751%`.  The declared threshold cannot be relaxed
after seeing the result.  Neither z16 nor z32 is selected.

## What the local Au papers actually do

The local paper set is under `/home/seunghyun200/papers`.  These detector
papers do not optimize a fabricated gold layer by assigning optical `rho^3`
and thermal/electrical `rho` to the same gray voxel.

1. `s41467-022-32309-w.pdf` and its SI use explicit 50-nm-thick Z-shaped Au
   metamaterial resonators.  Their finite geometric lengths, widths, periods,
   gap, handedness, rotation, and left/right area are selected by geometry
   sweeps/global optimization.  Optical absorption is evaluated with
   Lumerical FDTD and used as a heat source in Lumerical HEAT.  Reverse/bipolar
   response is produced by arranging explicit rotated resonators and active
   areas, not by interpreting intermediate density as gold.  DOI:
   [10.1038/s41467-022-32309-w](https://doi.org/10.1038/s41467-022-32309-w).
2. `s41467-024-51599-w.pdf` and its SI use explicit inverse-T, T, and propeller
   Au resonators.  Resonator length, orientation, unit cell, and channel
   geometry are swept.  SI Note 2 explicitly states that rotating a resonator
   is not equivalent to rotating polarization because the source-drain
   direction defines the measured-current projection.  The MIR example near
   4.75 um is again a binary resonator geometry.  DOI:
   [10.1038/s41467-024-51599-w](https://doi.org/10.1038/s41467-024-51599-w).
3. `Adv Funct Materials - 2026 - Blevins - Large Transverse Thermoelectric
   Effect in Weyl Semimetal TaIrTe4 Engineered for.pdf` uses
   `J_loc=-sigma S grad(T)` and a Shockley-Ramo weighting field to obtain the
   collected current.  It shows that electrode/crystal geometry controls
   magnitude and sign; the rectangular transverse example maximizes an edge
   current ratio near a 45-degree a-axis/electrode angle.  It also shows that
   real Au electrodes can enhance LWIR fields plasmonically, so omitting them
   from Maxwell while using them for readout is unsafe.  DOI:
   [10.1002/adfm.75986](https://doi.org/10.1002/adfm.75986).

The relevant topology-optimization literature does permit continuous
relaxations, but treats them as numerical design carriers and adds an explicit
material interpolation, damping/binarization strategy, filter/projection, and
exact-binary reevaluation.  It does not justify the historical O3/TE1 split.
Relevant examples already cited in the forensic audit are Christiansen et al.
([10.1016/j.cma.2018.08.034](https://doi.org/10.1016/j.cma.2018.08.034)),
Zeng and Xu
([10.1021/acsphotonics.1c00260](https://doi.org/10.1021/acsphotonics.1c00260)),
Hassan and Cala Lesina
([10.1364/OE.458080](https://doi.org/10.1364/OE.458080)), and Gedeon et al.
([10.1109/TAP.2024.3517156](https://doi.org/10.1109/TAP.2024.3517156)).

## Density-law code findings

Three distinct code paths must not be conflated:

1. The historical FDTDX production relaxation used optical O3 and
   thermal/electrical TE1.  In the manual dispersive-array construction, the
   Au pole dynamics were held while the coupling strength was scaled by the
   optical fraction.  The same numerical `rho` therefore represented
   different geometries in different PDEs.
2. The patched fork's generic continuous `Device` path is also not approved
   for Au.  It forms a linear static-permittivity mixture and independently
   linearly interpolates stored dispersive `c1/c2/c3/c4` coefficients.  That is
   not automatically one literature-backed causal `epsilon(omega,rho)` and
   can change pole dynamics at intermediate density.
3. The completed mesh campaign bypasses both relaxations.  It rejects float
   masks, accepts only exact integer/bool 0/1 occupancy, applies piecewise-
   constant replication, and reads back only air or ordinary-Au finite-dt
   coefficient endpoints.  It therefore answers the mesh question without
   asserting a gray material law.

Removing `rho^3` is necessary but not sufficient.  A future density route
would need one canonical filtered/projected occupancy, a published dispersive
interpolation with its numerical damping assumptions declared, AD-FD tests of
the complete material-to-Yee map, every robust projection constrained, and
independent exact-binary ordinary-Au promotion.  Given the local detector
papers and FDTDX cost, explicit binary shape/level-set or a lower-dimensional
binary geometry search should be preferred if the independent Maxwell solver
can provide validated gradients.

## Coupled-model blockers found in the code

The current thermal/electrical operators are internally useful prototypes,
not the target device.

- Thermal center pitch is 100 nm.  SiO2 is represented by 85/100/100-nm
  cells, the 100-nm TaIrTe4 by ten 10-nm cells, and 50-nm Au by 10/10/30-nm
  cells.  Lateral faces at +/-32 um and the substrate bottom at -20 um are
  fixed at ambient; the top uses an assumed `10 W m^-2 K^-1` convection.
  No x/y/z, domain-size, boundary, or interface-conductance convergence
  certificate exists.
- The electrical model has 25,600 TaIrTe4 nodes plus 6,400 Au nodes.  It fixes
  ideal terminals over the complete left/right flake edges and solves 31,680
  free unknowns.  It assumes solver x=b, y=a with no rotation, a centered
  floating Au sheet in direct electrical contact, and no actual electrode
  outline.
- Even at nominal void, the Au sheet retains `1e-8` of bulk Au conductivity,
  about `0.412 S/m`, and the vertical contact retains `1e-10` of its nominal
  conductance, or `1 S/m^2`.  No zero-floor sensitivity exists.
- The currently stored physical-device contract has every geometry/contact/
  beam confirmation false.  Wrong contacts, axis rotation, or Au electrical
  role can change the Shockley-Ramo weighting field and current sign; mesh
  refinement cannot repair the wrong boundary-value problem.

## Required next actions

1. Preserve this negative FDTDX result.  Do not run z64 and do not restart the
   historical optimizer.
2. Treat FDTDX z32 as too expensive for optimization and still unconverged.
   Any further FDTDX optical work must first propose a balanced, cheaper
   spatial discretization or a mathematically justified observable-based
   convergence strategy; it may not simply waive the failed fields.
3. Obtain the actual flake outline/thickness, in-plane a-axis angle, electrode
   and pad polygons, signed output contact, patterned-Au electrical role,
   SiO2/Si stack, beam parameters, and accepted Au-TaIrTe4 contact ranges.
4. Once an independently validated Maxwell Q field exists, freeze it and
   build a thermal x/y/z/domain/contact ladder.  Then build the actual-geometry
   electrical pitch/contact/void-floor ladder.
5. Only after Maxwell, thermal, electrical, and complete coupled AD-FD gates
   pass may a binary Au inverse-design route be considered.
