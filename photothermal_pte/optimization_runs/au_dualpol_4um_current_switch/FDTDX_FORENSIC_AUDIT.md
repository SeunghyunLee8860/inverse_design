# FDTDX forensic audit: 4 um Au / TaIrTe4 dual-polarization PTE design

Status: **BLOCKED_FDTDX_FORENSIC_AUDIT**

Audit date: 2026-08-24

This document is the entry point for the historical FDTDX campaign. It is a
read-only forensic audit: no Maxwell, thermal, electrical, adjoint, or
optimization solve was run. The concurrently developed Lumerical route is out
of scope and no Lumerical source or artifact is modified here.

Run the machine-readable audit from the repository root:

```bash
python3 -m photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_forensic_audit
```

## Executive decision

Do **not** resume the historical FDTDX optimizer and do **not** promote its
geometry. The committed evidence establishes all of the following:

1. The historical nominal design produces the same current sign for `Ea` and
   `Eb` at every completed full-domain-z refinement factor.
2. Every exact-binary candidate that passes the discrete 500 nm opening audit
   also produces the same current sign.
3. The z-mesh study failed its own tolerance and did not select a mesh.
4. That study did not refine x or y; `dx=dy=100 nm` at every level.
5. No thermal-mesh, electrical-mesh, or combined-gradient certificate exists
   on a selected optical mesh.
6. All ten required physical-device confirmations remain false.
7. The Au relaxation, first as optical `rho**3` versus thermal/electrical
   `rho`, and later as shared `rho`, is not a physical gray-Au constitutive
   model supported by the inspected papers.

Small conservation residuals in the historical runs prove consistency with
the implemented discrete equations. They do not prove convergence to the
continuous metal-interface problem.

## Requested observable and decisive endpoint results

The design objective implemented by the campaign is

```text
maximize min(I_Ea, -I_Eb)
```

under the convention `x = crystal b`, `y = crystal a`, `psi=0` on the complete
`x_min` edge and `psi=1` on the complete `x_max` edge. This requests
`I_Ea > 0` and `I_Eb < 0`, which is stronger than merely requiring opposite
signs.

The later shared-linear full-domain-z reevaluation gives:

| z factor | dz TaIrTe4 | dz Au | I(Ea) | I(Eb) | opposite sign? |
|---:|---:|---:|---:|---:|:---:|
| 1 | 20 nm | 25 nm | -6.36185 nA | -8.30711 nA | no |
| 2 | 10 nm | 12.5 nm | -6.86533 nA | -7.92006 nA | no |
| 4 | 5 nm | 6.25 nm | -7.09992 nA | -6.74433 nA | no |

The old evaluation 112 did report `I_Ea=+5.41918 nA` and
`I_Eb=-5.40366 nA`, but it cannot be treated as a device design: 53.20% of its
cells were between density 0.01 and 0.99, its binarization metric was 0.3062,
and 189 cells failed the exact 500 nm audit. It also belongs to the superseded
gray-law/coarse-grid campaign.

The later robust run produced six thresholded exact-binary candidates with
zero discrete 500 nm bad cells. All six had `I_Ea > 0` and `I_Eb > 0`; none
passed the opposite-sign gate. This failure is decisive for those candidates
because an interpolation choice cannot alter the constitutive law at exact
`rho=0` and `rho=1` endpoints.

## Mesh audit

### Optical mesh

The baseline FDTDX grid has physical bounds `20 x 20 x 6 um` and
`186 x 186 x 40` cells. Its central lateral pitch is 100 nm. The baseline
vertical stack uses approximately:

| Region | Baseline cells | Baseline dz |
|---|---:|---:|
| resolved Si above bottom PML | 5 | 203 nm |
| 285 nm SiO2 | 3 | 95 nm |
| 100 nm TaIrTe4 | 5 | 20 nm |
| 50 nm Au | 2 | 25 nm |
| near air | 4 | 50 nm |

`15_validate_4um_z_mesh_convergence.py` later refined the full z domain by
factors 1, 2, and 4, but retained the same `186 x 186` lateral grid and
`central_dx=central_dy=100 nm`. Consequently, it is a z-only study and cannot
certify the topology's lateral metal boundaries.

No one of the twelve factor-2-to-factor-4 comparisons passed. The worst
relative changes over the three projections and two polarizations were:

| Quantity | worst 2x -> 4x change |
|---|---:|
| total Q | 3.314% |
| remapped Q spatial L2 NRMSE | 34.05% |
| TaIrTe4 temperature NRMSE | 3.634% |
| maximum temperature | 30.15% |
| terminal current | 37.66% |

The summary status is
`BLOCKED_SHARED_LINEAR_FULL_DOMAIN_Z_CONVERGENCE`, its
`selected_optical_z_contract` is null, and the production full-mesh
certificate path does not exist.

### Metal-interface resolution

At 4 um the committed Au readback is `n+ik = 2.2 + 28.9i`, corresponding to
`epsilon = -830.37 + 127.16i`. The amplitude skin depth estimate
`lambda/(2*pi*k)` is about 22 nm. A 100 nm lateral cell is therefore about 4.5
skin depths wide.

The project creates TaIrTe4 and Au with `fdtdx.UniformMaterialObject`. In the
inspected compatible FDTDX source, this assigns material coefficients to grid
slices without the subpixel-smoothing path available to multi-material static
objects. The project never requests `subpixel_smoothing`. Thus:

- the 100 nm pixel edge is a staircase boundary;
- a topology pixel is only one optical cell in x and y;
- a tangential Yee-field dual volume can cross a metal interface while the
  loss coefficient is assigned from one material slice;
- conservative remapping of Q preserves the discrete integral but cannot
  repair an already misassigned interface loss.

This is a high-contrast-interface accuracy blocker, even when Q and closed
surface flux agree on the same grid.

### PML, domain, source, and time

The caller gives `fdtdx.BoundaryConfig` only six PML thicknesses. It does not
set alpha, kappa, or sigma profiles. In the inspected upstream FDTDX tree at
commit `f26f84b70a8cceec9b889553955a868624736bf1f`, the CPML default
`alpha_start` is explicitly based on 1.55 um, not the 4 um carrier. Historical
records hash `fdtd/update.py` and, in one diagnostic, `dispersion.py`; they do
not pin the entire FDTDX tree or the PML implementation. Therefore the exact
historical PML implementation is not provenance-closed, and no PML-parameter,
PML-thickness, or boundary-distance convergence result exists.

The lateral PML is 1 um thick. The 16 um flake has only 1 um between each
flake edge and the lateral PML interface. The source begins at z=0.75 um and
the top PML interface begins at z=1.4 um, a 0.65 um separation. These distances
must be varied as convergence axes; source-only power calibration cannot by
itself certify near-field/PML isolation.

The later 40-period, Courant-factor-0.25 runs improved time/ADE stability and
the rectangular phasor-window workaround made flux accounting internally
consistent. They did not close any spatial convergence gate.

## Material and gray-density audit

### Endpoint material data

The committed material contract is a single-frequency readback at 4 um:

- Au: Ordal `n=2.2`, `k=28.9`;
- TaIrTe4 a: `epsilon=-30.7133+50.8481i`;
- TaIrTe4 b: `epsilon=15.9007+9.28919i`;
- TaIrTe4 c: copied exactly from b;
- Si and SiO2: implemented as lossless in the FDTDX solve.

The one-pole float32 ADE refit accurately reproduces its target at the single
carrier frequency. That is a discrete one-frequency fit check, not broadband
material validation. Copying `c=b` is a closure assumption, not an independent
out-of-plane measurement.

### Why O3/TE1 was invalid

The early implementation used an optical fraction proportional to `rho**3`
while thermal and electrical properties were proportional to `rho`. The same
optimization variable therefore represented three different geometries. A
combined adjoint can still differentiate those equations correctly, but it
does not describe one realizable structure.

### Why shared rho is only a diagnostic relaxation

Changing all three solvers to a shared linear fraction removed the O3/TE1
inconsistency. It did not turn intermediate density into physical Au. In the
optical ADE it linearly scales oscillator strength; in the thermal and
electrical models it linearly mixes bulk transport and contact conductance.
No inspected local detector paper defines such a common effective medium.

The relevant plasmonic topology-optimization literature also warns that naive
linear metal/dielectric interpolation can create nonphysical intermediate
plasmon resonances or amplification. Published alternatives include:

- nonlinear interpolation of optical constants with filter/projection and
  binary reevaluation;
- Drude/CCPR endpoint-parameter interpolation plus an artificial
  `rho*(1-rho)` damping/conductivity term that vanishes at both physical
  endpoints;
- shape or level-set optimization, which avoids interpreting a gray cell as a
  fabricated material.

Artificial damping is a numerical continuation device, not a new material.
Whatever relaxation is used for gradients, promotion requires ordinary
sampled-data dispersive Au on an exact-binary geometry.

### Literature inspected

The local papers under `/home/seunghyun200/papers` establish useful device
physics, but they do not justify `rho**3` or any gray-Au law:

1. Dai et al., *On-chip mid-infrared photothermoelectric detectors for
   full-Stokes detection*, Nature Communications 13, 4560 (2022),
   [doi:10.1038/s41467-022-32309-w](https://doi.org/10.1038/s41467-022-32309-w).
   It uses fixed 50 nm Au antenna/backplate geometry and Lumerical FDTD/HEAT;
   its optical heat source is the usual material loss density.
2. Koepfli et al., *Controlling photothermoelectric directional photocurrents
   in graphene with over 400 GHz bandwidth*, Nature Communications 15, 7734
   (2024),
   [doi:10.1038/s41467-024-51599-w](https://doi.org/10.1038/s41467-024-51599-w).
   It uses fixed T-shaped Au resonators and geometric parameter sweeps, CST FEM
   for optics, and COMSOL for electron flow.
3. Blevins et al., *Large Transverse Thermoelectric Effect in Weyl Semimetal
   TaIrTe4 Engineered for Photodetection*, Advanced Functional Materials
   (2026), [doi:10.1002/adfm.75986](https://doi.org/10.1002/adfm.75986).
   It makes the collection current explicitly geometry-sensitive through the
   Shockley-Ramo weighting field. The local folder contains the main paper but
   not its parameter-bearing supplementary PDF; assumptions attributed to
   Note S5/Table S2 cannot be independently reconstructed from the local paper
   set.

Broader inverse-design references inspected for the interpolation question:

1. Zeng and Xu, *Inverse Design of Plasmonic Structures with FDTD*, ACS
   Photonics 8, 1489-1497 (2021),
   [doi:10.1021/acsphotonics.1c00260](https://doi.org/10.1021/acsphotonics.1c00260).
   It identifies nonphysical amplification from poor interpolation and uses a
   nonlinear material interpolation with filter/projection.
2. Hassan and Cala Lesina, *Topology optimization of dispersive plasmonic
   nanostructures in the time-domain*, Optics Express 30, 19557-19572 (2022),
   [doi:10.1364/OE.458080](https://doi.org/10.1364/OE.458080).
   Its Drude formulation adds parabolic artificial conductivity
   `rho*(1-rho)*sigma_max` to suppress intermediate-density plasmonic
   convergence failures.
3. Gedeon et al., *Time-Domain Topology Optimization of Power Dissipation in
   Dispersive Dielectric and Plasmonic Nanostructures*, IEEE TAP (2024),
   [doi:10.1109/TAP.2024.3517156](https://doi.org/10.1109/TAP.2024.3517156).
   It uses CCPR material interpolation, explicit artificial damping for Au,
   filter/projection, and binary thresholding; it also documents persistent
   gray-surface difficulties for gold.
4. Christiansen et al., *A non-linear material interpolation for design of
   metallic nano-particles using topology optimization*, Computer Methods in
   Applied Mechanics and Engineering 343, 23-39 (2019),
   [doi:10.1016/j.cma.2018.08.034](https://doi.org/10.1016/j.cma.2018.08.034).

## Thermal audit

The current thermal operator is a useful differentiable prototype, not a
validated device model:

- domain: lateral boundaries at +/-32 um and substrate bottom at -20 um;
- central x/y pitch: 100 nm, coarsened outside the center;
- fixed-temperature lateral/bottom faces and top convection `10 W m^-2 K^-1`;
- 285 nm SiO2 represented by 85/100/100 nm cells;
- TaIrTe4 by 10 nm cells and Au by 10/10/30 nm cells;
- assumed diagonal conductivities and fixed interface conductances;
- gray Au conductivity and Au/Ta contact are linearly mixed with density.

The heat-map remap is conservative and the linear solve residual/energy
balance checks are good. Missing are independent refinement of x/y/z, outer
domain, bottom/lateral boundary placement, top convection, and every uncertain
thermal boundary conductance. A fixed Q field must first be used to isolate
thermal discretization error from optical error.

## Electrical and PTE audit

The TaIrTe4 weighting problem is a 160 x 160, 100 nm-pitch 2D network with a
floating 80 x 80 Au sheet. The implemented tensor mapping is
`sigma_x=sigma_b`, `sigma_y=sigma_a`, `S_x=S_b`, `S_y=S_a`. The discrete
current expression and its fixed-temperature adjoint are internally
consistent with the documented sign convention.

The physical model remains unconfirmed:

- ideal terminals cover the complete left and right flake edges;
- no actual electrode outline is represented;
- Au is assumed directly electrically connected to TaIrTe4 and floating;
- Au can therefore shunt and reshape the weighting potential, while its own
  thermopower is omitted;
- void Au and void contact retain finite numerical floors;
- no electrical pitch, contact-conductance, or void-floor convergence exists;
- only unrotated diagonal in-plane tensors are supported.

Because the Shockley-Ramo collection field is geometry-sensitive, wrong
terminal or Au-contact assumptions can change both magnitude and sign. Mesh
refinement cannot fix the wrong physical boundary-value problem.

## Provenance and reproducibility audit

The repository contains code, JSON summaries, CSV tables, and hashes, but not
the raw checkpoints/NPZ fields. The 18 full-z raw paths are absolute paths
under `/home/seunghyun/tairte4/raw/...`; none exists in this audit environment.
The default FDTDX source path is another absolute path,
`/home/seunghyun/.local/fdtdx_main_src`.

Import-location checking prevents an accidental site-package import, but the
repository does not pin a complete FDTDX git commit/tree. Two individual
source hashes are insufficient to reproduce all grid placement, boundary,
detector, and update behavior. A fresh campaign must record the complete git
commit, dirty-tree state, environment lock, accelerator information, every
input hash, and portable raw-artifact manifest.

## Fresh-start gate order

No optimizer should run until Gates 0-6 pass.

### Gate 0 - freeze the measured device

Obtain the flake outline/thickness, crystal-axis rotation, actual terminal and
pad outlines, signed output convention, Au electrical role, SiO2/Si stack,
beam power/waist/center/incidence, and accepted thermal/electrical contact
ranges. Generate the weighting potential for the actual contacts before
judging a current sign.

### Gate 1 - freeze dependencies and endpoint materials

Pin the full solver tree and environment. Validate Au and all three TaIrTe4
tensor components over the finite source bandwidth. Do not silently set
`epsilon_c=epsilon_b`. Define exact-binary air/Au endpoints as the only
promotable materials.

### Gate 2 - exact-binary Maxwell reference problems

Before using the historical topology, validate empty, full-Au film, planar
multilayer, isolated strip/patch, and one DFM-scale corner. Compare reflection,
transmission, layer-resolved Q, and field phase against analytic transfer
matrix results where available and against the independent Lumerical work
when it is ready. This session must not modify that Lumerical work.

### Gate 3 - multidimensional optical convergence

For each fixed exact-binary reference and later for a frozen candidate, vary:

1. lateral interface pitch/subpixel treatment;
2. TaIrTe4, Au, oxide, Si, air, and PML z resolution;
3. PML thickness/profile and boundary distance;
4. source distance/aperture and monitor placement;
5. Courant factor, total time, and phasor window;
6. source calibration on every exact grid/time contract.

Use staged anisotropic studies rather than blindly refining all dimensions at
once. Require convergence of spatial Q, layer/component Q, complex field,
temperature, and signed terminal current, not total power alone.

### Gate 4 - thermal convergence with frozen Q

Refine x/y/z and outer domain independently, then sweep interface conductance
and external boundary conditions. Certify TaIrTe4 temperature gradients and
current-relevant thermal functionals, not only `Tmax`.

### Gate 5 - electrical/PTE convergence

Use the actual contact geometry. Refine the TaIrTe4 and Au sheets, sweep Au/Ta
contact and void floors to their zero limits, test Au isolated/touching/
terminal-connected scenarios as physically appropriate, and validate the
weighting-field/current sign with independent direct calculations.

### Gate 6 - gradients on the selected coupled discretization

Run multi-direction centered finite differences for both polarizations,
including optical field, direct optical-loss, thermal-property/contact, and
electrical-weighting contributions. A coarse historical AD-FD pass cannot
certify a new mesh/material law.

### Gate 7 - optimization relaxation

Prefer a binary shape/level-set route if gradients and DFM permit it. If a
density route is necessary, select one literature-backed dispersive
interpolation, document artificial damping as numerical-only, apply
filter/projection continuation, constrain every robust projection, and monitor
uniform-density resonances. Never interpret gray density as fabricated Au.

### Gate 8 - independent endpoint promotion

Threshold and repair to exact 500 nm solid/void geometry, then reevaluate
ordinary dispersive Au on converged Maxwell/thermal/electrical meshes for all
robust cases. Promotion requires the requested signed currents on the nominal
binary geometry and preserved opposite sign under fabrication and uncertain
interface scenarios.

## Classification of existing work

Already useful but diagnostic:

- objective sign convention and Shockley-Ramo discrete expression;
- source-power scaling machinery;
- conservative optical-to-thermal remap;
- thermal/electrical adjoint algebra and fixed-Q checks;
- 500 nm filter/projection and exact discrete opening audit;
- time/ADE stability and rectangular phasor-window diagnosis;
- the failed full-domain-z study, because it supplies quantitative evidence
  that the historical solution is not converged.

Previously corrected but not production-validated:

- inconsistent O3/TE1 fractions changed to shared linear density;
- robust projection list expanded to include nominal density;
- grayness constraints expanded to every robust projection;
- historical sign and adjoint-source phase fixes;
- float32 ADE long-time refit and time contract.

These fixes prevent known implementation mistakes. They do not close the
physical-device, material, mesh, binary-endpoint, or selected-mesh gradient
gates described above.
