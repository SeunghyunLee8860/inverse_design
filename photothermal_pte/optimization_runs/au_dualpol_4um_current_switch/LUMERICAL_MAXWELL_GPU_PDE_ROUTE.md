# Exact-Au Lumerical Maxwell + custom GPU-PDE inverse-design route

Status: `BLOCKED_PENDING_4UM_EXACT_AU_GEOMETRY_ESTIMATOR_AND_B200`

## Solver architecture

The production architecture remains the split already used by the earlier
TaIrTe4-flake optimization:

- optical fields and native-Yee absorbed power: Ansys Lumerical FDTD, with
  production execution required on an NVIDIA B200;
- steady thermal equation: the repository's custom CUDA finite-volume sparse
  solver;
- weighting potential, Shockley-Ramo PTE current, and electrical adjoint:
  the repository's custom CUDA finite-element sparse solver.

No Lumerical HEAT or CHARGE license is assumed or required. "Lumerical-only"
in this project means no FDTDX/JAX Maxwell surrogate; it does not replace the
custom thermal or electrical solvers.

## Exact material rule

Every physical objective evaluation must follow one path:

```text
continuous shape/level-set parameters p
  -> geometry construction and 500-nm solid/void DFM enforcement
  -> exact binary Au mask m in {0,1}
  -> ordinary exact dispersive-Au geometry in Lumerical FDTD
  -> native-Yee absorbed power Q
  -> the identical m and geometry hash in custom thermal/electrical solves
  -> Ia and Ib
```

Continuous optimizer parameters are allowed to move an exact Au/void
boundary. A continuous Au material fraction is not allowed in any physical
solver. In particular, the following are prohibited:

- gray `importnk` or diluted complex permittivity;
- `np density` reinterpreted as Au occupancy;
- optical `rho**3` with thermal/electrical `rho`;
- independently altered material occupancy or geometry in the three solvers.

The Au material object used in Lumerical must remain a frozen dispersive
material. Only its exact geometry changes. Each evaluation must record the
SHA-256 of the canonical realized geometry; all three solvers must report the
same hash. The geometry hash includes mask values and shape, physical x/y cell
edges, Au z bounds, and the x=b/y=a mapping. A mask-only hash is merely a
payload checksum and is not sufficient physical identity.

`lumerical_4um_exact_au.py` constructs sampled Ordal-Au dispersion over a
3.2--4.8 um guard band around the 3.6--4.4 um numerical source pulse. It also
constructs sampled anisotropic TaIrTe4 and Kitamura-SiO2 inputs and maps the
exact mask to deterministic non-overlapping ordinary-Au prisms. These are
audited inputs, not yet a material-fit certificate: the actual Lumerical
multi-coefficient fits must be read back across the band on the run host.

This is a physical-geometry rule, not a demand that unlike numerical meshes
store identical cell arrays. Lumerical conformal-interface tensors and
thermal/electrical cut-cell fractions may represent the boundary on their own
meshes. They are numerical discretizations of the same hash-identified binary
geometry, not optimizer-controlled gray Au. Their convergence must be tested.

`NP_DENSITY_ROUTE_REJECTED.md` records and retracts the invalid carrier route.
There is no R1.3 requirement arising from `np density`.

## What inverse design means under this rule

Exact material does not mean that inverse design is impossible. It changes
the optimization variable and derivative strategy:

1. **Preferred: shape or level-set optimization.** A finite set of boundary
   parameters or a level-set moves the interface while each Lumerical solve
   still contains only ordinary Au and void. Use a Lumerical forward/adjoint
   shape derivative only after central AD-FD passes at 4 um on the selected
   mesh.
2. **Fallback: exact-geometry finite differences or SPSA.** Central finite
   differences are viable for a compact shape basis. SPSA needs two
   perturbed physical evaluations per random direction, independent of the
   number of parameters, but is noisy and must be averaged and checked
   against independent coordinate/directional finite differences. Both sides
   of every perturbation use exact dispersive Au.
3. **Discrete candidate search.** Pixel flips, BESO-like updates, or a
   trust-region candidate set may also preserve exact Au, but are expensive
   and need full candidate reevaluation. They are not an excuse to reuse a
   gray-material gradient.

Hard-thresholding thousands of independent density pixels and feeding that
mask to LD_MMA would have zero/undefined derivatives almost everywhere. The
existing 80 x 80 gray-density LD_MMA path therefore cannot simply be relabeled
as exact-Au optimization. A new parameterization and validated estimator are
required.

## Existing evidence and the next AD-FD gate

Lumerical can forward-simulate ordinary dispersive Au. However, earlier
repository controls at another wavelength found that several moving,
conformal high-contrast Au shape derivatives failed finite differences,
including wrong-sign errors. Those failures do not prove that the 4-um case
is impossible, but they prohibit assuming that a generic LumOpt shape
gradient is correct.

The next optical experiment is therefore deliberately small:

1. build the confirmed 4-um TaIrTe4 stack with an ordinary dispersive-Au
   shape described by a compact smooth parameterization;
2. use identical sources, time windows, conformal settings, and mesh for the
   forward, adjoint, and independently rebuilt `p+h`/`p-h` projects;
3. compare the Lumerical shape derivative with central finite differences for
   multiple parameters, directions, and step sizes;
4. separately check that each realized geometry is exact binary and that its
   absorbed power/time closure is stable;
5. accept the adjoint only if the error converges with step size and retains
   the correct sign across representative geometries.

If this gate fails, switch to the exact-geometry SPSA/finite-difference route;
do not return to FDTDX, gray Au, or `np density`.

## Thermal/electrical consistency

The old O3/TE1 defect came from solving different effective devices. The new
rule removes that ambiguity: thermal and electrical solvers receive the same
realized 0/1 geometry as Maxwell, then apply their endpoint material
properties to that geometry. Constitutive quantities have different units,
but material occupancy does not:

```text
Maxwell:     m=1 -> dispersive Au; m=0 -> background/underlying stack
thermal:     m=1 -> Au thermal coefficients and Au interfaces
electrical:  m=1 -> Au electrical coefficients and contacts/interfaces
```

Any numerical void-floor needed to make an electrical system nonsingular is
a solver regularization, not an Au fraction. It must receive an independent
sensitivity study and must not alter the geometry hash.

## B200 launch and version status

Maxwell entry points are launched through:

```bash
LUMERICAL_B200_GPU_INDEX=<physical-index> \
  ./run_lumerical_b200.sh <python-script> [arguments]
```

The launcher refuses a non-B200 device. The current session host exposes RTX
6000 Ada GPUs, so it cannot issue a B200 certificate. The current local
installation is Lumerical 2026 R1.2. Nothing in the rejected `np density`
experiment justifies an R1.3 upgrade. Exact version compatibility must be
decided only by launching the ordinary exact-Au control on the actual B200 and
recording the engine log.

## Fail-closed sequence before optimization

1. Confirm the experimental flake outline/thickness, contacts, crystal axes,
   stack, beam, Au role, and interface parameters.
2. On the actual B200, pass fixed empty/full/simple exact-Au Lumerical forward
   controls, time closure, native-Yee Q, and six-face energy balance.
3. Perform full x/y/z/PML mesh convergence for both polarizations using exact
   dispersive Au; AD-FD alone is not mesh convergence.
4. Implement and validate the compact 4-um exact-Au shape/level-set estimator.
   If adjoint AD-FD fails, validate an exact-geometry FD/SPSA estimator.
5. Connect the same binary geometry to custom CUDA thermal/electrical solvers
   and pass endpoint, mesh, floor-sensitivity, current-sign, and combined
   directional-FD tests.
6. Only then start an optimizer compatible with the certified estimator,
   optimize `max min(+Ia,-Ib)`, and independently reevaluate the final 500-nm
   DFM geometry for both polarizations.
