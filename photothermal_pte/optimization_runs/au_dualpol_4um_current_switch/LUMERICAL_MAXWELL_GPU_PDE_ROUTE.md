# Lumerical density-topology Maxwell + custom GPU-PDE route

Status: `RTX_SOURCE_GATE_PASSED_EXACT_AU_BLOCKED_Q_FLUX_AND_B200`

## Selected architecture

- Maxwell forward/adjoint and native-Yee absorption: Ansys Lumerical FDTD on
  NVIDIA B200;
- steady heat equation: repository custom CUDA finite-volume solver;
- weighting potential, PTE current, and electrical adjoint: repository custom
  CUDA solver.

No Lumerical HEAT or CHARGE license is assumed. FDTDX/JAX is not an allowed
production Maxwell substitute.

## The design variable

This route uses density topology, not shape/level-set optimization:

```text
latent rho
  -> 500-nm spatial filter
  -> tanh projection with beta continuation
  -> projected topology occupancy rho_bar in [0,1]
  -> documented optical, thermal, and electrical constitutive maps
  -> final thresholded 0/1 mask
  -> independent ordinary dispersive-Au Lumerical reevaluation
```

`rho_bar` is not an electron or hole density and is not claimed to be a
fabricated gray Au alloy. It is the differentiable relaxation of the binary
topology problem. The canonical state is 81x81 nodal values over the exact
80x80 100-nm physical cells. Lumerical consumes those nodes directly;
thermal/electrical consume the committed four-node cell average and return
cotangents through its exact transpose. The nodal values, coordinates, axes,
and optical law share one SHA-256. No subsystem may invent another rho field.
The constitutive formulas may differ because permittivity, thermal
conductivity, and electrical conductivity are different physical quantities.

## Optical material law: no rho cubed

At the 4-um optimization frequency, use the nonlinear metal/dielectric
interpolation

```text
n(rho_bar) = n_bg + rho_bar (n_Au - n_bg)
k(rho_bar) = k_bg + rho_bar (k_Au - k_bg)
epsilon(rho_bar) = [n(rho_bar) + i k(rho_bar)]^2
```

with the passive `n+i k` convention. The frozen Ordal endpoint is
`n_Au+i k_Au = 2.2+28.9i`, hence
`epsilon_Au = -830.37+127.16i` at 4 um. The implementation and analytic
complex derivative are in `au_density_relaxation.py`.

This is the physically motivated nonlinear interpolation proposed for
metallic topology optimization by Christiansen et al.
([DOI 10.1016/j.cma.2018.08.034](https://doi.org/10.1016/j.cma.2018.08.034))
and used in the plasmonic FDTD inverse-design framework of Zeng et al.
([DOI 10.1021/acsphotonics.1c00260](https://doi.org/10.1021/acsphotonics.1c00260)).
It avoids assigning an unsupported cubic law to Au oscillator strength.

`rho**3` is not used. Binarization is produced by the filter/projection
continuation and final discrete audit, not by pretending that a physical Au
property scales cubically.

## Lumerical carrier and its limitation

The first Lumerical-compatible implementation candidate is the complex
`importnk2` layer in `lumerical_4um_density.py`, generated from the equation
above on all 81x81 physical nodes. The same module supplies the PDE cell map,
transpose, and cross-solver state hash. The repository already
has a validated precedent for this implementation pattern in
`legacy_v261_optical_support`: a nonuniform complex density was mapped through
the actual Lumerical component-Yee mesh, a sparse material Jacobian was built,
and a full latent/filter/projection AD-FD gate passed for the earlier TaIrTe4
optimization.

That precedent proves the software pattern, not the Au physics. Au adds a
large negative real permittivity and a possible intermediate-density
zero-crossing resonance. Therefore the 4-um Au carrier remains blocked until
the Au-specific gates below pass.

`importnk2` supplies a spatial complex index for the single-frequency
relaxation; it is not being called an exact broadband Au material. A final
binary candidate must use an ordinary sampled-data dispersive Au material.
At rho=1, its 4-um objective-frequency field, absorption, and Q must first
agree with the ordinary dispersive-Au control. Across the finite source band,
the difference must be reported as approximation error rather than called
material parity, and the time-domain solve must pass decay/closure gates. If
that source-band error changes the objective or gradient beyond tolerance,
the single-frequency carrier is rejected; it is not repaired by claiming the
gray relaxation is physical Au. A custom Flexible Material Plugin is not a
B200 solution because Lumerical GPU does not support that plugin framework.

The endpoint/final control builder `lumerical_4um_exact_au.py` is retained for
this distinction. It samples Ordal Au, anisotropic TaIrTe4, and Kitamura SiO2
over a 3.2--4.8 um guard band around the 3.6--4.4 um source pulse, hashes the
complete physical 0/1 geometry, and maps it to non-overlapping ordinary-Au
prisms. Its sampled inputs still require actual Lumerical MCM fit readback.
`lumerical_4um_mesh_contract.py` defines the sequential source/time/z/x-y/PML
and domain-clearance controls for the exact endpoint/final cases. These files
do not replace the density carrier or its uniform-rho resonance/AD-FD gates.
Metal-interface mesh refinement is now a separate CV0/CV1/staircase axis.
Ansys warns that CV1 can create artifacts when the magnitude of a metal's
permittivity is much larger than the surrounding dielectric and recommends
comparison with the default CV0 treatment; that warning directly applies to
4-um Au (`epsilon` about `-830+127i`). See the official
[mesh-refinement guidance](https://optics.ansys.com/hc/en-us/articles/360034382614)
and [FDTD convergence guidance](https://optics.ansys.com/hc/en-us/articles/360034915833-Convergence-testing).

`lumerical_4um_forward.py` now assembles the common six-PML scalar-Gaussian
layout for `source_only`, exact `empty/full/simple_L`, or `import_density`.
The density runner accepts a uniform scalar only for endpoint/sweep controls;
an optimizer state enters as an explicit 81x81 NPY/NPZ nodal array. Its file
SHA and canonical coordinate/material-law state SHA are recorded, and the
exact array plus x/y coordinates are retained in the external raw NPZ. An
80x80 thermal/electrical cell field is deliberately rejected as optical input.
The actual case entry point is `25_run_lumerical_4um_exact_au_control.py`; the
sequential endpoint batch is `run_lumerical_4um_endpoint_b200.sh`. B200 is the
default and only promotable policy. An explicit development policy permits a
selected NVIDIA GPU for debugging while marking the result non-promotable. It
saves native component-Yee Q and a fixed air-side endpoint
field, and checks all fitted/finite-dt material readbacks plus Q/closed-flux
closure. A material case requires a passed, hash-matching source-only
waist/power record from the same accelerator policy, physical GPU UUID, and
solver version. These are provisional-device numerical gates, not a production
current prediction.

The earlier `np density` proposal remains rejected. It is a semiconductor
carrier-density attribute, not topology occupancy, and it is unnecessary for
this route.

## Maxwell derivative

Do not use the bundled LumOpt metal gradient without an independent gate. The
installed legacy and LumOpt2 topology implementations discard information
needed for lossy negative-real Au in parts of their material derivative path.

Use the repository's explicit discrete construction instead:

1. update `rho_bar -> n+i k -> importnk2`;
2. read the realized component-Yee permittivity on the frozen Lumerical mesh;
3. build `J_c = d epsilon_Yee,c / d rho_bar` by colored centered material-map
   finite differences without one Maxwell solve per pixel;
4. verify every JVP/VJP transpose identity;
5. contract `J_c^T` with the Lumerical forward/adjoint field product;
6. add direct-loss, thermal-material, and electrical-material derivatives;
7. pull the result through projection and filter transposes;
8. compare the complete latent directional derivative with independently
   rebuilt central finite differences for both polarizations and several
   steps/directions.

No empirical gradient scaling is allowed.

The reusable Au-specific implementation now lives in
`lumerical_4um_yee_jacobian.py`. It is deliberately nonperiodic and therefore
does not reuse the fencepost ownership assumptions from the older TaIrTe4
optimizer. Its sparse complex operator and endpoint handling pass synthetic
local-map FD and transpose tests. This is code validation only: it remains
blocked from optimizer use until it is built and checked against a completed,
hash-identical nonuniform `import_density` Lumerical FSP on the selected mesh.

## Required Au gates on the B200

1. Empty layer, uniform `rho_bar=0`, uniform `rho_bar=1`, and ordinary
   sampled-data Au controls must pass material readback, time stationarity,
   native-Yee Q, and six-face flux closure.
2. Imported `rho_bar=1` and ordinary dispersive Au must pass 4-um
   field/absorption/Q parity. Their finite-source-band constitutive difference
   must be quantified separately; it must not be mislabeled broadband parity.
3. Uniform `rho_bar` from 0 to 1 must be swept to detect artificial field/Q
   peaks and optimizer-favored gray resonances. Passivity of the algebraic
   material law alone is insufficient.
4. The nonuniform density-to-component-Yee map must pass multi-direction
   centered FD and transpose tests on the exact frozen mesh.
5. Optical and complete Maxwell/thermal/electrical latent AD-FD must pass for
   `Ea` and `Eb` before LD_MMA is enabled.
6. Full x/y/z/PML mesh convergence, source recalibration, and time/Q closure
   must pass on the same route. CV0/CV1/staircase is an explicit
   metal-interface convergence axis, not an implementation preference.
7. The final 500-nm solid/void mask must be independently rebuilt with
   ordinary dispersive Au and reevaluated for `Ia>0`, `Ib<0`.

The current host is not a B200 and therefore cannot issue any B200 Maxwell
certificate. It can run explicitly labeled RTX development diagnostics.

## Thermal and electrical maps

The historical O3/TE1 defect was that optical used `rho**3` while thermal and
electrical used `rho`. The correction is not to force every property to share
one arbitrary exponent. The correction is:

- share exactly one canonical nodal `rho_bar`, coordinates, and hash;
- derive solver grids only through tested forward/transpose maps;
- give each physical coefficient an explicit endpoint-correct law;
- differentiate every law through the same `rho_bar`;
- pass fixed-Q and combined AD-FD;
- verify the final exact-binary endpoint independently.

The present shared-linear thermal/electrical maps remain provisional until
their mixture/bound and void-floor sensitivity studies are complete. They are
not promoted merely because the optical rho-cubed law was removed.

## Exact endpoint/final GPU runner

The B200 launcher refuses a non-B200 device. A separate development launcher
accepts an explicitly selected NVIDIA GPU but can never issue a B200
certificate. The current session host exposes RTX 6000 Ada GPUs. The current local
installation is Lumerical 2026 R1.2. Nothing in the rejected `np density`
experiment justifies an R1.3 upgrade. Exact version compatibility must be
decided only by launching the ordinary exact-Au control on the actual B200 and
recording the engine log.

On 2026-08-24, solver `8.35.4413` passed Ea and Eb all-air source gates on RTX
GPU 5 after calibrating the source-object waist to `3.956143303046143 um`; the
realized effective waists were about `4.00077 um`. An ordinary dispersive-Au
full/Ea baseline run passed material fit, finite-dt material, native-Q,
`pabs_adv`, mesh-readback, decay, and GPU-log gates but failed Q versus
six-face flux closure by 30.43%. A PML-safe closed surface did not change that
error. A linked 5-nm stack-z / 50-nm bulk-z source run passed. Its exact-full
retry 2 then completed on the realized `183 x 183 x 212` grid and passed every
gate except Q/flux, which remained 29.239%. Thus stack/bulk z refinement alone
does not explain the discrepancy. The dynamic preflight still requires all 9
tasks; fewer CPU threads do not reduce this GPU checkout. The next diagnostic
is the exact-empty case on the identical mesh/source, followed by empty-to-full
incremental Q/flux comparison. That empty case subsequently passed with
`0.01636%` closure while full remained at `29.239%`. CV0, CV1, staircase,
2.5-nm z, and a stricter 2-ps/1e-9 decay run all reproduced the full-Au error.
The root cause was instead the allowed Au MCM complexity: max 20 selected a
failed overfit branch, whereas MCM4/6/8/12/16 formed a common field/Q plateau.
MCM6 passed closure at `0.08935%` and is now the default. See
`AU_MCM_FIT_FINDINGS.md`. The imported rho=1 carrier separately passes closure
but still has 1.849% complex-field endpoint error versus exact MCM6, so that
parity gate remains open.

The concrete forward entry point is
`25_run_lumerical_4um_exact_au_control.py`. It has an audit-only path that
does not open Lumerical and a Maxwell path that calls the selected accelerator
preflight again inside Python. For each numerical contract and polarization,
run the all-air
source control first:

```bash
LUMERICAL_B200_GPU_INDEX=<physical-index> \
  ./run_lumerical_b200.sh \
  ./25_run_lumerical_4um_exact_au_control.py \
  --case source_only --polarization Ea \
  --output-dir /path/outside/git/source_only_Ea
```

Then pass that result JSON to each material control with exactly the same
mesh/source arguments:

```bash
LUMERICAL_B200_GPU_INDEX=<physical-index> \
  ./run_lumerical_b200.sh \
  ./25_run_lumerical_4um_exact_au_control.py \
  --case simple_L --polarization Ea \
  --source-calibration-json /path/outside/git/source_only_Ea/<result>.json \
  --output-dir /path/outside/git/simple_L_Ea
```

The material run refuses a source-contract hash, accelerator policy, physical
GPU UUID, or solver-version mismatch. It also fails
closed unless all 81-point fitted/finite-dt Au, TaIrTe4, and SiO2 readbacks,
actual requested x/y/thin-stack/Si-bulk/air/PML mesh limits, native-Yee Q,
six-face flux, auto-shutoff, and engine-log GPU checks pass. Raw fields and Q
are never rescaled; 285 uW appears only as a derived scalar report. A
disjoint coordinate-based material partition is reported for convergence,
while raw component-specific epsilon arrays are retained so conformal
interface cells are not misrepresented as pure bulk material.

## Fail-closed sequence before optimization

1. Confirm the experimental flake outline/thickness, contacts, crystal axes,
   stack, beam, Au role, and interface parameters.
2. On the actual B200, pass source-only and fixed empty/full/simple exact-Au
   controls, time closure, native-Yee Q, and six-face energy balance.
3. Compare imported `rho_bar=0/1` against the matching empty/ordinary-Au
   endpoints, quantify source-band error, and reject artificial uniform-gray
   field/Q resonances.
4. Perform full x/y/z/PML mesh convergence for both polarizations. The z
   candidates must refine the thin stack and linked Si-bulk/air/PML z limit
   together; AD-FD alone is not mesh convergence.
5. Build and validate the nonuniform density-to-component-Yee Jacobian and
   discrete adjoint, then connect the same canonical density state to custom
   thermal/electrical maps and pass mesh, contact/floor, current-sign, and
   combined directional-FD gates.
6. Only then start LD_MMA filter/projection continuation, optimize
   `max min(+Ia,-Ib)`, and independently rebuild and reevaluate the final
   500-nm binary geometry with ordinary dispersive Au for both polarizations.
