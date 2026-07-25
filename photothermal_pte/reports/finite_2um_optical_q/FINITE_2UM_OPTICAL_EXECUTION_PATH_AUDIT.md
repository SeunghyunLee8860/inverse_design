# Finite 2 um optical Q execution-path audit

Baseline optical commit: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`

## Scope

This branch is independent of HEAT Draft PR #2. It creates a new optical
execution path for one finite 2 um by 2 um by 100 nm TaIrTe4 flake. It does not
modify PR #2, does not crop/tile/rescale the validated periodic Q artifact, and
does not run HEAT, PTE current, adjoint, gradient, or optimization.

## Existing production path

The periodic production chain is:

1. `eqc_lib.bootstrap_env()` pins the 3--6 um source, 4 um analysis point,
   2.7--13.2 um / 600-sample TaIrTe4 material table, auto non-uniform mesh,
   conformal variant 1, accuracy 5, 5 nm flake dz, v261, and GPU execution.
2. `bundle/tairte4_volume_model.py::build_case()` constructs a 6 um by 6 um
   cell with x/y periodic boundaries, a periodic plane source, a laterally
   extended TaIrTe4 slab, and an index-defined design region.
3. `eqc_lib.assert_production_contract()` reads the realized FSP and rejects a
   source/material/mesh/solver mismatch before a solve.
4. `24_run_production_optical_regression.py` enables the installed
   `pabs_adv` periodic correction and compares Q with a local flux box.

The following pieces are reusable without changing periodic behavior:

- the sampled anisotropic TaIrTe4 material builder (`eps_flake()` receives nm);
- the source/material/mesh constants and v261/GPU resource checks;
- native Yee-component Q evaluation;
- one-point 4 um monitors and SHA-256 artifact manifests.

The following pieces are periodic-only and must not be reused for the finite
solve:

- x/y periodic boundaries;
- the laterally infinite TaIrTe4 slab;
- the periodic plane source;
- periodic seam wrapping and periodic `pabs_adv` correction;
- `1-R-T` as the total absorption definition;
- arbitrary design index 4 instead of the bottom-SiO2 optical model.

The finite path is therefore implemented only in
`27_validate_finite_2um_optical_q.py`; the periodic builder is left unchanged.

## Finite source decision

The primary candidate is a normally incident TFSF source. The official v261
object documentation states that TFSF supports non-periodic scatterers on
multi-layer substrates. Its incident field in this case is the field of the
unperturbed multi-layer stack:

- https://optics.ansys.com/hc/en-us/articles/360034902093-Total-Field-Scattered-Field-TFSF-source-Simulation-object

The official best-practices page requires:

- the scatterer to remain completely inside the TFSF box;
- the injection axis to be normal to the substrate;
- the reference corner to cross the unperturbed layered background rather than
  the finite feature;
- the source not to cross a PML boundary;
- an empty-stack control before interpreting a layered-background result.

See:

- https://optics.ansys.com/hc/en-us/articles/360034382934-Tips-and-best-practices-when-using-the-FDTD-TFSF-source

The installed v261 API was probed directly and exposes `addtfsf` with x/y/z
min/max, injection-axis, direction, polarization, broadband wavelength, and
time-domain properties. TFSF is not accepted on documentation alone: the
empty-stack leakage and source-boundary controls are mandatory gates.

If those controls fail, the workflow stops with
`BLOCKED_TFSF_LAYERED_BACKGROUND_UNVERIFIED`. A Gaussian-beam fallback may be
implemented only as an explicitly labelled beam result with a measured waist,
focus, total power, central intensity, footprint uniformity, and waist
convergence. It must not be called a plane wave.

## Power and normalization contract

The finite structure uses a six-face power box wholly inside the TFSF volume
and outside the TaIrTe4/design geometry. Net inward flux is compared with the
TaIrTe4 volume integral:

`abs(P_Q - P_six_face) / abs(P_six_face) < 0.5%`.

No `1-R-T` acceptance metric is used. The physical finite-object quantity is
the absorption cross section:

`sigma_abs = P_abs / I_inc`.

Official TFSF normalization guidance explicitly recommends source intensity,
not the arbitrary finite source-box power:

- https://optics.ansys.com/hc/en-us/articles/360034902133-Understanding-source-normalization-in-the-TFSF-source

The final Q artifact is normalized to measured `I_inc = 1 W/m2`. No empirical
flux-matching gain, clipping, smoothing, rescaling, periodic cropping, or
periodic tiling is permitted.

## Mandatory pre-solve and post-solve checks

- all six FDTD boundaries are PML;
- one centered 2 um by 2 um TaIrTe4 flake, thickness 100 nm;
- bottom SiO2 thickness 285 nm and Si substrate;
- fixed design solid, if enabled, uses the same SiO2 optical model as the
  bottom oxide and is not periodically repeated;
- broadband source 3--6 um, evaluation monitors at one 4 um point;
- material table 2.7--13.2 um with 600 samples;
- auto non-uniform mesh, conformal variant 1, accuracy 5;
- no `global_uniform_mesh`;
- TaIrTe4 dz equals the case-requested 10, 5, or 2.5 nm value;
- Pabs top/bottom zero padding equals 50 nm;
- v261 GPU resource and realized dt are recorded;
- source box, six-face box, and geometry have strict non-intersection margins;
- Qx, Qy, Qz are retained without clipping;
- artifact coordinates contain the exact flake bounds so HEAT needs no crop.

## Convergence matrix

The planned matrix includes lateral domains 8/12/16 um, at least two PML-layer
settings, and flake dz 10/5/2.5 nm. The four required physics cases are finite
flat x/y/45-degree polarization and finite fixed-design x polarization.

No convergence or validated-Q claim is made until fresh v261 simulations
satisfy source injection, six-face closure, domain, PML, mesh, and spatial-Q
criteria.
