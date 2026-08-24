# Fresh FDTDX exact-binary convergence plan

Status: **EXACT_BINARY_MATERIAL_BRIDGE_UNIT_VALIDATED_NOT_PLACED**

This is the numerical contract for a fresh FDTDX investigation. It does not
resume the historical topology optimization, does not use the historical
gray checkpoint, and does not modify or execute the concurrent Lumerical
route.

The executable contract is `fdtdx_exact_binary_convergence.py`. It has no
FDTDX, JAX, NumPy, thermal, or optimizer dependency and can be audited before
GPU work:

```bash
python3 -m photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_exact_binary_convergence
```

## What changed from the historical mesh study

The historical code had one topology pixel per 100 nm Yee cell and varied
only z. The new contract makes the topology grid and the Maxwell grid separate:

```text
80 x 80 exact-binary geometry
        |
        +-- repeat 1x -> 100 nm Yee cells in the Au window
        +-- repeat 2x ->  50 nm Yee cells, identical physical edges
        +-- repeat 4x ->  25 nm Yee cells, identical physical edges
```

The repeat operation is piecewise constant. It creates no gray cells and does
not shift any air/Au boundary. The TaIrTe4 flake wings, air gap, lateral PML,
and full z stack have independent factors, so one mesh axis can be changed
while all others remain hash-identical.

The lateral grid is segmented as:

```text
PML | air gap | TaIrTe4 wing | 8 um Au design window | TaIrTe4 wing | gap | PML
```

At the new anchor mesh, the non-PML air gap is 100 nm rather than the
historical 200 nm, the full z factor is already 4, and the grid has
`196 x 196 x 160 = 6,146,560` Yee cells. Local refinement of the Au window to
25 nm gives `436 x 436 x 160 = 30,415,360` cells, instead of refining the full
20 um lateral domain to roughly 100 million cells. The outer region must still
be checked separately; local refinement is a cost-control mechanism, not a
waiver of flake-edge convergence.

## Exact-binary reference geometries

Six deterministic 80 x 80 masks are defined and SHA-256 identified:

1. empty design window;
2. full 8 um x 8 um design window;
3. centered 2 um square;
4. 4 um x 1 um x-directed bar;
5. 1 um x 4 um y-directed bar;
6. asymmetric 4 um L with 1 um arms.

They test different failure modes:

- empty: source, substrate, TaIrTe4, PML, and background closure;
- full: exact Au endpoint, thin-film/interface loss, and ADE stability;
- square: compact corners and four lateral metal interfaces;
- x/y bars: axis mapping and anisotropic polarization response;
- L: asymmetric phase, absorption, temperature, and current response.

These are verification structures, not candidate devices. Their purpose is to
identify numerical error before the historical or any newly optimized
geometry is evaluated.

`fdtdx_exact_binary_material.py` is the only allowed material application for
these references. It rejects float masks even when their values happen to be
0.0/1.0, repeats the integer mask without interpolation, and writes all three
Au ADE recurrence coefficients (`c1`, `c2`, and `c3`) to exact endpoints. An
air cell has epsilon-infinity 1 and zero Au ADE coefficients; an Au cell has
epsilon-infinity 1 and the complete locked ordinary-Au ADE coefficient tuple.
There is no `rho`, density exponent, or gray material in this path.

## Independent convergence axes

Each ladder has three levels and changes one field of `MeshSpec` only.

| Axis | Levels | Meaning |
|---|---|---|
| full-domain z | 2, 4, 8 | all Si/SiO2/TaIrTe4/Au/air/z-PML segments |
| Au-window x/y | 1, 2, 4 | 100, 50, 25 nm at patterned Au boundaries |
| outer x/y | 1, 2, 4 | TaIrTe4 wings and non-PML air gap |
| lateral PML grid | 1, 2, 4 | fixed physical PML with more cells |
| flake-to-PML gap | 1, 2, 4 um | boundary distance |
| PML thickness | 1, 1.5, 2 um | fixed inner boundary, outward expansion |
| PML alpha | 0.5, 1, 2 times | explicit 4 um CPML profile sweep |
| time | Courant 0.25/0.125, 40/60 periods | ADE and phasor stationarity |

The CPML contract explicitly calculates alpha from 4 um. It also records
kappa, sigma, orders, target reflection, and physical thickness. No upstream
default is permitted in the fresh runner.

The exact sequence is coordinate refinement:

1. run low-cost pilots to reject unstable or obviously reflecting contracts;
2. converge full z on all exact-binary references;
3. freeze the selected z level and converge the Au-window x/y mesh;
4. freeze it and converge outer x/y and PML grid;
5. converge gap, PML thickness/profile, and time;
6. rerun a joint selected-mesh confirmation;
7. only then evaluate a device candidate.

If changing a later axis invalidates an earlier comparison, the affected
ladder is rerun. A coordinate sweep is not a license to assume separability.

## Quantities that must converge

Total power alone is insufficient. Every geometry, polarization, and adjacent
mesh pair records and gates:

- source-only calibrated incident power;
- Q versus closed-surface flux;
- previous/late complex-field stationarity;
- total Q;
- maximum material/component-resolved Q change;
- complex E spatial NRMSE;
- conservatively remapped Q spatial NRMSE;
- TaIrTe4 temperature NRMSE and maximum temperature;
- signed terminal current.

The current check uses a mixed tolerance: 1% relative or 0.05 nA absolute,
whichever is larger. A sign is not considered preserved unless both currents
are at least 0.5 nA from zero. Final endpoint promotion specifically requires
`I_Ea >= +0.5 nA` and `I_Eb <= -0.5 nA` under the current campaign convention.

Two successive mesh-pair comparisons must pass. One coarse/fine agreement is
not a convergence certificate.

## Source and provenance requirements

Every combination of grid, PML, time contract, and polarization receives a
new all-air source calibration. Reusing the historical source scaling is
forbidden.

Every solve record must contain:

- complete FDTDX git commit and dirty-tree hash;
- Python/JAX/CUDA package lock and GPU identity;
- code, material, mask, grid, PML, time, and source-calibration hashes;
- raw complex fields, component Q, flux monitors, remapped heat, temperature,
  and weighting-field/current data in a portable configured raw root;
- an atomic progress manifest that revalidates every raw hash on resume.

The current branch now contains full source and runtime locks, but intentionally
does not contain the historical raw NPZ files. None of the old summaries can be
silently reused as a new certificate; every fresh raw artifact must be produced
through the locked runtime and re-hashed.

## Fail-closed promotion rule

The contract intentionally reports:

```text
is_mesh_certificate = false
optimizer_start_allowed = false
```

Those fields may change only in a result certificate produced by real solves
that passes all required reference geometries, both polarizations, two
successive comparisons per axis, the joint selected-mesh confirmation, the
physical-device contract, and independent solver endpoint comparison.
