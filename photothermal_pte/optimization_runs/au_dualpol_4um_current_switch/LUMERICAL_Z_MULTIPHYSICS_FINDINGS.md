# Lumerical z-mesh downstream multiphysics findings

Date: 2026-08-24. The first sections record CV0 RTX 6000 Ada development
diagnostics for the linked 1.25/12.5-to-0.625/6.25-nm Ea exact-control pair;
the final section records the subsequent staircase series. They are not B200
or production evidence. Raw FSP/JSON/NPZ inputs and outputs remain outside Git
under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.

## Selected Lumerical material-filter definition

The selected diagnostic now follows Ansys' official multi-material advanced
absorption example, not a home-made effective-epsilon decomposition. Lumerical
`pabs_adv` first constructs the common-grid spatial `Pabs`. The material filter
then compares both real and imaginary parts of `index.index_x` against the
material index returned by `getfdtdindex`, with relative tolerance `1e-15`,
and multiplies `Pabs` by that exact mask. See Ansys'
[higher-accuracy absorption method](https://optics.ansys.com/hc/en-us/articles/360034915693-Calculating-absorbed-optical-power-Higher-accuracy)
and
[multiple-material example](https://optics.ansys.com/hc/en-us/articles/360034395254-Calculating-absorbed-optical-power-Higher-accuracy-method-with-multiple-materials).

Script `29_extract_lumerical_4um_official_pabs.py` applies only
`runanalysis("finite_device_pabs")` to a completed, SHA-verified FSP. It does
not rerun the Maxwell engine. Future calls to
`25_run_lumerical_4um_exact_au_control.py` save `Pabs_W_m3`, `Pabs_index_x`,
and their axes directly in the raw NPZ. Script
`28_validate_lumerical_4um_z_multiphysics_pair.py` verifies the companion
NPZ/JSON hashes, scales by each run's own source-only incident power to 285 uW,
maps each exactly identified material only into thermal cells of that material,
and calls the repository custom CUDA thermal and electrical solvers. Lumerical
HEAT and CHARGE are never called.

No missing conformal-interface power is redistributed. No local or global
rescaling, clipping, smoothing, gain, or tiling is applied. Common-grid Pabs
contains only roundoff-scale negative interpolation samples: their integrated
magnitude is at most `6.91e-19` of signed absorption in these four inputs. They
are preserved, audited, and gated below `1e-12`; they are not clipped.

The earlier two diagnostics remain useful negative evidence:

1. Mapping every entire Yee dual cell into all overlapping thermal cells
   conserved power but leaked interface absorption into low-conductivity air,
   producing false 2.4--4.8-K hotspots.
2. Reconstructing material loss as `Q/Im(epsilon_effective)` times physical
   material loss/overlap removed the air hotspot, but its material-power
   reconstruction error was 0.78--1.55% even on the fine member. It is no
   longer the selected definition.

## Official-filter results

Every selected remap conserves its filtered material power below `1e-15`.
Official spatial Pabs reproduces `Pabs_total` to `3.77e-16` or better. All
custom CUDA PDE residual, energy-balance, and terminal-balance gates pass.

| exact control | coarse unassigned absorption | fine unassigned absorption | remapped Q volume-L2 NRMSE | TaIrTe4 temperature NRMSE | Tmax change | symmetry-current cancellation | result |
|---|---:|---:|---:|---:|---:|---:|:---:|
| empty, Ea | 3.0869% | 1.5552% | 2.4932% | 1.7909% | 1.8115% | 5.36e-5 | fail |
| full Au, Ea | 2.1728% | 1.1939% | 2.3285% | 1.3931% | 1.7418% | 6.84e-4 | fail |

Empty Tmax is 1.01220 K on the coarse member and 1.03087 K on the fine
member. Full-Au Tmax is 0.061825 K and 0.062921 K. The official material mask
therefore confirms, rather than removes, the downstream z blocker.

Empty and full are mirror-symmetric controls whose physical x-current should
cancel. The official filtered source fails the one-part-per-million symmetry
gate because the unassigned conformal-interface Pabs is not symmetric enough
after filtering. Those small signed currents are diagnostics of an incomplete
heat-source partition, not valid PTE control currents.

## Consequence and next mesh axis

The 1.25/12.5-to-0.625/6.25-nm pair passes total Q, six-face flux, and common
endpoint-plane Maxwell metrics, but it does not pass the official
material-resolved source, thermal field, Tmax, or symmetry-current gates. The
z axis remains blocked; x/y convergence and optimization must not start.

The unassigned fraction decreases by roughly a factor of two when z is halved,
which is consistent with mixed-index interface samples. Blindly extending CV0
to 0.3125/3.125 nm would approximately double an already 53.6-million-point
grid and still does not predict a sub-0.5% empty-control omission. Ansys also
states that CV0 excludes metal interfaces from CMT, CV1 includes them but can
create metal artifacts, and the correct choice requires convergence testing;
see
[mesh refinement selection](https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object).

The next bounded experiment is therefore the already-required MCM6
CV0/CV1/staircase interface axis at a tractable linked mesh, with matching
source-only calibration and the same official Pabs filter. Staircase is not
assumed accurate: it is tested because it gives one material per Yee sample
and hence an unambiguous thermal material assignment. Its Maxwell and
downstream convergence must still pass independently. Do not close any gap by
rescaling filtered material power to total Pabs.

## Staircase 0.625/6.25-nm extension

The matching 0.625/6.25-nm staircase source, exact-empty, and MCM6 exact-full
runs are complete. The 1.25/12.5-to-0.625/6.25-nm pair passes all four Maxwell
sub-gates for both controls. The official material-filtered Pabs was then
mapped into the same custom CUDA thermal/electrical system with no Lumerical
HEAT/CHARGE and no rescaling, clipping, smoothing, gain, or tiling.

| exact control | coarse unassigned absorption | fine unassigned absorption | remapped Pabs volume-L2 NRMSE | TaIrTe4 temperature NRMSE | Tmax change | symmetry-current cancellation | result |
|---|---:|---:|---:|---:|---:|---:|:---:|
| staircase empty, Ea | 0.001018% | 0.001019% | 1.5580% | 0.2144% | 0.1813% | 5.344e-5 | fail |
| staircase full Au, Ea | 0.198425% | 0.198992% | 1.6799% | 0.3484% | 0.2152% | 6.020e-4 | fail |

Every individual material-assignment, mapping-conservation, thermal residual,
thermal energy-balance, electrical residual, terminal-balance, and finiteness
gate passes. Staircase therefore fixes the CV0 mixed-index material omission,
and the temperature outputs now change by less than 0.5%. It does not yet
close the stricter volumetric-source or symmetry-current gates.

The remapped-Pabs L2 error exhibits nearly first-order convergence over three
staircase meshes:

| exact control | 2.5/25 -> 1.25/12.5 nm | 1.25/12.5 -> 0.625/6.25 nm | improvement ratio |
|---|---:|---:|---:|
| empty, Ea | 3.0327% | 1.5580% | 1.947 |
| full Au, Ea | 3.2575% | 1.6799% | 1.939 |

The L2 difference is dominated by the first 10-nm TaIrTe4 thermal cell below
the z=0 interface; full Au also has a comparable contribution from the first
10-nm Au cell above it. This is a thin-stack interface-resolution error, not
an unassigned-material-power error. A first-order estimate predicts roughly
0.8% for the next halving and roughly 0.4% for the following halving, so the
current finest source must not be promoted.

The zero-current diagnostic is a separate issue. Empty current is about
11.0 pA against a 207-nA absolute integrand scale and is nearly unchanged by
z refinement; full current is about -19.2 pA against 31.9 nA. Since staircase
material omission is already tiny, the earlier attribution of this residual
solely to unassigned conformal Pabs is not supported. Lateral/source symmetry,
custom-PDE discretization, iterative tolerances, and the one-ppm gate itself
must be isolated without using these residuals as physical PTE currents.

The next z experiment should refine the thin stack first while holding the
already-fine 6.25-nm bulk/air/PML limit fixed. This avoids blindly doubling
all full-domain z cells. A matching source-only/empty/full control is still
required for every new mesh. Do not begin x/y convergence or optimization
until the volumetric-source blocker and the meaning of the symmetry-current
gate are resolved.
