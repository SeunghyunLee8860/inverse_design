# Lumerical z-mesh downstream multiphysics findings

Date: 2026-08-24. These are RTX 6000 Ada development diagnostics for the
linked 1.25/12.5-to-0.625/6.25-nm Ea exact-control pair. They are not B200 or
production evidence. Raw FSP/JSON/NPZ inputs and outputs remain outside Git
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
