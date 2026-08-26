# Lumerical-only execution boundary

This is the authoritative solver boundary for the active Au inverse design.

- Maxwell forward: Lumerical FDTD 2026 R1.2.
- Maxwell adjoint: Lumerical FDTD 2026 R1.2 distributed FieldRegion source.
- Thermal forward/adjoint: repository custom CUDA finite-volume PDE.
- Electrical weighting/adjoint: repository custom CUDA sparse PDE.
- Lumerical HEAT: not used.
- Lumerical CHARGE: not used.
- Any alternative Maxwell solver: forbidden in the production import graph.

`lumerical_only_boundary.py` statically audits the 36 active production source
files and aborts if a forbidden Maxwell package is imported or already loaded.
The production launcher, forward, Yee-Jacobian, CUDA-PDE, and Maxwell-adjoint
entry points execute this gate before doing work.

Historical alternative-solver files remain in the directory only to preserve
the Git record. They are not an executable dependency of the production
Lumerical path. Generic overlap and provenance helpers formerly reached
through historical files now live in `lumerical_4um_overlap_remap.py` and
`lumerical_4um_provenance.py`. The continuous topology occupancy used by the
active path lives in `lumerical_4um_material_fraction.py`.

## Au thermopower contract

The forward current now contains both

\[
I = I_{\mathrm{TaIrTe4}} + I_{\mathrm{Au}}.
\]

The Au term uses `S_Au=+1.94 uV/K` on in-plane floating-Au sheet edges. The
value is the 300 K bulk absolute thermopower tabulated by Cusack and Kendall,
Proc. Phys. Soc. 72, 898 (1958), DOI
`10.1088/0370-1328/72/5/429`; it is a reference scenario, not a certified
50-nm-film value. The numerical void-conductivity floor is subtracted from
the thermoelectric source, so an exact void creates exactly zero Au PTE.

No measured vertical Au/TaIrTe4 interface or out-of-plane thermopower is
available. It is explicitly set to zero rather than guessed. That assumption
and the bulk-to-thin-film uncertainty require a parameter sweep or measurement
before an experimental prediction claim.

The Au source participates in the electrical forward, `dI/dT_Au` thermal
adjoint, and direct `dI/drho` derivative. The temperature and separate
TaIrTe4/Au current contributions are written to every new custom-PDE result.

## Validation state

On 2026-08-26, completed Lumerical Ea/Eb native-Q artifacts at the same
beta-2 projected state were reused without a Maxwell solve. The custom-CUDA
fixed-Q downstream AD-FD gate passed:

| polarization | total current (nA) | TaIrTe4 (nA) | Au (nA) | AD-FD errors |
|---|---:|---:|---:|---:|
| Ea | +0.0296343 | +0.0313155 | -0.00168120 | 1.01e-6, 2.60e-7 |
| Eb | -0.0251781 | -0.0313150 | +0.00613691 | 1.55e-6, 3.88e-7 |

Each case took about 26.5 s on one RTX 6000 Ada and performed five custom
thermal/electrical forwards plus one adjoint pair. It performed zero Maxwell
solves and zero Lumerical HEAT/CHARGE solves. This certifies the downstream
density derivative with fixed optical Q only.

The full combined blocker was subsequently closed at code commit `80e3ef8a`.
This certificate uses the **active optimizer mapping**, not the historical
mapping: 81x81 latent density -> 250-nm nonperiodic conic filter -> beta-4
projection -> one shared 81x81 physical occupancy -> 80x80 custom-PDE cells.
The centered latent step was 0.0025 in one common smooth direction.

| polarization | baseline current (nA) | AD (A/rho) | centered FD (A/rho) | relative error |
|---|---:|---:|---:|---:|
| Ea | -7.6677681 | -2.479281441e-8 | -2.479165780e-8 | 4.6651e-5 |
| Eb | -14.6552073 | -4.818613528e-8 | -4.818049197e-8 | 1.1711e-4 |

The signed balanced-objective AD/FD error is `4.6651e-5`; its two epigraph
constraint errors are `4.6651e-5` and `1.1711e-4`. The latent/projected
transpose errors are `1.33e-16` and `1.37e-16`. Every gate passed without an
empirical gradient scale or finite-difference fit. The evidence manifest is
external at
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_s_au_combined_adfd/active250_commit_80e3ef8a/active250_combined_adfd_manifest.json`.
It records zero alternative-Maxwell solves and zero Lumerical HEAT/CHARGE
solves.

Older combined latent certificates used `NOMINAL_MAPPING`, whose conic-filter
radius is 500 nm. They remain useful projected-density/solver evidence but do
**not** certify the active 250-nm optimizer chain. Scripts 36, 38, and 39 now
fail closed unless the pair identifies `OPTIMIZER_250NM_MAPPING` and its
250-nm solid/void contract.

The required 50-nm final-promotion source calibrations also passed on
2026-08-26 at commit `52b6ed79`, using the same GPU-5 UUID, development
accelerator policy, Lumerical build, and CV0 z contract as the 100-nm
calibrations. Both use mesh label
`fine_z2p5_bulk50_xy50_cv0_pml8_span20_z6_t1ps`. Ea and Eb solver times were
82.59 s and 82.51 s; their incident powers were both
`3.17671555e-14 W`. The external manifest is
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/r12_gpu5_xy50_source_only_cv0_52b6ed79/xy50_source_calibration_manifest.json`.

The prior production continuation omitted `S_Au`. Its artifacts are preserved
for diagnosis but its objective and gradients are stale. Never resume that
MMA checkpoint. The next production run must use a new committed worktree, a
new empty output root, and the exact uniform `rho=0.5` beta-1 start.

## Measured evaluation time

Nine completed evaluations from the stopped Lumerical run give the realistic
cost of one optimizer physics evaluation:

- complete Ea+Eb forward/PDE/adjoint evaluation: mean 1107.94 s (18.47 min),
  median 1101.98 s (18.37 min), range 17.69--19.57 min;
- one Lumerical forward: mean 155.07 s (2.58 min), or about 5.17 min for Ea+Eb;
- one Lumerical Maxwell adjoint: mean 218.40 s (3.64 min), or about 7.28 min
  for Ea+Eb;
- one custom-CUDA thermal/electrical forward+adjoint chain: mean 16.10 s per
  polarization, or about 32.2 s for Ea+Eb;
- the remaining roughly 5.5 min is component-Yee Jacobian work,
  postprocessing, process startup, and I/O.

These are measured engineering-run times, not guarantees. The new `S_Au`
algebra does not add another Lumerical solve and should change the custom-PDE
portion only slightly.
