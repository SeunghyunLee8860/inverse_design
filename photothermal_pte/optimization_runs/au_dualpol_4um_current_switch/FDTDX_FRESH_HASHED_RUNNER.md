# Hashed numerical-case workflow for fresh FDTDX

Status: **runner contract, L500 time settling, and Courant convergence validated; first full-domain-z ladder run and rejected**

The fresh source-only and exact-binary material runners are no longer limited
internally to the anchor mesh, 16 periods, Courant 0.5, and the default CPML.
They consume one canonical numerical-case contract containing the complete
`MeshSpec`, time request, and CPML request.  The contract records its own
canonical SHA-256; the runner additionally requires the SHA-256 of the actual
JSON file bytes.

The old v1 source-pair certificate remains immutable evidence for the four
endpoint controls at its original commit.  It lacks this new canonical case
identity and is intentionally not accepted for a v2 convergence case.

## Required chain for every unique numerical case

The repository and pinned FDTDX dependency must be clean for promoted source
pairs and material cases.  Raw directories below must be absolute, pre-created,
writable, and empty.  Keep the case JSON outside those result directories.

Generate one case.  This example is the primary L-reference 16-period anchor
time level:

```bash
/home/seunghyun200/.venvs/fdtdx-fresh-py312/bin/python -m \
  photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_case_contract \
  --output /absolute/raw/contracts/l500_anchor_t16.json \
  --mesh-axis anchor --mesh-level 0 \
  --total-periods 16 --window-periods 4 --courant-factor 0.5
```

Record both printed hashes.  `file_sha256` is supplied to the runners;
`case_contract_sha256` identifies the canonical request inside reports.

Run both all-air polarizations on the same clean code commit and case file:

```bash
FDTDX_FRESH_GPU_INDEX=<one_idle_physical_gpu> \
FDTDX_FRESH_OUTPUT_DIR=/absolute/raw/source_t16_Ea \
photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_fdtdx_fresh_gpu.sh \
  -m photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only \
  --polarization Ea \
  --case-contract /absolute/raw/contracts/l500_anchor_t16.json \
  --case-contract-sha256 <file_sha256>

FDTDX_FRESH_GPU_INDEX=<the_same_idle_physical_gpu> \
FDTDX_FRESH_OUTPUT_DIR=/absolute/raw/source_t16_Eb \
photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_fdtdx_fresh_gpu.sh \
  -m photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_only \
  --polarization Eb \
  --case-contract /absolute/raw/contracts/l500_anchor_t16.json \
  --case-contract-sha256 <file_sha256>
```

Create the pair certificate in another empty directory:

```bash
/home/seunghyun200/.venvs/fdtdx-fresh-py312/bin/python -m \
  photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_source_pair \
  --ea /absolute/raw/source_t16_Ea/FDTDX_FRESH_SOURCE_ONLY.json \
  --eb /absolute/raw/source_t16_Eb/FDTDX_FRESH_SOURCE_ONLY.json \
  --output-dir /absolute/raw/source_pair_t16
```

The pair generator rehashes both reports and raw NPZ files and independently
reconstructs the canonical case.  It blocks different or noncanonical Ea/Eb
cases, a case-file audit mismatch, mesh/time/PML disagreement, a dirty source
repository, or polarization-specific normalization.

After recording the pair-certificate file SHA-256, run both material cases:

```bash
FDTDX_FRESH_GPU_INDEX=<one_idle_physical_gpu> \
FDTDX_FRESH_OUTPUT_DIR=/absolute/raw/l500_t16_Ea \
photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_fdtdx_fresh_gpu.sh \
  -m photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_exact_binary_pilot \
  --reference l_shape_4um_with_500nm_arms --polarization Ea \
  --case-contract /absolute/raw/contracts/l500_anchor_t16.json \
  --case-contract-sha256 <file_sha256> \
  --source-pair /absolute/raw/source_pair_t16/FDTDX_FRESH_SOURCE_ONLY_PAIR.json \
  --source-pair-sha256 <source_pair_file_sha256>
```

Repeat with `Eb` in a distinct empty output directory.  The pilot checks the
canonical case before importing/building the FDTD model, then checks the full
realized mesh, time step/count, all six PML profiles, object placements, source
vector, finite-dt ADE readback, and exact-binary mask again before the solve.

## Completed 16/24/32-period chain

The complete chain has now run for the exact-binary
`l_shape_4um_with_500nm_arms` reference on the anchor spatial grid. The source
and material runs were made at clean repository commit `01a8ad8a`; the
independent certificate generator was committed and pushed as `5e376ce1`. Raw
files remain outside Git under
`/home/seunghyun200/fdtdx_results/l500_time_settling_01a8ad8a_20260824`.

Use `fdtdx_fresh_time_settling_certificate.py` to revalidate the complete
chain. It requires all six external hashes rather than discovering and trusting
whatever bytes happen to be present. The generated certificate is
`time_settling_certificate_5e376ce1/FDTDX_FRESH_TIME_SETTLING_CERTIFICATE.json`,
SHA-256
`20ab99b8488606475d2ed8d604d1810c9f3953176b68f42ed0689685ed505ab0`.
It passed 21/21 top-level gates, selected 24 periods, and confirmed that choice
with 32 periods. The 16-period level remains preserved as a rejected coarse
case rather than being relabeled as valid.

The verifier checks canonical case and source-pair SHA bindings, complete gate
schemas, clean source provenance, exact 375-pixel L500 geometry, zero Au Q
outside the binary mask, grid edges, independently reconstructed electric-Yee
dual volumes, raw Q integrals, field/Q stationarity, closed-flux agreement, and
both successive cross-time comparisons. It compares the actual fixed
`[-4,+4] um` x/y probe at `z=0.250 um`; the larger stored target detector is
not silently treated as the convergence probe.

## Completed four-level Courant chain

The 24-period Courant chain `[0.5, 0.375, 0.25, 0.1875]` is complete under
`/home/seunghyun200/fdtdx_results/l500_courant_4d79a439_20260824`. Revalidate
it with `fdtdx_fresh_courant_certificate.py`, supplying the four case-file and
four source-pair SHA-256 values recorded in
`FDTDX_FRESH_CONVERGENCE_DESIGN.md`. The clean-commit certificate is
`courant_certificate_876cfff3/FDTDX_FRESH_COURANT_CERTIFICATE.json`, SHA-256
`7fd86bc8582d27002c226b6395a7d803f29ba98deda4abff00e60def9560a869`.

The rejected coarse 0.5-to-0.375 result remains in the certificate: its worst
material/Cartesian Q-component change is 2.339%, above the 2% gate. Both finer
successive pairs pass, so Courant 0.25 is selected and 0.1875 is its independent
confirmation. The verifier also records and constrains the two raw-run commits;
it does not falsely report that all four levels came from one commit.

## Rejected first full-domain-z chain

`run_fdtdx_fresh_full_z_campaign.sh` completed z factors 2, 4, and 8 at
24 periods and Courant 0.25 on GPU 7. The raw root is
`/home/seunghyun200/fdtdx_results/l500_full_z_150a7592_20260824`. The clean
corrected certificate is
`full_z_certificate_7b687684/FDTDX_FRESH_FULL_Z_CERTIFICATE.json`, SHA-256
`319743a29b8dd4869c5d1feedf564850ff10e4c30fb1888fd28eb7ed8764036c`.
Use the exact contract and source-pair hashes in
`FDTDX_FRESH_CONVERGENCE_DESIGN.md` when revalidating.

All artifact, exact-grid-coordinate, placement, source/material provenance, CFL,
end-time, binary-mask, and conservative-remap audits pass. Both z2-to-z4 and
z4-to-z8 physical comparisons fail. In the finer pair, total Q changes by
1.846 percent, the fixed tangential probe by 6.882 percent, and conservative
spatial Q by 13.925 percent; their limits are 1, 2, and 5 percent. No z level is
selected and optimization remains forbidden. Extend full-domain z before any
x/y or downstream multiphysics promotion. Raw NPZ files remain external.

The first z16 extension attempt is also preserved externally. Its canonical
case-file SHA-256 is
`74fca414c3c82ce1031f0f688cab0c3a3d252de6ea66e2fceb22ee40c0493e3a`.
Source-only Ea stopped before the FDTD solve because the single-Drude float32
carrier refit error was `1.17579e-4`, above `1e-5`. Do not retry that directory
or relax the gate. The clean-commit solver-free diagnosis from `a4cf66d5` is
external at `ade_precision_a4cf66d5/FDTDX_FRESH_ADE_PRECISION_DIAGNOSTIC.json`,
SHA-256
`bfa98e74b81eae816b888bfbe1b460f94d5cf407f4be4954742c91e2b540911c`.
Even a 0.01-to-10-times-seed Au single-pole scan bottoms out at
`2.21443e-5`. The full-tensor follow-up at `ecc33c22` is external at
`ade_precision_ecc33c22/FDTDX_FRESH_FULL_MATERIAL_ADE_PRECISION_DIAGNOSTIC.json`,
SHA-256
`cb15e83073887fc0b7bd328f81b1b5463087024d98277bd740027bd82a412741`.
It shows z16 TaIrTe4 a also fails (`2.75931e-5`) and every current single-pole
axis fails at z32. Stable positive-strength two-pole candidates cover all four
axes at z8/z16/z32 with recurrence roots no larger than one, but remain
candidate-only. Any implementation changes the material law and requires fresh,
hash-distinct z8, z16, and z32 source/material runs before comparison; old
single-pole and new two-pole levels must never form a pair.

The canonical candidate-law generator was committed at `f959a9ef`. Its external
z8/z16/z32 law-file SHA-256 values are
`6352e58e0b3b2449f5316948adb3247bfc9c71547cbb2252a8beba69571d67bc`,
`558eae569446993096081320c1f6e9439ee78ef799c8aeb0b0af8810a72e6fb2`, and
`302ab4e8991b55d0fb17c2ff5332b156fb29401ac04e026e0394f7e6c1fbcd1d`.
The z32 case-file SHA-256 is
`33398486f542fa0f1c7b063011e61992f7830b7cd36c25c8d6863c553aa3fbf4`.
All files remain candidate-only and cannot be passed to historical runners
until explicit two-pole solver-array readback support is implemented.

The pinned coefficient preflight at `7504045c` passed bit-exact c1/c2/c3 and
zero-c4 readback for every axis at all three levels. External JSON SHA-256
values are z8
`1b892395e5d989dcb12a679d0d0c19389d2017cff026326670fd9a074cf0aeb2`,
z16 `aa91d260271982f2bf3c4aba523cda6127a18ab6ddac50514628fbe8f5f59a9f`,
and z32
`4f5e3da15bcbc571fd8c9d98bc30ca4be4f5b5ac8b3266efe78a01d93d9202b6`.

Placed-array support and its zero-time-step preflight were committed at
`011e0d36`. The historical single-pole builder remains the default; candidate
use is opt-in, canonical-self-hash checked, and rejected for adjoint placement.
The external report hashes under `two_pole_solver_array_011e0d36` are z8
`126ac0aa053cc31c576700f1527e8a6f9a9d1dbdbda433bf7b38df13f272ec5c`,
z16 `28887e54bf29b51f818b962e0072fb774cb496e9bf19cfcf4c0bf858c9d0465c`,
and z32
`adfa0e0332bc487df31296b7116ea757f501a408f73e77939cf989ec68c74266`.
All three pass exact solver-array, material-axis, binary-Au, case, dt, mesh, and
PML readback. They execute zero field steps and remain candidate-only.

Reproduce one level in a new absolute output file with:

```bash
CUDA_VISIBLE_DEVICES=<one_idle_gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
FDTDX_SOURCE_DIR=/absolute/pinned/fdtdx \
/home/seunghyun200/.venvs/fdtdx-fresh-py312/bin/python -m \
  photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_fresh_two_pole_solver_array_preflight \
  --case-contract /absolute/case_zN.json \
  --case-contract-sha256 <case_file_sha256> \
  --material-law /absolute/two_pole_law_zN.json \
  --material-law-sha256 <material_law_file_sha256> \
  --fdtdx-source /absolute/pinned/fdtdx \
  --output /absolute/new/FDTDX_FRESH_TWO_POLE_SOLVER_ARRAY_zN.json
```

The candidate-bound source/pair runner was committed at `e722ba73`. Its first z8
Ea/Eb pair is complete under `two_pole_forward_e722ba73/z8`. The Ea report/NPZ
SHA-256 values are
`30cbc8b18c5aaaa289994b5bafe2c7b8821983aff31b7f97e7c9647d4b113901` and
`7e586000eb4a5681011062f9fe78e972120a8b0bb9b05c3eda55fd24f326d133`;
the Eb values are
`49788884d2ce62660cfba923a123940426211f8394fe98c6103a7058782bf459` and
`93cff39dbeaf200f0db43987c077b700ae1713774c71fc480bc7c982f8e393e1`.
The law-bound pair SHA-256 is
`d5196cc7c715260e5c0436ccc59ca258c22173ae9adc69340405dc5e1e05a582`.
All gates pass; the raw incident-power mismatch is `1.1513098e-7`.

The next blocker is a separate candidate exact-binary material runner and its
z8 Ea/Eb time/stationarity/field/absorption results. z16 and z32 still need
their own source pairs. Adjoint and optimizer paths remain disabled throughout.

## Convergence rule

The 16-, 24-, and 32-period levels are three different numerical cases.  Each
requires its own case JSON, Ea/Eb source-only runs, and source-pair certificate.
The same applies to every Courant, mesh, domain, or PML level.  A source pair
must never be copied between levels even when incident powers look similar.

There is no implicit-anchor CLI mode. Both the absolute case path and its file
SHA-256 are mandatory for source-only and material commands. Completion of
the time and Courant ladders, and execution of a rejected z2/z4/z8 ladder, does
not authorize an optimizer, thermal/electrical solve, PTE-current claim, or mesh
certificate. The next numerical case is a finer full-domain-z extension at 24
periods and selected Courant 0.25 only after a separately contracted material
representation passes its own readback/time gates, with a fresh source pair at
every new level.
