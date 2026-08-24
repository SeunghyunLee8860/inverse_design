# Hashed numerical-case workflow for fresh FDTDX

Status: **runner contract implemented and unit-tested; no convergence case run**

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

## Convergence rule

The 16-, 24-, and 32-period levels are three different numerical cases.  Each
requires its own case JSON, Ea/Eb source-only runs, and source-pair certificate.
The same applies to every Courant, mesh, domain, or PML level.  A source pair
must never be copied between levels even when incident powers look similar.

There is no implicit-anchor CLI mode. Both the absolute case path and its file
SHA-256 are mandatory for source-only and material commands. This workflow does
not authorize an optimizer, thermal/electrical solve, PTE-current claim, or mesh
certificate.
