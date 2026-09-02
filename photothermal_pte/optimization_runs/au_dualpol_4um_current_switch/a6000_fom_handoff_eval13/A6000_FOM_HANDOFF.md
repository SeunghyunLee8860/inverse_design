# A6000 FOM-change handoff

This is an immutable density-restart snapshot copied from the live B200 run
without stopping it.  The source code is commit
`be8aa5c0d824a1abcaae844c7bc5553c730a679f`.

## Snapshot

- beta: 2
- completed evaluation: 13
- FOM: 9.832544948595896 nA
- Ia: +9.832544948595896 nA
- Ib: -9.855522194283846 nA
- grayness: 0.43050143330213925
- checkpoint attempt: 6
- checkpoint latent and latest successful latent: bitwise identical

The current FOM is `min(+I(E||a), -I(E||b))`, implemented through the MMA
epigraph.  Start with `contract.py`, `lumerical_4um_signed_objective.py`,
`lumerical_4um_optimizer.py`, and
`41_optimize_lumerical_4um_dualpol_continuation.py`.

## Before using the state

Run:

```bash
python photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/a6000_fom_handoff_eval13/verify_bundle.py
```

The snapshot is a density restart.  MMA internal asymptotes/history are not
serialized, so a changed FOM must start a new MMA from this density.  Do not
claim a bitwise continuation of the B200 MMA.

Create a new branch/worktree before changing the FOM.  Do not rewrite the
live production branch.  If the FOM or its gradient changes, regenerate the
component-Yee and full-chain AD-FD certificates; do not reuse the B200
certificates as proof for the new objective.

The A6000 has a different GPU UUID.  Regenerate exactly four source-only
calibrations there: Ea/Eb at xy100 and Ea/Eb at xy50.  The optical/PDE route
must remain Lumerical FDTD Maxwell plus the repository custom CUDA thermal
and electrical PDE.  Never use FDTDX or Lumerical HEAT/CHARGE.

To request the portable density restart, point the continuation at:

```bash
bundle=photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/a6000_fom_handoff_eval13
export AU_LUMERICAL_RESTART_CHECKPOINT="$bundle/continuation_checkpoint.npz"
export AU_LUMERICAL_RESTART_MANIFEST="$bundle/restart_manifest.json"
```

The A6000 Codex must discover its local Lumerical/Python/GPU paths and must
not claim B200 promotion.  The restart validator will hash-check the state.

## Files

- `bundle_manifest.json`: hashes and snapshot metadata
- `continuation_checkpoint.npz`: continuation state, including latent density
- `stage_final_state.npz`: portable terminal-stage representation
- `latest_successful_state.npz`: original live-run successful state
- `restart_manifest.json`: relative-path stopped-run restart provenance
- `objective_history.json`: all successful callbacks through evaluation 13
- `source_production_manifest.json`: immutable source manifest evidence

