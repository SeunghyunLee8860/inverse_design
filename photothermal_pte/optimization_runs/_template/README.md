# RUN_ID

This folder is one immutable inverse-design run.  Record the scientific
question and every approved deviation from the baseline in this file before
launching a solver.

## Required sequence

1. Complete `run_config.json` and external artifact paths.
2. Run the repository preflight with `--require-external`.
3. Record the preflight output and update `STATUS.json`.
4. Launch only the reviewed optimizer command.
5. Write one compact checkpoint summary per accepted iteration.
6. Preserve failed iterations and raw artifact SHA-256 values.
7. Publish final JSON/CSV/plots/report without committing raw FSP/NPZ files.

No solver is launched by the template itself.
