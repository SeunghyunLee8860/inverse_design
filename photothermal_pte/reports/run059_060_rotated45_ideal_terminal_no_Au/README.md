# Run059/060: rotated 45-degree ideal-terminal optimization without Au

## Model contract

- Device and electrical terminal direction: +45 degrees.
- Crystal axes remain fixed: Lumerical `x=b`, `y=a`.
- Optical model: stable axis-aligned Run58 proxy with no Au.
- Thermal model: evaporated TaIrTe4/SiO2 interface,
  `G = 73,700 W m^-2 K^-1`, with no Au layer or Au interface.
- Electrical model: ideal equipotential electrodes exist only in the
  weighting-field solve.
- Maxwell and thermal/electrical calculations used one GPU sequentially.

## Final status

| Run | Polarization | Continuous current at 285 uW | Chosen exact current at 285 uW | Exact objective change | Status |
| --- | --- | ---: | ---: | ---: | --- |
| 059 | `E||a` | +670.204 nA | +655.262 nA | -2.229% | Exact 500 nm geometry passed; 1% objective-preservation gate failed |
| 060 | `E||b` | +801.690 nA | +793.731 nA | -0.993% | Validated exact-binary optimization |

Run059 is retained as a completed, physically validated calculation, but it
must not be labeled a validated final exact-binary optimum because its exact
cleanup loss exceeds the declared 1% limit.

## Published artifacts

- Run059: [`results_v5_no_Au`](../../optimization_runs/run_059_diagonal45_evaporated_sio2_Ea_bounded_official_dfm_exact_repair/results_v5_no_Au)
- Run060: [`results_v5_no_Au`](../../optimization_runs/run_060_diagonal45_evaporated_sio2_Eb_bounded_official_dfm_exact_repair/results_v5_no_Au)
- `FINAL_RESULT.json`: final gates, currents, solver checks, and provenance.
- `EXACT_REPAIR_DIAGNOSTIC.json`: independent 500 nm audit and repair search.
- `chosen_exact_candidate_density.npz`: selected 241 x 241 binary density.
- `chosen_exact_candidate_fields.npz`: mapped heat source, nodal temperature,
  weighting potential, and element weighting-field gradient.
- `optimization_history.json` and `evaluation_*.json/png`: complete continuation
  history through beta 128.

Large regenerable FSP projects remain in the server artifact store and are
identified by path, size, and SHA-256 in `RAW_ARTIFACT_MANIFEST.json`.
