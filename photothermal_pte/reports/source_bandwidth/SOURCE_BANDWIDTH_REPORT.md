# TaIrTe4 source-bandwidth validation

HEAT was not run. No Q channel was deleted; no clipping, gain, or rescaling was applied.

| Source range | Pulse type | Flat-y closure | Flat-y TMM error | Disk closure | Solver state | Result |
|---|---:|---:|---:|---:|---:|---:|
| 3.6-4.4 µm | standard | 223265552.366081% | 100.000103% | 11.693050% | CONVERGED/CONVERGED | FAIL |
| 3-6 µm | broadband | 0.000418% | 0.115412% | 0.061206% | CONVERGED/CONVERGED | PASS |
| 2.67-8 µm | broadband | 113.655061% | 51.504473% | DIVERGED (invalid) | CONVERGED/DIVERGED | FAIL |
| 3-12 µm | broadband | 0.003576% | 0.113003% | 0.058990% | CONVERGED/CONVERGED | PASS |

The 2.67–8 µm disk run terminated after the solver divergence marker; its post-run flux/Q values are invalid and were excluded from selection.
The 3.6–4.4 µm source used Lumerical's standard-pulse branch and failed both flat and patterned closure tests.

## Selection

The narrowest range passing all three 0.5% criteria is **3-6 µm**.

The compact per-case component absorption, pulse properties, fitted epsilon, dt, and solver state are in `source_bandwidth_cases.csv`; full mesh coordinates remain in the local case artifacts.

## Production contract (proposal only)

- Source: 3-6 µm broadband; analysis remains a single point at 4 µm.
- TaIrTe4 sampled material: 2.7–13.2 µm.
- Mesh: auto non-uniform, conformal variant 1, accuracy 5; no global_uniform_mesh.
- TaIrTe4 z override: 5 nm.
- Production code was not modified by this report.

## Selected-range regression

- flat_x: solver=CONVERGED, closure=0.003011%, TMM error=0.335691%
- flat_y: solver=CONVERGED, closure=0.000418%, TMM error=0.115412%
- flat_45: solver=CONVERGED, closure=0.001312%, TMM error=0.054570%
- disk_x: solver=CONVERGED, closure=0.061206%
