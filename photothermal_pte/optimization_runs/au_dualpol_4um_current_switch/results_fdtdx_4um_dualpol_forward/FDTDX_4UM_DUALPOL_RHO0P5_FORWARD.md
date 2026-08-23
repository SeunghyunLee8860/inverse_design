# FDTDX 4 um dual-polarization rho=0.5 forward gate

Status: **VALIDATED_FDTDX_4UM_DUALPOL_RHO0P5_FORWARD**

This checkpoint ran two real Maxwell forward solves on the identical finite six-PML grid.
It did not solve thermal, weighting, current, adjoint, or optimization problems.
The target-plane field is a total-field diagnostic and is not labelled a pure incident beam.

| polarization | P_Q (W) | P_Au (W) | P_TaIrTe4 (W) | closure | Q window change | runtime |
|---|---:|---:|---:|---:|---:|---:|
| Ea | 1.60973825e-13 | 4.23396610e-14 | 1.18634164e-13 | 0.43493% | 0.07909% | 6.61 s |
| Eb | 3.87471755e-13 | 1.03719651e-13 | 2.83752104e-13 | 0.15108% | 0.01536% | 6.05 s |
