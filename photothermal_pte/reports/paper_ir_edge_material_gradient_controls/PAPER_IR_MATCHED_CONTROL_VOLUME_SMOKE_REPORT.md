# Paper-IR matched-control-volume GPU smoke

## Status

- Official project status:
  `PARTIAL_PAPER_IR_CONTROL_VALIDATION_BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC`
- Smoke status:
  `FAILED_MATCHED_CONTROL_VOLUME_SMOKE_AUTO_SHUTOFF_UNRESOLVED`
- The matched-volume optical closure passes, but the requested
  auto-shutoff gate does not.
- This is a reduced 12 x 12 um, one-polarization diagnostic. It is not
  production Q, a paper reproduction, or a thermal/PTE result.

## One solver run

The single approved rerun used the straight 45-degree TaIrTe4 half-plane,
`a` polarization, a 6 um finite Gaussian aperture, 2 um waist, 24-layer PML
on all six boundaries, and a 10 nm flake-region z override. The material
closure was `epsilon_x=epsilon_b`, `epsilon_y=epsilon_a`,
`epsilon_z=epsilon_b`.

The v261 solver ran on GPU 4 only. It completed 131,247 iterations and
4.000005 ps normally. Solver wall time was 640.481187 s, including
627.725899 s of GPU stepping. No CPU FDTD fallback occurred.

The run itself completed, but the original Python postprocessor stopped when
it rejected an Ex-grid sample whose dual-cell edge coincided with a flux
face. The raw `case_result.json` and its
`BLOCKED_EXECUTION_ERROR` status are preserved. A separate read-only recovery
opened the completed FSP, called neither `run` nor `runanalysis`, and
intersected each component-specific Yee dual cell with the independently
read six-face volume.

## Matched control-volume result

The realized common control volume is:

- x: [-4.542372881356, +4.542372881356] um
- y: [-4.542372881356, +4.542372881356] um
- z: [-180, +50] nm

| Metric | Value | Gate |
|---|---:|---:|
| native Yee `P_Q` | 8.715867473376e-17 W | |
| common-grid `P_Q` | 8.701470836178e-17 W | |
| six-face inward power | 8.717844152299e-17 W | |
| native Yee / six-face closure | 0.022674% | <0.5%, pass |
| common-grid / six-face closure | 0.187814% | <0.5%, pass |
| native / common difference | 0.165177% | <0.5%, pass |
| final auto-shutoff | 1.80982e-5 | <1e-5, **fail** |
| max independent E/index coordinate mismatch | 8.47033e-22 m | <1 fm, pass |

The common-grid component powers are:

- `Qx`: 1.802014885272e-17 W
- `Qy`: 6.844722670579e-17 W
- `Qz`: 5.473328032790e-19 W

The native Yee component powers are:

- `Qx`: 1.802053911372e-17 W
- `Qy`: 6.859080281676e-17 W
- `Qz`: 5.473328032790e-19 W

The common-grid hotspot is at
(x, y, z) = (1.491525, 1.491525, approximately 0) um with
`Q=55.0020706 W/m^3` in native source-amplitude units.

## Correct interpretation

The earlier 9.18% number compared different control volumes: its six-face
box extended to approximately +/-5 um while its Q integration ended near
+/-4.542 um. It must not be interpreted as a 9.18% FDTD energy-conservation
error. With matched realized bounds, both native-Yee and common-grid
closures pass 0.5%.

The old claim of exactly zero E/index coordinate mismatch is also retracted.
It was produced by copying field coordinates into the index-coordinate path.
The saved index-detail coordinates were subsequently read independently.
The maximum mismatch is 8.47033e-22 m, so collocation is validated only
within floating-point precision, not by an asserted exact zero.

Extending the requested simulation time from 1.2 ps to 4 ps did not reach
the requested decay threshold: the log plateaued near 1.81e-5. Therefore the
smoke remains fail-closed despite its passing power closure. No empirical
normalization, Q clipping, smoothing, gain, global rescaling, tiling, or
source deletion was applied.

## Consequence

No thermal, PTE, adjoint, gradient, or optimization calculation follows from
this smoke. The result closes the control-volume and coordinate-audit
questions, but it does not promote the Q artifact while the explicit
auto-shutoff gate remains unresolved.

## Provenance

- Solver generation commit: `3b08e8a9251f1531ac053bf017697c157fd785f4`
- Quadrature fix commit: `6a9b74c`
- Raw common-grid NPZ:
  `/home/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/diagnostic_matched_smoke_a_w2_L12_dz10_4ps_gpu4_20260730/diagnostic_q_common_grid_artifact.npz`
- NPZ size: 55,581,646 bytes
- NPZ SHA-256:
  `951a4fe38a3ff57a48c2a11499e7e16ea1c431e2206b6fb5241c74e734c59ee2`

All raw paths, sizes, hashes, commands, and the individual face powers are
recorded in the accompanying JSON, CSV, and manifest. Raw NPZ/FSP/H5/log
files remain outside Git.
