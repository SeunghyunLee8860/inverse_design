# Paper-IR reduced GPU diagnostic smoke

> Superseded interpretation (2026-07-30): the 9.18% number below compares
> different Q and six-face control volumes and is **not** an FDTD
> energy-conservation error. A later matched-volume smoke gives 0.0227%
> native-Yee and 0.1878% common-grid closure. See
> `PAPER_IR_MATCHED_CONTROL_VOLUME_SMOKE_REPORT.md`. This file retains the
> original unmatched-volume checkpoint and raw numbers for provenance.

## Status

- Official project status: `PARTIAL_PAPER_IR_CONTROL_VALIDATION_BLOCKED_OPTICAL_RUNTIME_AND_UNRESOLVED_EDGE_METRIC`
- Diagnostic substatus: `FAILED_DIAGNOSTIC_ONE_POL_GPU_SMOKE_SIX_FACE_CLOSURE`
- This is **not** a paper-like production optical result.
- No thermal, PTE, adjoint, gradient, or optimization calculation ran.
- No CPU FDTD fallback, Q clipping, smoothing, gain, rescaling, tiling, or
  source deletion was used.

## What ran

The single approved smoke used a 12 x 12 um straight-45-degree half-plane,
one `a` polarization, 6 um Gaussian aperture, 2 um waist, six PML boundaries,
24 PML layers, and 10 nm flake-region z mesh.  It retained the production
material closure `epsilon_x=epsilon_b`, `epsilon_y=epsilon_a`,
`epsilon_z=epsilon_b`, but deliberately reduced the lateral optical geometry
and monitor set.  It ran on GPU 4, an NVIDIA RTX 6000 Ada Generation; GPU 2
was not used because only 4.8 GB was free before launch.

The engine completed 39,374 iterations normally.  Logged FDTD size
was 402 x 402 x 161
(26,018,244 gridpoints), precise GPU
memory estimate was 1.336 GiB, solver wall time was
155.807 s, and GPU stepping took 142.705 s.

## Result

| Metric | Value |
|---|---:|
| native source power | 2.339588229783e-15 W |
| common-grid P_Q | 8.701460132991e-17 W |
| native-component-grid P_Q | 8.704063329997e-17 W |
| six-face P | 9.580832894734e-17 W |
| common-grid closure | 9.178458% |
| native-grid closure | 9.151288% |
| native/common Q difference | 0.029908% |
| mismatch / sum(abs(face power)) | 0.369893% |
| final auto shutoff | 1.810760e-05 |

Component powers on the common Q grid are:

- Qx: 1.802012616061e-17 W
- Qy: 6.844714403843e-17 W
- Qz: 5.473311308673e-19 W

Qz is finite and nonzero, as required by the lossy
`epsilon_z=epsilon_b` closure.  The hotspot is
(1.491525,
1.491525,
0.000000) in (um, um, nm).

## Why this historical comparison failed

The 9.18% comparison is numerically reproducible, but it is not a valid
same-control-volume energy-closure metric and is not corrected empirically.

1. The control volumes are not identical.  The six-face box is x/y = +/-5
   um, while the actual common Q output ends near +/-4.542 um.  Because the
   TaIrTe4 half-plane continues laterally, the surface encloses a lossy shell
   that the volume-Q monitor does not integrate.
2. The run exhausted its 1.2 ps simulation time with final auto shutoff
   1.810760e-05, above the requested 1e-5.  The DFT convergence
   criterion therefore was not reached.
3. The six-face net power is only
   4.042% of the sum of absolute face
   powers.  This cancellation makes the absorption closure sensitive to small
   residual face errors.
4. Native-component and common-grid P_Q differ by only
   0.0299%.  Component
   interpolation is not large enough to explain the closure failure.

The existing data cannot separate the shell contribution from finite-time DFT
error.  Doing so requires another solve with a matched control volume and
sufficient decay time.  Per the one-smoke fail-closed contract, that solve was
not started.

## Provenance

- Generation commit: `0f086255e92092573ccfe01376ded1b0fc335647`
- Raw Q NPZ: `/home/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/diagnostic_smoke_a_w2_L12_dz10_gpu4_20260730/diagnostic_q_native_artifact.npz`
- Raw Q NPZ size: 57,367,466 bytes
- Raw Q NPZ SHA-256: `59aa4b6e4f0289c8b425d80785f387b9f5d1b574b2cde9dbfc817194d9b7b76a`
- FSP, log, coordinate NPZ, case JSON, and read-only postprocess hashes are
  recorded in `PAPER_IR_DIAGNOSTIC_RAW_ARTIFACT_MANIFEST.json`.  The transient engine H5 was not
  retained after session close, so the manifest explicitly records it as
  absent and does not invent a hash.
- Individual face powers are in `paper_ir_diagnostic_six_face_fluxes.csv`.

The raw per-case JSON and raw solver artifacts were not modified.
