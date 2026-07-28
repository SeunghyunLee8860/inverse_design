# CPU TFSF protection of the central 2 µm ROI

Status: `VALIDATED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE`

## Scope

This is an empty-air source-integrity and runtime gate in Ansys Lumerical
2026 R1.2 v261. It is not a device, optical-Q, thermal, AD, FD, or
optimization result.

- protected ROI: `x,y=[-1,1] µm`;
- lateral FDTD domain: `4×4 µm`;
- TFSF transverse span: `2.6 µm`;
- six outer boundaries: PML, no periodic/Bloch boundary;
- source: normal incidence, x polarization, 3–6 µm;
- analysis: 4 µm;
- mesh: auto non-uniform, accuracy 5;
- CPU: one process, 16 threads; GPU resource disabled.

## Baseline result (PML 24)

- mean |E|² error from the unit-amplitude incident field:
  `0.01443117%`;
- spatial intensity RMS: `0.00000856%`;
- spatial intensity peak-to-peak: `0.00005593%`;
- maximum phase deviation: `6.18654569e-06 degree`;
- Ey/Ex L2: `2.35998913e-08`;
- Ez/Ex L2: `5.33590441e-08`;
- closed-box energy error:
  `0.00007052%`.

Runtime and memory from the native solver log:

- grid: `86×86×70`
  (`517720` gridpoints);
- engine overall wall time: `3.270815 s`;
- time stepping: `1.865693 s`;
- Python `run()` wall time including engine launch/return: `5.525168 s`;
- complete session wall time: `10.508481 s`;
- peak CPU memory: `0.194561 GiB`;
- completed iterations: `644` (auto-shutoff reached).

## PML refinement (PML 32)

- mean |E|² relative change: `1.03409152e-06%`;
- energy-closure absolute change:
  `5.58559053e-07` percentage point;
- grid: `102×102×86`;
- engine overall wall time: `4.346625 s`;
- Python `run()` wall time: `7.462453 s`;
- complete session wall time: `12.466042 s`;
- peak CPU memory: `0.224140 GiB`.

Both PML cases passed every preregistered ROI and energy gate. PML 24 is the
faster promoted source-gate setting; PML 32 is retained as the refinement
control.

## Geometry limitation before device AD–FD

A 4 µm TaIrTe4 flake cannot be placed in this same 4 µm FDTD domain for a
valid TFSF calculation. The finite flake must be fully inside the TFSF box,
and the TFSF box must be strictly inside the PML boundaries. Therefore this
result validates the illumination in the central 2 µm ROI, not the final
finite-flake device geometry. The device calculation must enlarge the FDTD
domain while keeping the physical design/PTE ROI exactly 2 µm.

Raw FSP and solver logs remain outside Git. Their paths, byte sizes, and
SHA-256 values are recorded in `CPU_TFSF_4UM_ROI_RAW_MANIFEST.json`.
