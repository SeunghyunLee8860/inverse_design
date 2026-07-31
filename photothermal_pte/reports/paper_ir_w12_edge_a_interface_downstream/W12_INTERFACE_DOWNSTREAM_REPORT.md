# W12 interface-slab and downstream convergence

Status: `BLOCKED_W12_INTERFACE_SLAB_OR_DOWNSTREAM_CONVERGENCE`

No new FDTD calculation was run. Completed 50/25 nm artifacts were used
without Q clipping, smoothing, gain, global rescaling, tiling, or deletion.

## Interface power

| Metric | 50 nm | 25 nm | Change |
|---|---:|---:|---:|
| total P_Q (W) | 2.255431773196e-11 | 2.254734131029e-11 | 0.0309% |
| z=0 dual-layer power fraction | 2.8732% | 2.8594% | — |
| -10..0 nm slab power fraction | 9.4203% | 9.4156% | — |

The z=0 common-grid sample has a bounded dual cell
`[-2.500,
2.500] nm`.
The -10..0 nm slab is integrated by exact dual-cell overlap, not by selecting
cell centres.

- slab power change: `0.0810%`
- slab equal-power lateral NRMSE:
  `1.3725%`

## Component interface assignment

The saved FSPs were reopened read-only. No `run` or `runanalysis` call was
made. Ex/Ey have an exact z=0 index sample; Ez is z-staggered and instead has
samples at approximately -2.5/+2.5 nm. Full numerical values and E/index
coordinate mismatches are stored in the summary JSON.

- Ex/Ey z=0 median loss participation is approximately `50.0122%` of the
  bulk fitted material loss; the -5/+5 nm samples are material/air.
- Ez has no z=0 sample: -2.5 nm is material and +2.5 nm is air.
- Maximum independently read E/index coordinate mismatch:
  `6.776e-21 m`.
- Central component-local interface cell volume is approximately
  `1.25e-23 m³` at 50 nm x/y and `3.125e-24 m³` at 25 nm x/y.

## Named downstream model

Both Q artifacts use the same exact-overlap/nearest-support remap and the
same 60 µm explicit anisotropic/interface FVM: 20 µm Si depth, 100 nm core
x/y cells, and 10 nm TaIrTe4 z cells. The PTE number is a uniform-45-degree
diagnostic with lab `x=b`, `y=a`; it is not a solved-electrode terminal
current.

| Downstream metric | 50→25 nm difference |
|---|---:|
| Q_T volume-weighted NRMSE | 0.9685% |
| Tmax | 0.1698% |
| TaIrTe4 T-field NRMSE | 0.1230% |
| in-plane gradient-vector NRMSE | 0.9945% |
| uniform-45 PTE diagnostic | 0.4260% |

The temperature field and signed uniform-45 PTE diagnostic pass 0.5%, but
the interface-slab spatial shape, mapped Q_T, and in-plane gradient-vector
metrics do not. Therefore this remains a partial downstream pass, not a
source-convergence promotion.

Any pass is limited to this named remap, thermal grid/boundary model, and
PTE functional. It does not relabel the raw full-3D voxel-Q gate as
universally converged.

Published figures:

- `W12_INTERFACE_DOWNSTREAM_COMPARISON.png`
- `W12_INTERFACE_GRADIENT_COMPARISON.png`
