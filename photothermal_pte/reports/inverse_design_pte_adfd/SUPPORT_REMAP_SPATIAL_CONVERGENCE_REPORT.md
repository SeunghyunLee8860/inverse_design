# Support-remap spatial-deposition convergence

Status: `VALIDATED_SUPPORT_REMAP_SPATIAL_CONVERGENCE`

## Matched pair

Both forward cases use exactly the same:

- uniform physical `rho=0.5`;
- `2 µm × 2 µm × 600 nm` design;
- `7.2 µm × 7.2 µm` optical outer domain;
- `z=[-3.6,3.4] µm` optical outer bounds;
- normal-incidence CPU TFSF;
- six PML faces;
- PML-32 with stabilized x/y and standard z;
- realized PML-inner target, TFSF, Q/six-face volume, source
  normalization, material stack, and transverse `50 nm` mesh.

Only the TaIrTe4 optical mesh changes from `dz=5 nm` to `dz=2.5 nm`.
Both independent forward runs passed the `P_Q/P_six <0.5%` closure gate.

## Independent exact-support remaps

Each FSP was reopened and its native Yee `Qx/Qy/Qz` was independently
extracted. Each native source was conservatively embedded into the same
thermal target:

- named TaIrTe4 footprint: `4 µm × 4 µm`;
- thermal domain / Si depth: `32 µm / 20 µm`;
- `core_xy_cell_size=100 nm`;
- `flake_dz=25 nm`;
- `design_dz=100 nm`;
- target array shape: `76 × 76 × 76`.

The support projection is a fixed linear operator that relocates staggered
Yee control-volume energy only along z to the nearest exact TaIrTe4 cell in
the same x-y column. It deletes no nonzero source and applies no clipping,
smoothing, gain, global rescaling, or periodic tiling. Its transpose tests
were `2.30080e-15` and `2.01596e-15`.

TaIrTe4-exterior nonzero cell counts were `0 / 0`.

## Convergence result

- mapped `P_Q`, 5 nm / 2.5 nm:
  `1.6890916194508477e-12 / 1.6887880194040323e-12 W`
- mapped-power relative difference: `1.7977392e-4` (`0.0179774%`)
- volume-weighted spatial-Q NRMSE: `4.8813859e-3` (`0.488139%`)
- normalized cell-energy shape NRMSE: `4.8757876e-3`
- lateral-integrated energy NRMSE: `1.6534580e-3`
- depth-integrated energy NRMSE: `2.3006082e-4`
- peak-Q relative difference: `3.9759784e-4`
- Qx relocated fraction, 5 nm / 2.5 nm:
  `3.1218334% / 1.5636897%`

The promoted spatial gate is `0.5%`; the total spatial-Q NRMSE passes but is
close to the limit. Later thermal mesh/domain convergence must therefore
remain independent and cannot be inferred from this optical-mesh result.

The mapped hotspot changes from
`(-0.05,-0.05,-0.0125) µm` to `(0.05,-0.05,-0.0125) µm`.
This is one `100 nm` thermal cell across a reflection-symmetric,
near-degenerate central maximum, not a macroscopic hotspot displacement.

No thermal solve, PTE evaluation, adjoint, finite difference, or optimization
was executed in this checkpoint.
