# Au material and density-interpolation audit

Status: `OFFLINE_AU_MATERIAL_CONTRACT_READY_FDTD_READBACK_PENDING`

This checkpoint performs no Maxwell, thermal, electrical, adjoint, or optimization solve.

## Frozen 10 µm optical endpoint

- Ordal Au: `n + ik = 12.1 + 69.2i`
- Relative permittivity: `epsilon = -4642.23 + 1674.64i`
- The earlier Lumerical CRC value at 11 µm is not reused as the 10 µm production endpoint.

Source: [Ordal et al. (1987)](https://doi.org/10.1364/AO.26.000744); [CC0 tabulation](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml).

## Interpolation decision

The legacy linear-complex-epsilon law is retained only as a diagnostic. The production candidate interpolates complex refractive index and then squares it. This is the nonlinear law described for plasmonic FDTD topology optimization by [Zeng et al.](https://doi.org/10.1021/acsphotonics.1c00260). Gray density is explicitly not a physical effective medium.

The nonlinear candidate preserves both endpoints and remains passive in this offline audit. It still crosses `Re(epsilon)=0`, so only a Lumerical material-fit/readback and binary-control campaign can certify it.

## Transport references

- Bulk-reference electrical conductivity: `4.11522634e+07 S/m`
- Bulk-reference thermal conductivity: `317 W/(m K)`
- Wiedemann-Franz check at 300 K: `301.235 W/(m K)`
- Initial electrical control uses `S_Au=0`; `+1.94 µV/K` is reserved for sensitivity.

These are bulk references, not certified properties of the fabricated Au film. Film thickness, deposition, grain size, and Au/TaIrTe4 contacts remain named physical uncertainties.

## Next fail-closed gate

Open a new v261 session, import the Ordal sampled material, and compare requested, fitted (`getfdtdindex`), and native index-monitor values. No optimization is permitted before that gate and binary Au/air controls pass.
