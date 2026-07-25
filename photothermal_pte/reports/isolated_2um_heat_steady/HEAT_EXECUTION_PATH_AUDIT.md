# Isolated 2 um steady-state HEAT execution-path audit

Baseline optical commit: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`

## Scope inspected

The complete `photothermal_pte` export was inventoried before changing the
thermal workflow: 36 Python files (13,184 lines at the audit point), all report
Markdown, JSON and CSV contracts, all compact numerical artifacts, solver logs,
and the metadata/hash of every binary solver artifact. Optical production code
and the validated `Q_on` values were not modified.

The active optical path is:

1. `bundle/tairte4_volume_model.py` builds the periodic 6 um optical cell.
2. `eqc_lib.py` pins and verifies the production optical contract.
3. `02_export_fdtd_qon.py` post-processes the validated production FSP and
   exports physical/unit-response `Q_on`.

The pre-existing thermal path is:

1. `00_probe_heat_api.py` probes DEVICE objects and conductivity storage.
2. `01_validate_heat_analytic.py` runs the single isotropic slab control.
3. `03_import_qon_heat_steady.py` builds a legacy 6 um by 6 um thermal cell.
4. `04_validate_heat_scaling.py` changes source scale and Si depth.
5. `05_validate_heat_transient.py` is a transient path and is prohibited in
   the present task.

`run_stage1.py` defaults to all six legacy stages, including scaling and
transient. It must not be used for this steady-only validation.

## Existing path versus requested finite-device contract

| Requirement | Existing implementation | Disposition |
|---|---|---|
| TaIrTe4 footprint | 6 um by 6 um | does not satisfy 2 um by 2 um |
| Thermal lateral domain | fixed 6 um | no 4/8/16/32 um sweep |
| Si depth | initial 2 um; legacy 3/6/12 um follow-up | no requested 2/5/10/20 um sweep |
| Lateral boundary | implicit adiabatic | requested far x/y fixed 300 K |
| Top exposed boundary | assumed adiabatic | not explicitly partitioned or read back |
| Interface conductance | perfect geometric contact | no independent top/bottom/oxide-Si G |
| Thermal mesh | solver defaults in physical case | no controlled mesh convergence |
| Saved fields | two T planes and one flux plane | no full 3-D T/delta-T/flux |
| Energy balance | bottom only, legacy 5% | no six-face balance; requested tolerance is 1% |
| Domain convergence | Si-depth-only legacy helper | no lateral convergence and wrong sweep |

## Validated Q artifact compatibility

The immutable production artifact is a periodic 6 um by 6 um `Q_on` grid with
shape `(241, 241, 36)` and integrated power
`1.6790733985800054e-11 W` in unit-response mode.

Direct integration over the requested 2 um by 2 um TaIrTe4 footprint gives
`5.465816178457092e-12 W`, only `32.552574432300224%` of the immutable source
power. Therefore `67.44742556769978%` of the validated power lies outside the
requested finite TaIrTe4 footprint. Restricting the imported source to that
solid would be geometric clipping and would fail the required FDTD-to-HEAT
power-conservation test by far more than 0.5%.

This is an independent fail-closed gate:

`BLOCKED_Q_ARTIFACT_INCOMPATIBLE_WITH_2UM_FOOTPRINT`

No cropping, periodic tiling, gain, smoothing, or rescaling is permitted.

## Conductivity and interface capability

The existing v261 round-trip evidence stores scalar `10 W/(m K)` correctly but
returns scalar zero after requesting diagonal `[14.4, 3.8, 1.0] W/(m K)`.
The documented Solid thermal material exposes a scalar
`thermal conductivity.constant`; no verified tensor storage path is present in
the probed object.

The legacy physical script contains no interface-conductance implementation.
The requested top, bottom, and oxide/Si conductances therefore require a
separate API capability test before any full-device solve.

Per the task contract, failure of the conductivity tensor gate is:

`BLOCKED_ANISOTROPIC_K_UNSUPPORTED`

An isotropic average is not an allowed fallback.

## Raw artifact policy

The local `.fsp`, `.ldev`, `.mat`, `.npz`, logs, and 3-D solver outputs remain
outside Git. The stage-specific execution script writes their SHA-256, byte
size, local server path, and generation command to a manifest.
