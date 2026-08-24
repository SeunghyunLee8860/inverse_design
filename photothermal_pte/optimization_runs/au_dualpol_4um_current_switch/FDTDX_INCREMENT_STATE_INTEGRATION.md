# FDTDX increment-state production-path integration

## Current endpoint

The later exact-binary z32 campaign is complete and still not converged. No mesh is selected; z64, adjoint timing, gray optimization, and optimizer restart are forbidden. `FDTDX_Z32_STOP_AND_AU_DESIGN_AUDIT.md` is the authoritative current result. Sections below retain the chronological integration evidence and must not be read as current launch instructions.

## Scope and boundary

This checkpoint integrates the cancellation-resistant Lorentz/Drude
`(P, delta-P)` ADE into an isolated FDTDX fork. It is opt-in; the upstream
second-order polarization recurrence remains the default. No Lumerical file or
run was edited, launched, or reinterpreted. No GPU was used for this checkpoint.

This closes small-domain Lorentz float32 and Drude float64
forward/checkpointed-adjoint software gates.
It does **not** certify the 4-um full geometry, any mesh, a gray Au material law,
the thermal/electrical maps, or optimization readiness.

## Reproducible fork state

- pinned upstream base: `f26f84b70a8cceec9b889553955a868624736bf1`
- isolated kernel commit: `24d0cb2374bf03b6bfdc528b189c69685b74dfee`
- opt-in production integration commit:
  `fc09ce54dc32ea13e27d2af799cdb3771801bf65`
- Drude full-FDTD AD-FD test commit:
  `6cc0e97252ee0b95de5016e8db1a5b414177efa4`
- fork branch: `codex/increment-state-ade`
- exported integration patch:
  `fdtdx_patches/0002-feat-dispersion-integrate-opt-in-increment-state-ADE.patch`
- patch SHA-256:
  `1532e032fbe3656b4397f6c8d94314339f4bd94b0e2583c162c317c106b901cb`
- exported Drude test patch:
  `fdtdx_patches/0003-test-dispersion-gate-increment-state-Drude-adjoint.patch`
- Drude test patch SHA-256:
  `77016668fb7dc77a7bdfbead26c9ce24b545ea246bb8a3dcc1fbcbe0fd2c3b31`
- integrated `src/fdtdx/increment_state.py` SHA-256:
  `bd2d11a3a5b10d49d3a9d13c997134f4cf9d7bb451b42a8536d673711890faf3`

A clean checkout reproduces the fork by applying patches `0001`, `0002`, and
`0003` in order to the pinned base. The project-side
`test_fdtdx_increment_state_integration_patch.py` pins the production patch
bytes, commit header, exact 16-file scope, required production markers, and the rule that the patch cannot touch Lumerical. It
separately pins patch `0003` to its test-only two-file scope.

## What changed in FDTDX

`SimulationConfig(dispersive_state_representation="increment")` now selects
the new path. Existing arrays are reused without another full-grid allocation:

- `dispersive_P_curr` stores `P`;
- `dispersive_P_prev` stores `delta-P`;
- `dispersive_c1/c2/c3` store `A/C/B`;
- `dispersive_c4` must be absent.

The same representation is used by uniform/static multi-material placement,
device parameter application, diagonal `update_E`, point/plane/mode sources,
mode-overlap detectors, carrier-frequency effective permittivity, and the TFSF
broadband impedance spectrum. CCPR, oriented poles, and a dispersive
full-tensor material path fail closed in increment mode.

## CPU validation evidence

All commands forced the CPU backend. The dedicated environment was
`/home/seunghyun200/.venvs/fdtdx-fresh-py312`.

- increment kernel/spectrum/full-FDTD gates: `12 passed`;
- four full-FDTD integration tests: `4 passed in 11.73 s`;
- dispersion/initialization/source/mode plus new gates:
  `167 passed in 33.66 s` after the Drude gate;
- complete FDTDX unit suite:
  `2605 passed, 2 skipped, 1 xfailed in 238.63 s`;
- project-side FDTDX audit suite including the patch gate:
  `164 passed in 11.924 s`;
- Ruff 0.15.22 on every changed Python file: no remaining errors.

The small driven lossy Lorentz scene ran through actual coefficient placement,
source correction, `checkpointed_fdtd`, and reverse-mode differentiation. For
one active `B` voxel:

```text
AD = 2.240709215e-01
FD = 2.239563912e-01
symmetric relative error = 2.556324618e-04
absolute difference = 1.145303249e-04
```

This Lorentz test is a full-FDTD checkpointed AD-FD gate, not merely
differentiation of the isolated one-cell kernel. A scoped-float64 passive
Drude `C=0` control also passes:

```text
AD = -4.862088902297e-05
FD = -4.862093454809e-05
symmetric relative error = 4.681640598235e-07
absolute difference = 4.552512737583e-11
```

## Remaining optimizer blocker: continuous dispersive mixing

The current generic FDTDX `Device` path linearly interpolates every stored
coefficient between its two materials. In increment mode that means a gray
air/Au voxel interpolates `A` and `C` as well as the coupling `B`. This changes
the oscillator damping/resonance with density; it is not automatically the
same as a fixed-pole susceptibility whose strength alone follows a chosen Au
occupancy law. The endpoints are exact, but intermediate gray voxels are not
yet a literature-backed material model.

Therefore this integration does not authorize a continuous-density inverse
design. Before that, the papers and the existing optical/thermal/electrical
maps must be audited together and one shared physical occupancy must be
defined. For a fixed-pole optical relaxation, the likely implementation is to
hold the physical pole dynamics fixed and interpolate only the susceptibility
strength, but that choice must be derived and AD-FD tested rather than assumed.
Exact-binary controls do not depend on this gray-law decision and are the next
allowed field-level timing/mesh probe.

## Runtime and GPU launch boundary

The old z16/t32 FDTDX model took about `19.1 min` for one polarization. Two
independent polarization solves therefore gave a measured Maxwell lower bound
near `38 min` per forward-plus-adjoint design iteration, or more than `63 h`
for 100 iterations before thermal/electrical work. That old validation grid is
not an optimizer grid.

No full-size timing exists yet for the patched increment-state fork, so it is
incorrect to promise a faster number. The next launch must be a short,
coarse, exact-binary timing/closure control with explicit compile and steady
execution times. Immediately before launch, inspect compute-process ownership;
run Ea and Eb concurrently on two distinct verified-idle GPUs, and never use a
GPU carrying another user's process. One FDTDX solve remains single-GPU.

Promotion remains blocked until those controls establish a practical runtime,
then source/material closure and z convergence pass on newly generated
increment-state artifacts. Old second-order source pairs and results cannot be
mixed into that comparison.

## Project-side exact-binary control runner

The project now has an explicit opt-in model path that passes
`dispersive_state_representation="increment"` into the patched fork. It uses
one mesh-independent passive physical pole per material axis and asks FDTDX to
generate realized float32 `A/C/B`; it never calls the historical per-mesh
float32 pole refit. The historical builder default remains `polarization`.

On the anchor mesh, CPU-only placement resolves `196 x 196 x 160` cells and
`25,664` time steps. Exact air/Au endpoint readback and the complete
Si/SiO2/TaIrTe4/Au material-stack audit pass. Realized epsilon relative errors
are `8.7825e-6` (Au), `1.00931e-5` (TaIrTe4 a), and `3.18525e-6` (b/c); all
responses are passive. The increment-state readback contract is `1e-4`; the
historical refit path retains its `1e-5` contract. This is a numerical
representation tolerance, not a gray-density law.

`fdtdx_increment_state_exact_binary_control.py` is the newly isolated cold
runtime/closure runner. It accepts one polarization and uses the 500-nm-arm
exact L reference by default. It checks the exact material stack, stationarity,
nonnegative Q, and volume-Q versus closed-flux closure, and records cold
build/compile/forward timing and visible-device memory statistics. Because no
new increment-state source-only pair exists yet, it intentionally removes
absolute absorbed fraction and Ea/Eb amplitude comparison from its gates. Raw
outputs remain external. `run_fdtdx_increment_state_control_gpu.sh` requires an
explicit physical GPU and refuses it if `nvidia-smi` reports any existing
compute process before setting `CUDA_VISIBLE_DEVICES`.

At this pre-launch checkpoint the affected tests are `9 passed, 3 subtests`;
Ruff, Python compilation, shell syntax, and diff checks pass. No GPU has yet
been used by the new runner. Commit and push this state before launching Ea/Eb.
Optimization remains forbidden.

## First B200 cold timing and closure result

Commit `c843276d1265a4652355b73ceecda2ce5be6230f` ran the same
500-nm-arm exact L reference concurrently as Ea on physical B200 GPU 6 and Eb
on physical B200 GPU 7. GPU 0 was occupied by another user and GPU 1 had a
new user launch, so neither was touched. The two chosen devices had no compute
process before either launch and contained only the two project processes
during the run.

External reports:

- Ea: `/home/seunghyun200/fdtdx_results/increment_state_control_c843276d/Ea/FDTDX_INCREMENT_STATE_EXACT_BINARY_CONTROL.json`, SHA-256 `36b48d9870b4cf46f2b5cc8159d9712e857af7ed6ec2f04a1d84c7fca45d485f`
- Eb: `/home/seunghyun200/fdtdx_results/increment_state_control_c843276d/Eb/FDTDX_INCREMENT_STATE_EXACT_BINARY_CONTROL.json`, SHA-256 `707bddd2fba704d9a4409139d54e4574b3f5a8c3c8d207e2c182f6c99221ec85`

Ea cold build/array preparation was `21.598 s`, cold compile plus forward was
`24.652 s`, host evaluation was `1.811 s`, and total was `48.076 s`. Eb was
`21.147 s`, `24.766 s`, `1.703 s`, and `47.631 s`. Since the cases ran
concurrently, the pair wall time was about 48 s. Peak JAX bytes-in-use were
about 3.71 GB per GPU; the allocator pool was about 4.36 GB and live
`nvidia-smi` process use was about 4.81 GiB. These are forward-only cold
numbers, not adjoint or full optimization-iteration timings.

Both material stacks, finite/nonnegative Q, total-Q drift, time-domain and
phasor closed-flux closure, and provenance gates passed. Q versus closed
phasor differed by only `9.0497e-5` (Ea) and `5.3532e-5` (Eb). The reports are
nevertheless correctly blocked: Au previous/late complex-field NRMSE was
`1.1339e-2` for Ea and `1.7934e-2` for Eb, above the `5e-3` gate. Eb Q spatial
NRMSE was also `5.5919e-3`, slightly above its `5e-3` gate; Ea was
`1.6996e-3`. TaIrTe4 field NRMSE was already below `8.25e-4`, and total Q
change was below `9e-5` in both cases.

Do not loosen the gates or call this mesh validated. The next allowed solve is
a same-mesh 24/32-period time-settling extension on two newly verified-idle
GPUs. Only after field and spatial-Q stationarity pass may newly generated
increment-state source controls and spatial mesh convergence begin.

### Canonical time-settling extension

Runner version `fdtdx-increment-state-exact-binary-control-v2` accepts
`--total-periods` and `--window-periods`, constructs a canonical `TimeSpec` and
`FreshCaseSpec`, and records the self-hashed realized case. Defaults remain
16/4. The GPU wrapper preserves its three safety arguments and forwards only
remaining runner options. Example for the next control:

```bash
run_fdtdx_increment_state_control_gpu.sh GPU_INDEX Ea /absolute/empty/output \
  --total-periods 24 --window-periods 4
```

This extension changes only simulation duration and detector-window timing;
mesh, material pole parameters, exact mask, source, PML, and gates remain
unchanged.

## Passed 24-period time-settling control

Commit `a7f2d6b9411a22cb18a2dcec23759a15c1519daa` ran the canonical
24-period/4-period-window extension concurrently on the same verified-idle
physical B200 GPUs 6/7. Both reports are
`VALIDATED_FDTDX_INCREMENT_STATE_EXACT_BINARY_CONTROL`; no 32-period run is
needed. The shared canonical case SHA-256 is
`4a1b16092a693953c075b9848bba3342951233b712e397005dc34312f6e30532`.

External reports:

- Ea SHA-256 `858f8d5b7ba42be29e18e0e1276a6da157d2cc21947c947a5a316f1f6baff309`
- Eb SHA-256 `ce7138c66301d7b16ba4f472a53a5c3e95e2aa9d89251f0714b724ec8e323d41`
- root: `/home/seunghyun200/fdtdx_results/increment_state_control_a7f2d6b9_t24/`

Ea cold compile+forward was `36.429 s` and total was `59.701 s`; Eb was
`36.693 s` and `60.029 s`. Concurrent pair wall time was about 60 s. Peak JAX
bytes-in-use were about 3.716 GB per GPU and the pool remained 4.364 GB.

Au previous/late field NRMSE fell to `1.8580e-4` (Ea) and `2.4892e-4`
(Eb), versus the `5e-3` gate. TaIrTe4 field NRMSE was at most `1.348e-5`. Q
spatial NRMSE was `3.2963e-5` and `6.9316e-5`; total-Q change was
`1.9868e-6` and `5.4228e-6`. Q/closed-phasor differences were
`1.1508e-4` and `7.0470e-5`. Every material, stationarity, Q, closure, GPU,
and provenance gate passed.

This closes anchor runtime and time settling only. It does not certify source
normalization, spatial mesh convergence, adjoint timing, a gray law, or the
optimizer. The next artifact must be a newly hashed patched-fork 24-period
source-only Ea/Eb pair; do not reuse the historical second-order pair.

## Increment-state source-only and pair pre-launch

`fdtdx_increment_state_source_only.py` builds the same canonical 24/4 anchor,
selects the patched increment state, then resets the complete domain to air. A
CPU placement/readback proves inverse epsilon is exactly one and every stored
`A/C/B` coefficient is exactly zero on the `196 x 196 x 160`, 38,496-step
case. It reuses the existing field stationarity, target-plane polarization,
beam-moment, incident-flux, and closed-flux evaluation math, but uses a new
status/version/provenance bound to patched commit `6cc0e97`. Detector fields
and grids are written only to an external NPZ.

`fdtdx_increment_state_source_pair.py` requires absolute Ea/Eb report paths and
explicit lowercase byte SHA-256 values. It rehashes both reports and raw NPZs,
checks finite schemas, exact polarizations, canonical identical 24/4 cases,
identical mesh/PML/placement/source/runtime/FDTDX provenance, clean worktrees,
and a `5e-3` incident-power mismatch limit. It then computes exactly one common
power and field-amplitude scale from the arithmetic-mean incident power;
per-polarization matching remains forbidden. The safe source wrapper rejects
any GPU with an existing compute process before CUDA export.

The new/affected source tests are `18 passed`; actual all-air CPU readback also
passes. No source GPU run or pair certificate exists at this pre-launch
checkpoint. Commit and push before running the pair.

## Validated 24-period increment-state source pair

At clean project commit `5756f50a15efca918a3318bd22a9c7bcf6c4ded8`, all-air
Ea/Eb cases ran concurrently on verified-idle physical B200 GPUs 6/7. Both
source reports passed every material, stationarity, polarization, beam, flux,
GPU, and provenance gate. Incident power was exactly
`1.88214721585922e-12 W` in both reports, so relative mismatch is zero.
Transverse purity is `0.9998025`, longitudinal fraction is `0.026369`, maximum
waist error is `0.036663`, closed residual is about `1.1e-6`, and maximum field
NRMSE is about `2.34e-6`. Cold compile+forward was `36.737 s` (Ea) and
`36.663 s` (Eb); pair wall time was about 61 s.

External artifacts and SHA-256 values:

- Ea report: `fe2415e6438cca995285d3c18b63fbea0fade0d7894e47ceac546293129accf4`
- Ea raw NPZ: `ef2ed341658202c2854c399159f94fcdcd2786c0658b692f9be84f2e923c4c0e`
- Eb report: `b89671354fb02ed7b8df68c4c529b14ad34bf613f9da4da8a417ac8fe7ee9c16`
- Eb raw NPZ: `0b10d9ec58e84a0d32a481a9e923c90d69fb5606917fdea8683c2e16234f625f`
- pair certificate: `6beab945b90513e9ce638932abdb25702fb4c97be4897abe6f794639fee98dba`
- root: `/home/seunghyun200/fdtdx_results/increment_state_source_5756f50a_t24/`

The validated certificate uses one arithmetic-mean common scale:
`common_power_scale=151422799.23618752` and
`common_field_amplitude_scale=12305.39715881562` for the 285-uW reporting
target. Every one of its cross-case/raw/hash/provenance gates passes. This
closes source normalization for the anchor case only. Spatially changed meshes
still require matching newly generated source controls or a rigorously proven
source-transfer rule; do not silently reuse this certificate on a different
case hash.

## Full-z source contracts prepared

Mesh-aware source cases at clean project commit `22d27c4a` completed on the
verified-idle GPU 6/7 pair. All reports and certificates pass.

- z2 (`196 x 196 x 80`, case `7d457ac10933752178ddd09efe7efd51f2672910d01562727a86bebe5c02ddbc`): Ea/Eb report hashes `72507b2da95b0b8a05f0637667d5fb1881edae0e04e2791efbdcfa5f369a7696` and `fdc3ff918503269f86898bf42cc43fa055844d71214bfdc2a270d9af905fd6ea`; pair SHA-256 `c1565e0bee3e79fe0ff0e87a3d891ebe9b419598953e760b68937dd2935cf9b3`; power mismatch `1.15523e-7`; cold forward `15.754/15.616 s`.
- z4 anchor (`196 x 196 x 160`, case `4a1b16092a693953c075b9848bba3342951233b712e397005dc34312f6e30532`): pair SHA-256 `6beab945b90513e9ce638932abdb25702fb4c97be4897abe6f794639fee98dba`.
- z8 (`196 x 196 x 320`, case `8dc9d5b2717b930b1585cd3a85cb9553dfba642a1290f52d3e5648a8c164193a`): Ea/Eb report hashes `695199ce723103b2f3f8c9332bb3e1bddf640afc17e436943b36b49a3ced6ea4` and `a3d456a47c9364050edc2120de762a663dc8c9c0fa959b460878269b03e94aa5`; pair SHA-256 `ffa8e5706d7dab65622757335e545c8a67dc15c08c420fd661ca5580d7ba3b4d`; mismatch zero; cold forward `101.789/101.407 s`; peak JAX bytes about 7.401 GB.

Roots are `/home/seunghyun200/fdtdx_results/increment_state_source_22d27c4a_z2_t24/` and `..._z8_t24/`; the z4 root was recorded above. These certificates authorize only their exact case hashes. The next runner must rehash and match the appropriate certificate before each material solve.


## Increment-state full-z material ladder is not converged

The source-bound runner committed at `74c523fd` evaluated the same exact-binary
500-nm-arm L reference at z2/z4/z8. Ea and Eb ran concurrently only on
verified-idle physical B200 GPUs 6/7; GPU 0, which carried an external solver
process, was not used. Every one of the six material cases passed its internal
material readback, stationarity, Q/flux closure, source binding, GPU, and
clean-provenance gates.

External material artifacts:

- z2 root `/home/seunghyun200/fdtdx_results/increment_state_material_74c523fd_fullz_z2_t24/`: Ea/Eb report SHA-256 `a252ec6d7a0a9767ce2d9928725e4ef0bdbd68a53d1d3c9b3fbedf29fd5ccea8` / `014cf2abb2535693c1681e435af4fe45b6ea3cea61f9a6c799dd6080250e455a`; raw SHA-256 `360ff73a1840bd1fe09b1e7caf823e72619aafbbded4a231e9c6a0e53cfbd421` / `06b54bfdb63ab9d6b117bd9643e92ab07564fb836e6e9fb96d1b412da0ab04e3`; total runtime `37.913/38.493 s` and cold forward `15.512/15.521 s`.
- z4 root `/home/seunghyun200/fdtdx_results/increment_state_material_74c523fd_fullz_z4_t24/`: report SHA-256 `6b080f1d390e4a4dbee7e530685e47910135b7f910c95c39114b06ff3684943e` / `eaad31ca9fa1e355b5ef912399466462416d367fd69d465fb9fc9040c2f31cc8`; raw SHA-256 `3a92380debd24c6bd7a099d20ad5790949b7ccd7e401f5f6cd1e4b42d6a64a9f` / `e1b3e5adb466eb25932ce2466fe4e76a063872ad4db4f32b01702b48d116650f`; total runtime `60.386/60.648 s` and cold forward `36.590/36.605 s`.
- z8 root `/home/seunghyun200/fdtdx_results/increment_state_material_74c523fd_fullz_z8_t24/`: report SHA-256 `5a1389eb62f676a8081595d9567028ad9bc4713873506cf59c8c66d532f0c540` / `1fd4217d60d349e34027fb36606264b27e571c79b0bf846bddc7a99895034f19`; raw SHA-256 `b79130aebe30ccef12f59c1f41284bb2b1f3c19a73ac0ccf1b5e9793fb226c1b` / `8702488e09ac4c6da306ba666a1ab4907c888b239c5bd31d533402da67b06029`; total runtime `129.888/129.831 s` and cold forward `101.689/101.649 s`.

The fail-closed certificate generator committed at `62137609` rehashed all
three source pairs, all six reports and NPZ files, recomputed Q integrals and
common 285-uW normalization from raw arrays, and used component-Yee physical-z
overlap restriction for the 3-D Q comparison. Certificate:
`/home/seunghyun200/fdtdx_results/increment_state_full_z_certificate_62137609/FDTDX_INCREMENT_STATE_FULL_DOMAIN_Z_CERTIFICATE.json`,
SHA-256 `92258e6ef598bcbe403090784e8d22757630cef0322f28605c96abd082e5bcae`.
All artifact/global checks pass, but both successive mesh pairs fail.

For z4-to-z8, the worst metrics are total-Q change `1.64575e-2` (limit
`1e-2`), fixed-probe complex-E NRMSE `7.00568e-2` (limit `2e-2`), conservative
3-D Q NRMSE `1.38791e-1` (limit `5e-2`), component-Q change `3.80702e-1`
(limit `2e-2`), and material-region complex-E NRMSE `9.38471e-1` (limit
`5e-2`). Source change, same-run stationarity, Q/flux closure, restriction
conservation, and all provenance checks pass. This is valid negative evidence:
the z4 anchor is not a production mesh, and z8 is not yet a converged selection.

Do not start adjoint timing, gray-density optimization, or non-z spatial axes.
The next permitted extension is a matching z16 canonical 24/4 source pair and
exact-L Ea/Eb material pair, followed by z8-to-z16 comparison under the same
gates. Raw outputs remain external. The independent Lumerical session remains
out of scope.


## z16 extension and interface diagnostic

The increment-state z16 resolver and v2 source/material runners were committed
at `18e84ddb`. The canonical z16 case has shape `196 x 196 x 640`, 24,586,240
Yee cells, 153,625 time steps, case-contract SHA-256
`b831a6de67048fd7b926fa011cea1394c99b99d51bf64e5534587c24cc5f3b63`,
and Courant `0.5`. Source-only Ea/Eb ran concurrently on verified-idle B200
GPUs 6/7. Cold forward was `320.179/319.937 s`, total was
`351.468/351.741 s`, and each process used about 17.17 GiB. Incident powers
were exactly equal at `1.8837390414888633e-12 W`.

z16 source artifacts under
`/home/seunghyun200/fdtdx_results/increment_state_source_18e84ddb_z16_t24/`:
Ea/Eb report SHA-256 `a88cd33bdd9fae2d730238818d887ed3cac13e368f29d9e6846e8f2429ac911f`
/ `18710d316ad2f9ccd33805f6b75f5db7e153ee44815310f5ab785311f0711913`;
raw SHA-256 `d00e134892edf5a3b15d712f7c5321f175d423301a1ce46deab2e0a60bba3782`
/ `9b9b9140c0644433dd7276a277430190d6ea7d689bec6133cfd0be5e20fa9dde`;
source-pair SHA-256 `dffb77b434bd2873f04305583cd7441c3c3de605c47d267f38020e64f54b90ee`.

Matching exact-L material cases under
`/home/seunghyun200/fdtdx_results/increment_state_material_18e84ddb_fullz_z16_t24/`
passed every internal gate. Ea/Eb report SHA-256 values are
`c6e192d265e2b35dbd9228fa65369c88ee34858b67238c032cef95bedf26841f`
and `aa96702d36c0653a7dd7c41ac7e0e20f14c9e68a43b3bf2dfe84b0e32934b756`;
raw SHA-256 values are
`6bba854619ee19e7c8feee4683a414890e4f07731898131c7876c18199271fa9`
and `e6d58c7280b2b2f3a05efa0261d635b3ed49f28a71270473a7d4cc8c972b637c`.
Cold forward was `320.223/320.512 s`, total `354.862/355.046 s`.

The cross-version certificate committed at `e2fbbd36` chains the immutable z8
certificate to the v2 z16 artifacts. External certificate SHA-256 is
`3e19b422b447f7605d3e40c8d5ada8a79560d0d7580a811ba31c6010a1e3d4fe`
under `/home/seunghyun200/fdtdx_results/increment_state_z16_extension_certificate_e2fbbd36/`.
All artifact/global gates pass, but z8-to-z16 remains blocked: total Q
`0.6306%` and conservative Q `3.1909%` pass; fixed-probe E `3.3382%`,
component Q `4.6970%`, and Au-region E `19.7920%` fail their 2%, 2%, and 5%
limits.

The corrected interface diagnostic at commit `39ebd077` has external SHA-256
`ded2ff57895608ffe84e0815573c712adf7fa551826d7a5a27322c582dc81f23`.
It shows that the Au field discrepancy is not confined to one or two boundary
Yee planes; trimming those planes does not reduce it. Global complex
scale/phase alignment lowers the Au discrepancy but leaves roughly 13--15%
residual. The maximum component-Q change carries only `0.6598%` of total
absorption, but that does not waive the declared gate.

A z32 forward is predicted from measured scaling at about 16--18 minutes and
about 34 GiB, so z32 source/material validation is still practical and is the
next allowed fine-grid diagnostic. A z64 forward is projected above 50 minutes
and must not be launched under the current runtime feasibility rule. No
optimizer, adjoint timing, non-z axis, or production mesh is authorized.


## z32 measured endpoint

The z32 diagnostic completed on idle GPUs 6/7. Source and material pairs took about 19.2 and 19.3 minutes wall time, respectively, with about 33.68 GiB per GPU. The source-pair SHA-256 is `e926729fe75cf5fa8fcd3a10e24137037963c2f218968262f663d1c62f2d4f6b`. The material report SHA-256 values are `f13c9ee53fd2b8fc5209324439a0a406c8d51ac879c44c5da3b4d316590eedc1` and `ffc23d15c448a2bb7002ae6f27231beef223aec8d38493326d1491536754847d`.

The clean-commit z16-to-z32 certificate SHA-256 is `079a6fbbb78aeab29d5e7460815f22208708a307f02572dc956f244433b9bb97`. Every source/material/raw/prior/provenance audit passes. Total Q (`0.3321%`), fixed-probe E (`1.6796%`), and conservative Q (`1.5708%`) pass, but component Q (`2.2751%`) and material-region E (`6.9513%`) fail their unchanged `2%` and `5%` gates. The z-only ladder is terminated, z64 is forbidden, and neither z16 nor z32 is selected. See `FDTDX_Z32_STOP_AND_AU_DESIGN_AUDIT.md`.
