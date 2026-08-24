# FDTDX increment-state production-path integration

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
