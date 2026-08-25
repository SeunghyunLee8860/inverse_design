# FDTDX gradient-detector and runtime report

Date: 2026-08-25 (Asia/Seoul)

Implementation commit: `97da38784cc9fdf8f08ab77d3230657316a4ced4`

This closes the bounded detector/checkpoint tuning gate.  It does not validate
the full 40-period gradient, complete optical/Q/PDE/current objective, or an
optimizer.

## Change and correctness contract

Forward certification still uses every source, temporal, profile, Q, and
closed-flux detector.  The differentiated simulation now has explicit profiles:

- `all`: all 11 validation detectors,
- `production`: only `au_late` and `tairte4_late`,
- `none`: no detector, used only as a field-only resource lower bound.

`fdtdx_parity_gradient_detectors.py` removes a detector from both the placed
`ObjectContainer.object_list` and `ArrayContainer.detector_states`.  It fails
closed on unknown, duplicate, or misaligned object/state names and proves that
the volume index and non-detector object count are unchanged.

On a small real dispersive FDTD scene, removing an unused control detector left
the final-field loss and complete c3 gradient bitwise identical.  The complete
target-folder CPU suite passed `207` tests.

## Exact detector payload

| detector state | bytes |
|---|---:|
| `au_late` | 3,072,000 |
| `tairte4_late` | 24,576,000 |
| two retained production states | 27,648,000 |
| nine removed validation states | 39,750,784 |
| original 11-state total | 67,398,784 |

With sparse regional ADE state, the per-checkpoint payload is:

| profile | checkpoint bytes |
|---|---:|
| all validation detectors | 423,902,660 |
| production detectors | 384,151,876 |
| no detectors | 356,503,876 |

The production profile removes `39,750,784` bytes per checkpoint while retaining
the late Au/TaIrTe4 field observables required to construct material-resolved Q.
The detector-free profile removes another `27,648,000` bytes but is not a
production objective.

## Exact-grid bounded AD-FD results

Both profiles used the exact `186 x 186 x 286` grid.  The scalar probe remains
the field-only Au-slab loss, so these results prove the latent-to-Maxwell path
and resource behavior, not the derivative of the final Q/current objective.

| steps | checkpoints | profile | value-and-grad | centered AD-FD error | XLA peak bytes |
|---:|---:|---|---:|---:|---:|
| 4,096 | 96 | none | 26.2657 s | 1.6373e-5 | 63,818,432,256 |
| 4,096 | 96 | production | 27.4033 s | 1.6373e-5 | 66,596,801,280 |
| 65,536 | 256 | none | 651.9622 s | 1.4082e-3 | 158,853,321,216 |
| 65,536 | 256 | production | 670.4281 s | 1.4082e-3 | 166,056,417,536 |

Every run returned 6,561 finite, nonzero latent gradients and same-sign AD/FD
directionals.  At each horizon, the `none` and `production` NPZ files are
bitwise identical.  The deep result includes the same complete gradient,
primal/plus/minus values, and directionals in both files.

The deep probes used `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`.  Production's largest
single allocation was `160,587,535,616` bytes and its XLA peak was
`166,056,417,536` bytes.  Thus 256 checkpoints are feasible only with an
explicitly enlarged allocator on this 179.06-GiB B200.  Adding another 64
production checkpoint payloads alone would add `24,585,720,064` bytes, already
pushing the largest-buffer estimate past the `181,927,936,000`-byte allocator
limit before other live arrays.  A larger production checkpoint count is not a
safe next experiment.

## Definitive runtime decision

Independent linear extrapolation of the deepest measurements gives:

- no-detector lower bound: `42.47248226 min/polarization`,
- production-detector route: `43.67545625 min/polarization`,
- two-polarization optical AD alone: `87.35091250 min/iteration`.

These are deliberately optimistic extrapolations, not measured full-gradient
times or upper bounds.  The full 256,163-step online schedule can require a
deeper recomputation pattern, the production scalar must actually differentiate
late-window Q, and thermal/electrical solves add work.  Most importantly, even
the detector-free resource lower bound exceeds the user's 30-minute-per-
polarization feasibility gate.

The current checkpointed reverse route is therefore
`BLOCKED_PRODUCTION_RUNTIME`.  Detector-state pruning and checkpoint-count
tuning are closed; do not run the full gradient, complete 16-forward AD-FD
certificate, two-iteration smoke optimization, or LD_MMA.

The next code task is an evidence-driven audit of an alternative reverse path.
FDTDX's existing reversible backward implementation must first be inspected
against the three-pole dispersive ADE update; it must not be enabled merely by
removing a dispersion guard.  Any custom adjoint/reconstruction must prove the
discrete E/H/P recurrence, source and detector adjoints, small-scene forward and
complete coefficient-gradient parity, and bounded exact-grid latent AD-FD before
any production timing claim.

## External raw artifacts

| artifact | file SHA-256 |
|---|---|
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_none_c96.json` | `3b7aa31865b447a27ed2d3a6b9fbf83d6c906da007a372503f46b18ecf3a3c4f` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_none_c96.npz` | `6d642834d4fb49bc34ac4c0a85e47d6837e2042dea7a6ba30ed4b188a4192c09` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_production_c96.json` | `6b0c954219c2491d94c2208a3bf7689724e8dcca9368b13196c0ba60eb1f4439` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_production_c96.npz` | `6d642834d4fb49bc34ac4c0a85e47d6837e2042dea7a6ba30ed4b188a4192c09` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_none_c256_s65536.json` | `98983e7a6431b18efaf8269352630cd4d418bc9abb5b457a6243ed4a8edb66a0` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_none_c256_s65536.npz` | `6bf9375ba58794e2509671360dcf7445279bfc19a5d43bd20285df7abfadacc9` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_production_c256_s65536.json` | `98728249b6feb8a2a5b6ce1ccd0040d35009858391e976ad4bd2f0d55ce90e7d` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_97da3878_Ea_sparse_production_c256_s65536.npz` | `6bf9375ba58794e2509671360dcf7445279bfc19a5d43bd20285df7abfadacc9` |

No raw result is stored in Git.  Lumerical, HEAT, and CHARGE were not called.
