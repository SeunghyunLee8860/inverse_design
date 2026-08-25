# FDTDX reversible long-slice report

Date: 2026-08-25 (Asia/Seoul)

Execution commit: `9fdb14d52be68f678d983dbda4da4bb45761499f`

Status: `BLOCKED_SLICE_1024_EB_ADFD`; Ea passes narrowly, Eb does not.

## Purpose and execution

The 4,096-step slice-256 probe passed both polarizations but its reset interval
cannot fit a full 256,163-step run. The next bounded test kept 16 sparse
checkpoints while increasing both horizon and reset interval by four:

- total steps: 16,384 (2.558 optical periods);
- steps per slice: 1,024;
- sparse checkpoints: 16;
- loss/density/direction/gate: unchanged from the 4,096-step probe;
- Ea/Eb ran concurrently on the same separately verified-idle GPU 6/7 UUIDs.

No occupied GPU was touched. Both selected GPUs returned to zero memory after
the test.

## Result

| metric | Ea | Eb |
|---|---:|---:|
| status | PASS | BLOCKED |
| value-and-grad | 85.1284 s | 85.0230 s |
| centered AD-FD error | 0.00410128 | 0.00655637 |
| frozen error gate | <0.005 | <0.005 |
| finite/nonzero latent gradients | 6,561 | 6,561 |
| peak device bytes | 15,665,396,992 | 15,669,545,728 |
| projected one-polarization runtime | 22.1830 min | 22.1555 min |

Every value, gradient, and directional is finite and nonzero, and AD/FD signs
agree. Runtime scaling remains almost exactly linear and below 30 minutes per
polarization. Accuracy is the blocker: Eb exceeds the predeclared `5e-3` gate
by 31.1%, while Ea has only 17.97% relative margin below the gate.

The checkpointed reference path at 16,384 steps previously had centered AD-FD
errors around `1.6e-4..1.8e-4` with the same latent/FD contract. The new error
growth is therefore not accepted as ordinary centered-FD truncation without
additional evidence.

## Decision

Slice 1,024 is not approved for production. A full-horizon slice-1,024 payload
would be 89,482,471,872 bytes and may fit the device, but memory feasibility
cannot override a failed derivative gate.

The next diagnostic keeps the 16,384-step horizon and changes only the reset
interval to slice 512. If Ea/Eb errors fall, inverse reconstruction between
resets is confirmed as the dominant source. Slice 512 itself is not a
full-horizon candidate: its 501 checkpoints would require about 178.6 GB before
other live arrays. The result must instead guide a hybrid recomputation/reset
scheme or a memory-safe intermediate slice search.

## External artifacts

| artifact | SHA-256 |
|---|---|
| Ea JSON | `12e2c8e34dcfe0c80a1b14eecf76ecb3e8f834165429683710ab6916dbd18aed` |
| Ea NPZ | `8fd1a26031d12d331077ebfd79eb9e4c1fce0c4454d53b9795faddb20525ac02` |
| Eb JSON | `b4e38c4e539c18719d301963c341403f0088e6b0d12fe6a13aa6aa813957d921` |
| Eb NPZ | `9c95a66ca91f69269dc342ce961fbbe3f9179a261fcf00c154b12ffef56c6e2f` |

Raw files remain outside Git. No full gradient, late-window Q gradient,
PDE/current solve, optimizer, Lumerical, HEAT, or CHARGE call was made.
