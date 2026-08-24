# Fresh FDTDX source-only anchor certificate

Status: **VALIDATED_SOURCE_ONLY_ANCHOR_PAIR_NOT_A_CONVERGENCE_CERTIFICATE**

This document records the first real field solves on the fresh, pinned FDTDX
route. Both solves are all-air source checks on the already validated anchor
placement. The detector volumes named `au` and `tairte4` contain air in these
runs; no Au, TaIrTe4, SiO2, or Si response is certified here.

## Locked interpretation

The coordinate contract is `x = crystal b` and `y = crystal a`. Therefore:

- `Ea` is a y-directed transverse source and its desired component is `Ey`;
- `Eb` is an x-directed transverse source and its desired component is `Ex`.

A focused Gaussian beam also has a physical longitudinal `Ez` component.
Transverse purity is consequently defined as desired transverse energy divided
by `Ex + Ey`; `Ez / (Ex + Ey + Ez)` is recorded and gated separately. Counting
`Ez` as cross-polarization was the bug in the first completed Ea evaluation,
not a source defect.

The anchor has `196 x 196 x 160 = 6,146,560` Yee cells. Its design and outer
x/y pitches are 100 nm, lateral PML pitch is 125 nm, and full-domain z factor
is 4. The time contract is Courant 0.5, 16 carrier periods, a four-period
source startup, a four-period previous window, and a four-period late window:
`dt = 8.318327221205701e-18 s` and 25,664 steps. These are anchor values only;
they have not passed any mesh or time ladder.

## Validated results

| Metric | Ea | Eb |
|---|---:|---:|
| incident power before reporting scale | 1.882146782178351e-12 W | 1.882146782178351e-12 W |
| previous/late maximum complex-E NRMSE | 6.30951807062889e-6 | 6.326310725189772e-6 |
| transverse polarization purity | 0.9998024972593894 | 0.9998024972623545 |
| longitudinal energy fraction | 0.02636889530213477 | 0.02636889563614616 |
| beam center `(x, y)` | `(5.9953e-15, 4.99463e-8) m` | `(4.99463e-8, 1.4426e-14) m` |
| second-moment waist `(x, y)` | `(3.940488, 4.146653) um` | `(4.146653, 3.940488) um` |
| maximum waist relative error | 0.03666328843043274 | 0.03666328987889042 |
| closed phasor flux / incident, absolute | 8.477101738563467e-7 | 8.477101738563467e-7 |
| closed time-domain flux / incident, absolute | 5.817701524225264e-7 | 5.682406293250666e-7 |
| Maxwell solve runtime | 27.7933 s | 27.7288 s |

Every per-case gate passed: finite detector state, field stationarity, positive
incident power, transverse purity, longitudinal fraction, beam center, beam
waist, closed phasor flux, and closed time-domain flux. The runs used clean
repository commit `15304d99d601ef15520c1ade8e73ddc4d281d9ac`, clean pinned
FDTDX commit `f26f84b70a8cceec9b889553955a868624736bf1`, the locked Python/JAX/
CUDA environment, and otherwise idle physical B200 GPU 7.

## One common power normalization

`fdtdx_fresh_source_pair.py` re-hashes both JSON reports and raw NPZ files,
requires identical mesh, time, all-air readback, FDTDX source, runtime lock,
source repository commit, and raw array schema, and rejects a relative source
power mismatch above 0.5%. It forbids independent polarization matching.

The clean generator commit is
`83835cdcb380a9d4f8b67d255d742f5111bd7201`. The measured Ea/Eb power mismatch
is exactly zero, so the certificate selects:

```text
common unscaled incident power = 1.882146782178351e-12 W
reporting target power          = 2.85e-4 W
common power scale              = 151422834.12675598
common field-amplitude scale    = 12305.398576509255
```

Power-like quantities such as absorption use the common power scale. Complex
fields use its square root. No downstream code may independently rescale Ea
and Eb to hide source imbalance.

## External immutable artifacts

Raw artifacts remain outside Git:

```text
/home/seunghyun200/fdtdx_results/source_only_Ea_15304d99_20260824/
  FDTDX_FRESH_SOURCE_ONLY.json
    sha256 31936cd062e753f27bd3e448e3d1263fd96b397c54612aca621b98a2ead2a538
  FDTDX_FRESH_SOURCE_ONLY_FIELDS.npz
    sha256 8d13da05cb8853c3cfc0896f62ebec49185f9b6a02fbe3cc9c827cb380c53ef4

/home/seunghyun200/fdtdx_results/source_only_Eb_15304d99_20260824/
  FDTDX_FRESH_SOURCE_ONLY.json
    sha256 6316dd7c04cecd61eb3e83de7fc5680a20a7051a99759c5a6cce448ad093a45a
  FDTDX_FRESH_SOURCE_ONLY_FIELDS.npz
    sha256 29f8b1e96e8ff9c7360893318a42db00c54d119527bc5e4d2274f3299dcc65d5

/home/seunghyun200/fdtdx_results/source_only_pair_83835cdc_20260824/
  FDTDX_FRESH_SOURCE_ONLY_PAIR.json
    sha256 d5216dda12e0e1450053ece6bda86ac55f3523d462c3ea0b344fc7fced2cda30
```

Two earlier Ea directories are deliberately preserved. The `aaade0e9`
attempt completed Maxwell propagation but postprocessing used the wrong
closed-surface detector-state key and emitted only a failure JSON. The
`6039587b` attempt preserved valid raw fields but used the incorrect
desired-transverse-over-total-field purity definition. Neither directory is
the promoted source pair and neither should be overwritten or deleted.

## What this does not authorize

This certificate does not validate any material response, Au ADE stability,
absorbed-power/closed-flux closure in metal, scattered-field PML behavior,
z or x/y mesh convergence, time convergence, thermal or electrical meshes,
PTE current sign, the open physical-device contract, or an optimizer.

The next allowed solve is a fixed exact-binary reference pilot using this
source contract as an input. Start with matching exact-empty and exact-full
ordinary-Au controls for both polarizations, preserve unscaled fields/Q, apply
the one common normalization only in reporting/downstream linear physics, and
gate material readback, stationarity, nonnegative component Q, and Q versus
closed-surface flux before launching any convergence ladder.
