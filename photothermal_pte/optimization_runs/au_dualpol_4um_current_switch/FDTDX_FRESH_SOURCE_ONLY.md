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
| incident power before reporting scale | 1.8821465653379166e-12 W | 1.8821465653379166e-12 W |
| previous/late maximum complex-E NRMSE | 5.8801227255494025e-6 | 5.882027606496750e-6 |
| transverse polarization purity | 0.9998024972592621 | 0.9998024972583772 |
| longitudinal energy fraction | 0.026368895323786185 | 0.02636889522792968 |
| beam center `(x, y)` | `(2.17909e-14, 4.99463e-8) m` | `(4.99463e-8, 2.22190e-14) m` |
| second-moment waist `(x, y)` | `(3.940488, 4.146653) um` | `(4.146653, 3.940488) um` |
| maximum waist relative error | 0.03666328628407878 | 0.036663286396297304 |
| closed phasor flux / incident, absolute | 9.324813316339027e-7 | 9.324813316339027e-7 |
| closed time-domain flux / incident, absolute | 6.020645613680110e-7 | 5.817702194477492e-7 |
| Maxwell solve runtime | 27.8232 s | 27.9001 s |

Every per-case gate passed: finite detector state, field stationarity, positive
incident power, transverse purity, longitudinal fraction, beam center, beam
waist, closed phasor flux, and closed time-domain flux. The runs used clean
repository commit `5af31f3836a35f8144964c762ec2156c60c1e23b`, clean pinned
FDTDX commit `f26f84b70a8cceec9b889553955a868624736bf1`, the locked Python/JAX/
CUDA environment, and otherwise idle physical B200 GPU 7.

The promoted pair additionally serializes and compares every PML face,
source startup/profile/vector, and integer placement slice. Electric-field
stationarity uses component-specific Yee dual volumes rather than a primal
cell-volume approximation.

## One common power normalization

`fdtdx_fresh_source_pair.py` re-hashes both JSON reports and raw NPZ files,
requires identical mesh, time, all-air readback, FDTDX source, runtime lock,
source repository commit, and raw array schema, and rejects a relative source
power mismatch above 0.5%. It forbids independent polarization matching.

The clean generator commit is
`5af31f3836a35f8144964c762ec2156c60c1e23b`. The measured Ea/Eb power mismatch
is exactly zero, so the certificate selects:

```text
common unscaled incident power = 1.8821465653379166e-12 W
reporting target power          = 2.85e-4 W
common power scale              = 151422851.57204625
common field-amplitude scale    = 12305.399285356256
```

Power-like quantities such as absorption use the common power scale. Complex
fields use its square root. No downstream code may independently rescale Ea
and Eb to hide source imbalance.

## External immutable artifacts

Raw artifacts remain outside Git:

```text
/home/seunghyun200/fdtdx_results/source_only_Ea_5af31f38_20260824/
  FDTDX_FRESH_SOURCE_ONLY.json
    sha256 559c30ac0b0f4904b2243d05f97b689fb2a07444a8d90432e15f64ec7511954d
  FDTDX_FRESH_SOURCE_ONLY_FIELDS.npz
    sha256 1f73e47df05bab2602ff6426a77c0e3051466b6f2c1659e3094f877160f50a16

/home/seunghyun200/fdtdx_results/source_only_Eb_5af31f38_20260824/
  FDTDX_FRESH_SOURCE_ONLY.json
    sha256 70ec4d5d8942293b10da07e5df650d2c6db65eef8e85e7ce446cc35bb324a93b
  FDTDX_FRESH_SOURCE_ONLY_FIELDS.npz
    sha256 5ab3589df2454ac17bfc2aca1907fbcd2b71ba171a0692e1d1434e9880b0fa4e

/home/seunghyun200/fdtdx_results/source_only_pair_5af31f38_20260824/
  FDTDX_FRESH_SOURCE_ONLY_PAIR.json
    sha256 cc86457678ba50becff8ec44408f7f519a8fd3ae44abedc248082eefeee28ee6
```

Earlier directories are deliberately preserved. The `aaade0e9`
attempt completed Maxwell propagation but postprocessing used the wrong
closed-surface detector-state key and emitted only a failure JSON. The
`6039587b` attempt preserved valid raw fields but used the incorrect
desired-transverse-over-total-field purity definition. The `15304d99` pair
fixed polarization interpretation but used primal instead of component-Yee
dual-volume weighting for stationarity. The `493d833e` pair fixed that metric
but did not yet serialize the complete PML/source/placement provenance that
the final `5af31f38` pair gates. None of the superseded directories should be
overwritten or deleted.

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
