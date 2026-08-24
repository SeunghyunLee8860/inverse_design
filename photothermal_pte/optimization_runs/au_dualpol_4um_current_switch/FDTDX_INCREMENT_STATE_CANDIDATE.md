# FDTDX increment-state ADE candidate

The first sections record the CPU-only dispersive-state candidate added at
commit `05d8e9ba`. The candidate has since been promoted into an isolated,
opt-in FDTDX production path and passed a small checkpointed full-FDTD AD-FD
gate; see `FDTDX_INCREMENT_STATE_INTEGRATION.md`. Neither stage certifies a
4-um mesh, authorizes optimization, or touches the separate Lumerical track.

## Root cause and rejected CCPR route

The old FDTDX Lorentz/Drude update stores polarization as

```text
P[n+1] = c1*P[n] + c2*P[n-1] + c3*E[n].
```

At the fine 4-um time steps, the carrier response subtracts values close to
`+1`, `+2`, and `-1`. The z16 Au carrier denominator has a cancellation
condition estimate near `1.66e7`. Repeated float32 updates therefore drift
even though the one-frequency algebraic fit and recurrence-root gates pass.

A one-point CCPR replacement was rejected before implementation. FDTDX maps
CCPR into the same second-order polarization recurrence. Passive/safe Au
candidates retained approximately the same z16 cancellation and could collapse
to a unit root in realized float32 coefficients. A substantially better
conditioned candidate required negative `c4` and became active at high
frequency. Carrier accuracy alone is not sufficient to revive this route.

## Candidate state equations

`fdtdx_fresh_increment_state_precision.py` stores the polarization increment,
`V[n] = P[n] - P[n-1]`, directly:

```text
V[n+1] = A*V[n] - C*P[n] + B*E[n]
P[n+1] = P[n] + V[n+1]
E[n+1] += -inv_eps*sum(V[n+1])
```

The coefficients are

```text
A = (1-gamma*dt/2)/(1+gamma*dt/2)
C = omega0^2*dt^2/(1+gamma*dt/2)
B = K*dt^2/(1+gamma*dt/2).
```

Eliminating `V` gives the same second-order differential-equation
discretization, with `c1=1+A-C`, `c2=-A`, and `c3=B`. The important numerical
difference is that the small resonance coefficient `C` is stored directly
instead of being lost when subtracted from `2` in float32. For a Drude pole,
`C=0`; the unit polarization-integrator root is decoupled from Maxwell and only
the strictly damped increment/current state enters the electric-field update.

## Fixed physical pole policy

The physical pole parameters are fixed before choosing a mesh:

- Au and negative-real TaIrTe4 a use one passive Drude pole fixed by their
  complex 4-um endpoint.
- Positive-real TaIrTe4 b/c use one passive Lorentz pole with
  `omega0/omega=2`; damping and strength are then fixed by the same endpoint.
- There is no per-mesh physical-pole refit. Only finite-`dt` `A/C/B`
  coefficients change with the mesh.
- No gray material law is claimed or authorized by this diagnostic.

The Au pole is exactly anchored to the Ordal 4-um row. Against the independent
Ordal table, its maximum relative epsilon error is `1.1006%` over 3--6 um and
`3.0910%` over 2--8 um; the 2--8-um RMS error is `1.4070%`. This is a
narrowband sanity check, not a replacement for a certified sampled-data
multi-pole fit.

## CPU evidence

The external report is:

```text
/home/seunghyun200/fdtdx_results/l500_full_z_150a7592_20260824/
increment_state_05d8e9ba/FDTDX_FRESH_INCREMENT_STATE_PRECISION.json
```

- report file SHA-256:
  `25933679a32a2bc949b6750ca1fb608b193d35393c72c1d32d2ca29c54091826`
- internal payload SHA-256:
  `d2d5dd338f185eceb5680881b3ad43f0009cbe47ddac83bc81705fae30b4af1c`
- diagnostic script SHA-256:
  `3897a7e4fff751ea7c1471ad9891680fbfd6e4467db00921b8df553dec9a8d81`
- material contract SHA-256:
  `6f698049dbbaa7f770d4595e9ac75ddca66422880dc60fbeac832db631e7747d`
- Ordal table SHA-256:
  `1a15720200262892fe26b5cac1949a1c0040dcbb021f15114a863d27a4515901`
- CPU wall time: `9.25 s`

Every material axis passes at z8, z16, and z32:

| z factor | max carrier error | max float32 late drift | max float32/float64 late difference |
|---:|---:|---:|---:|
| 8 | `2.940e-5` | `6.387e-7` | `1.920e-7` |
| 16 | `4.664e-5` | `1.852e-6` | `2.611e-6` |
| 32 | `5.372e-5` | `2.078e-6` | `1.642e-6` |

At the CPU-candidate checkpoint, the FDTDX-related project suite was `152 passed`.

## Promotion boundary and next work

At commit `05d8e9ba` this result was a solver-free representation candidate.
The isolated fork now closes coefficient generation, actual-JIT long-time
float32 stationarity, production placement/update/source semantics, and one
small driven Lorentz-`B` checkpointed full-FDTD AD-FD control. The remaining
promotion order is:

1. add the corresponding small full-FDTD Drude parameter AD-FD control;
2. define a new exact-binary runner/version and hash all new coefficient and
   source semantics;
3. run one short coarse 3-D exact-binary timing/closure control;
4. only if runtime and closure are practical, generate fresh z8/z16/z32
   source/material pairs;
5. keep continuous-density optimization blocked until a fixed-pole gray law
   and its material-placement Jacobian pass independent AD-FD controls.

The old two-pole artifacts remain negative evidence and cannot be mixed into a
new-law mesh comparison. A future runner requires a new version, a pinned
patched-FDTDX tree hash, new coefficient/readback semantics, and fresh source
pairs. No long z16/z32 pair, finer mesh, thermal/electrical solve, or optimizer
run is currently authorized.

When independent Ea/Eb controls eventually become necessary, recheck GPU
compute-process ownership immediately before launch and use two distinct idle
GPUs concurrently. Never select a GPU carrying another user process. A single
FDTDX solve remains single-GPU; assigning several GPUs to one solve does not
make it faster.


## Actual JAX kernel evidence

The first isolated fork gate is complete. The clean local FDTDX fork commit is `24d0cb2374bf03b6bfdc528b189c69685b74dfee`. It adds only `src/fdtdx/increment_state.py` and five unit tests; production `update_E`, placement, source, and detector paths remain unchanged. The module/test SHA-256 values are `ad01f797b9807fc1db994f4ff41c022078895f8d272379fbfe9ec9980c36d5eb` / `95815699805ea3b4322cfd435c47199d7cbff160693ed5a371a426ade0d5fa96`. The fork tests are `5 passed`; the existing dispersion/initialization regression subset is `106 passed`.

The reproducible git patch is `fdtdx_patches/0001-feat-dispersion-add-isolated-increment-state-ADE-ker.patch`, SHA-256 `df7c8e6c537d8f1a6f5f33bb24c6fef7bebf8f8c16bbf11b06a13654c6e4cc50`. Project preflight code and tests were pushed at inverse-design commit `4269c80a`; the project FDTDX-related suite at that isolated-kernel checkpoint was `156 passed`.

The fork-bound CPU/JAX report is `/home/seunghyun200/fdtdx_results/l500_full_z_150a7592_20260824/increment_state_jax_24d0cb2/FDTDX_FRESH_INCREMENT_STATE_JAX_PREFLIGHT.json`. File SHA-256 is `3bc7fc444765acf8f869765b14ac053eb1355bb3a17010769ace495b485551cc`; payload SHA-256 is `890e150bb325bb61bea45aa9e08877a63a826809d873632619404e43cbf9bfd7`; preflight-script SHA-256 is `565cce1ce1fb41dd2a36fd9ffb177ad8edac92653e15cee2f72a877e73c4231d`. It audited the exact clean fork commit, forced backend `cpu` with x64 enabled for the reference state, and completed all z8/z16/z32 32-period kernels in `3.61 s` total.

The worst actual-JIT float32 late drift is `1.8913e-6` at z32 Au. The worst float32/float64 late difference is `2.6211e-6` at z16 Au, and the worst carrier error remains `5.3718e-5` at z32 Au. Every axis and level passes. This closes compiler/JIT state precision only. The unit AD-FD test differentiates the isolated kernel coefficient; it is not a checkpointed full-FDTD adjoint certificate.

That opt-in integration is complete at clean fork commit
`fc09ce54dc32ea13e27d2af799cdb3771801bf65`. It reuses the existing states and
coefficient arrays, connects placement/update/source/mode/broadband-spectrum
semantics, and rejects CCPR, oriented poles, and the dispersive full-tensor
path. The small driven checkpointed AD-FD symmetric relative error is
`2.5563e-4`; the complete FDTDX unit suite is `2605 passed, 2 skipped, 1
xfailed`. The exported patch and exact validation evidence are in
`FDTDX_INCREMENT_STATE_INTEGRATION.md`. No production-width GPU solve has run.
The next allowed field launch is a short exact-binary timing/closure control,
not a gray optimizer; generic continuous `Device` interpolation still changes
`A/C` with density and remains a material-law blocker.
