# User-balanced z4 endpoint and FDTDX feasibility stop

## Decision

The user-balanced full-domain-z ladder has reached z4 and remains blocked.
The z2-to-z4 tail improves substantially but still fails total-Q,
material/component-Q, and material-region complex-field gates.  No z mesh is
selected.  Do not run the FDTDX optimizer, adjoint timing, x/y convergence,
or downstream thermal/electrical coupling from these optical fields.

The z4 code is commit `170eb036`; the byte-bound z2-to-z4 certificate code is
commit `e538b411`.  The complete project suite is
`452 passed, 7 subtests passed`.  Lumerical was not used or modified.

## z4 contract and measured cost

The x/y mesh remains the user baseline.  Every z segment is four times finer
than the baseline while every physical interface and PML thickness remains
fixed:

- grid: `186 x 186 x 600 = 20,757,600` Yee cells
- Au, TaIrTe4, SiO2 z pitch: `1.25 nm`
- non-PML air z pitch: `12.5 nm`
- resolved-Si pitch: `12.6875 nm`
- z PML: 32 cells per face at `50 nm`
- time: 24 periods, four-period windows, Courant `0.5`
- dt: `2.0844278871579257e-18 s`
- time steps: `153,626`

Ea/Eb source and material cases ran concurrently on physical B200 GPUs 6/7,
each selected only after an empty compute-process check.  No other user's GPU
process was touched.

| case | Ea total | Eb total | parallel wall time |
| --- | ---: | ---: | ---: |
| z4 source | `321.603 s` | `320.866 s` | about `322 s` |
| z4 material | `325.252 s` | `325.223 s` | about `325 s` |

Material compile+forward is `290.075 s` Ea and `289.428 s` Eb.  Peak JAX
bytes-in-use is about `11.14 GB` per case.  These are forward-only timings,
not adjoint or optimization-iteration timings.

Both source and material cases pass all internal gates.  The source mismatch
is `1.1511064e-7`.  Material total Q is
`4.5126754620296493e-13 W` Ea and `7.757154110278576e-13 W` Eb.  Maximum
stationarity complex-E NRMSE is `1.0935e-5` Ea and `8.9243e-5` Eb; maximum
Q/closed-flux error is below `1.93e-4`.

## z2-to-z4 tail result

The byte-bound tail certificate revalidates both source pairs, all four
material reports, all four raw NPZ files, and each historical runner blob.
Every artifact gate passes.  Spatial convergence does not.

| metric | measured worst case | limit | result |
| --- | ---: | ---: | --- |
| source power relative change | `0.014654%` | `0.5%` | pass |
| Q/closed-flux error | `0.019693%` | `2%` | pass |
| refined-case stationarity E NRMSE | `0.008924%` | `0.5%` | pass |
| total Q relative change | `1.735013%` | `1%` | fail |
| material/component Q max change | `3.536004%` | `2%` | fail |
| fixed-probe complex-E NRMSE | `1.970194%` | `2%` | pass |
| conservative 3-D Q NRMSE | `2.672352%` | `5%` | pass |
| material-region complex-E max NRMSE | `14.944994%` | `5%` | fail |

Per polarization, total-Q change is `0.657772%` Ea and `1.735013%` Eb.
Material-region field NRMSE is `12.356116%` Ea and `14.944994%` Eb.  The
tail is trending toward convergence, but z2 and z4 both remain unselected.

## Artifact ledger

z4 source root:

```text
/home/seunghyun200/fdtdx_results/user_balanced_z4_source_170eb036/
```

- source-pair SHA-256:
  `bcd86d61b52a14e448a5b5105b55473e2ccfc733fc0e00a5a88b8b824794ad80`
- Ea/Eb source reports:
  `796b64901d40387128efcf6a1727699cfde4735653ca5c55f5351440aab48ab8`,
  `6544a528f16e4f4813b8d179467a52d7c14d655cf5af9a050a5cd87bb79aeda6`
- Ea/Eb source raw NPZ:
  `251f2eafc2706a09ea311e1b0add544208fcba6b343a0c7a2098b4a72cacbcd9`,
  `b4cb02e24acf32c88bf1fe62fef5e3eb607d8fa45edecb718462203e1cd2facf`

z4 material root:

```text
/home/seunghyun200/fdtdx_results/user_balanced_z4_material_170eb036/
```

- Ea/Eb material reports:
  `5b1ed6fb7392ac33359aa343aa046e8ff81441e36b24e22aa3ba56eb912e527a`,
  `fbdc1c3dc01b8835dea4e6bcf1138500786297e5d34227eda6689c2c4282a8b7`
- Ea/Eb material raw NPZ:
  `21ad2d0b2fa147883f4019ca492e88feb4a7926696bb9d1e244c36f753328ea5`,
  `0c269547a84895db0425ed9b16a2f22e5192c531a834280afff614c872e91922`

Formal certificates:

- z1-to-z2:
  `/home/seunghyun200/fdtdx_results/user_balanced_z_certificate_92f5828c/`
  SHA-256
  `560ff77c00247720a7ac7277fe4df91c5c95de3d316c5f83af97780756b921de`
- z2-to-z4:
  `/home/seunghyun200/fdtdx_results/user_balanced_z_tail_certificate_e538b411/`
  SHA-256
  `b91219355b04cf747f0c20ad213972021d6cae3a7d260d482abf8b1ecdcb6b8c`

Raw artifacts remain outside Git.

## Why z8 is not launched

Measured compile+forward scaling is approximately `35 s -> 95 s -> 290 s`
for z1, z2, and z4.  Extrapolating the latest ratio puts z8 near 15 minutes
per forward.  This is an inference, not a measured z8 runtime.  A source plus
material diagnostic campaign would take roughly 30 minutes wall time even
with Ea/Eb parallelized.  A dual-polarization forward-plus-adjoint iteration
would also be around or above the user's 30-minute impracticality threshold,
before thermal/electrical work.

Therefore z8 is not justified for an FDTDX inverse-design route.  The honest
FDTDX conclusion is: the requested baseline is fast but not z-converged, while
the refinement apparently needed to approach convergence makes optimization
impractical.  Preserve this track as forensic evidence and let the separately
owned Lumerical effort address Maxwell feasibility; do not silently restart a
historical FDTDX optimizer or relax the convergence gates.
