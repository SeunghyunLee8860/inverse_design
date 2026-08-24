# Fresh FDTDX exact-binary material pilot

Status: **VALIDATED_FOUR_CASE_CONTROL_MATRIX_NOT_A_CONVERGENCE_CERTIFICATE**

This is a parallel forensic/rebuild track for the historical FDTDX route. It
does not modify or replace the separate Lumerical work. The purpose of this
pilot is to establish that the pinned FDTDX forward model can propagate an
exact ordinary-Au endpoint, the anisotropic TaIrTe4 stack, and a closed energy
balance before any mesh ladder, adjoint, thermal/electrical solve, or optimizer
is allowed.

## Locked input contract

- source-pair certificate SHA-256:
  `cc86457678ba50becff8ec44408f7f519a8fd3ae44abedc248082eefeee28ee6`
- pilot runner SHA-256 shared by all four solves:
  `c7aee0763c19fa970a7989ce8b031052a9f737f27bf1fce1124de79586e4aa24`
- clean repository commits: Ea cases
  `08a6f60e2ea499aa76e1ceea323f2473bf4a7411`; Eb cases
  `226f70e6f2b9ff387ce02e71cc6811d5019c4855`
- matrix generator commit:
  `4336a54732cbc59e821fc180234845910e2fb9e2`
- pinned FDTDX commit:
  `f26f84b70a8cceec9b889553955a868624736bf1`
- material-contract SHA-256:
  `6f698049dbbaa7f770d4595e9ac75ddca66422880dc60fbeac832db631e7747d`
- anchor grid: `196 x 196 x 160 = 6,146,560` Yee cells
- time: Courant 0.5, 16 periods, four-period source startup, four-period
  previous and late windows

The Au carrier is exact binary. Air uses zero ADE coefficients and ordinary Au
uses the locked finite-time-step ADE endpoint. No continuous density, `rho`
power, optical/thermal/electrical exponent, or per-polarization rescaling is
present. Si and SiO2 inverse permittivity and the component-wise TaIrTe4/Au ADE
coefficients are read back from the realized solver state before propagation.

Absorbed power is evaluated on component-specific electric Yee dual volumes:

```text
Q = 0.5 * omega * eps0 * eta0^2
    * Im(realized discrete susceptibility) * |E_phasor|^2
```

The saved fields and Q arrays are unscaled. The single source-pair reporting
scale is applied only to reporting or later linear downstream physics.

## Validated four-case controls

Because `x = crystal b` and `y = crystal a`, `Ea` uses the explicit source
polarization vector `[0, 1, 0]` and `Eb` uses `[1, 0, 0]`. Both use the same
all-air incident reference and common reporting scale.

| Metric | empty Ea | full Ea | empty Eb | full Eb |
|---|---:|---:|---:|---:|
| Au design solid cells / 6400 | 0 | 6400 | 0 | 6400 |
| unscaled Au Q (W) | 0 | 1.92497e-14 | 0 | 2.11477e-14 |
| unscaled TaIrTe4 Q (W) | 4.69677e-13 | 4.30851e-14 | 8.06713e-13 | 7.59155e-14 |
| unscaled total Q (W) | 4.69677e-13 | 6.23348e-14 | 8.06713e-13 | 9.70632e-14 |
| closed phasor inward power (W) | 4.69988e-13 | 6.24990e-14 | 8.05801e-13 | 9.72650e-14 |
| Q/closed-phasor symmetric error | 0.0662% | 0.2628% | 0.1129% | 0.2075% |
| Q/closed-TD symmetric error | 0.0655% | 0.2559% | 0.1130% | 0.2032% |
| maximum previous/late complex-E NRMSE | 3.987e-5 | 2.478e-3 | 9.656e-6 | 4.490e-3 |
| previous/late spatial-Q NRMSE | 2.475e-5 | 1.773e-3 | 9.043e-6 | 4.363e-3 |
| previous/late total-Q relative change | 3.290e-6 | 7.647e-5 | 5.872e-7 | 1.195e-4 |
| absorbed fraction of all-air reference | 0.249543 | 0.033119 | 0.428613 | 0.051570 |
| Maxwell solve runtime (s) | 27.7767 | 27.8168 | 28.2647 | 27.7335 |

All four cases passed material readback, passivity, exact-binary, finite-value,
nonnegative-Q, field/Q stationarity, TD/phasor agreement, and
Q/closed-surface gates. Both empty cases have exactly zero Au Q. Both full
cases have positive Au Q and all 6,400 design cells at the ordinary-Au
endpoint. The larger empty-case TaIrTe4 absorption for Eb is retained as the
physical x=b/y=a anisotropic response; it is not removed by polarization-wise
renormalization.

`fdtdx_fresh_exact_binary_matrix.py` re-hashes the four reports, four raw NPZ
files, source pair, pilot runner, material contract, and pinned dependency. It
also reloads every raw binary mask and component-Q/dual-volume array,
recomputes the Au and TaIrTe4 powers, and requires exact agreement with the
reports. All 32 top-level matrix gates passed. The repository-commit difference
between Ea and Eb is permitted only because the solver runner SHA and every
physics contract are identical; both repositories were clean.

The source-side plane inside a material case is a net downward-flux diagnostic,
not an incident-power calibration because it contains reflected fields. The
all-air source pair remains the only incident reference.

## External immutable artifacts

Raw artifacts remain outside Git and must not be overwritten:

```text
/home/seunghyun200/fdtdx_results/exact_binary_pilot_empty_Ea_08a6f60e_20260824/
  FDTDX_FRESH_EXACT_BINARY_PILOT.json
    sha256 d41538ab06f949dbf8d46a327c4189e9bebe1fae97d0ba5292ec4c2422badf4c
  FDTDX_FRESH_EXACT_BINARY_PILOT_FIELDS.npz
    sha256 c61c1996f864ceaec9fa2d09f56776130e2875cd971e247606d418a6e0b18f4a

/home/seunghyun200/fdtdx_results/exact_binary_pilot_full_Ea_08a6f60e_20260824/
  FDTDX_FRESH_EXACT_BINARY_PILOT.json
    sha256 81f7c024099577adb9cf258a87a10e6fe7270501c70bb37e070923602ccc9708
  FDTDX_FRESH_EXACT_BINARY_PILOT_FIELDS.npz
    sha256 5e04151405a7d0f565701ce137f4f932a33fc613b5a8102ce2824868c8df29ac

/home/seunghyun200/fdtdx_results/exact_binary_pilot_empty_Eb_226f70e6_20260824/
  FDTDX_FRESH_EXACT_BINARY_PILOT.json
    sha256 3b7e6bfc01c1d216b16bcf92d34f3915c3d126bb86e03ab9687eb6bcaa74aebc
  FDTDX_FRESH_EXACT_BINARY_PILOT_FIELDS.npz
    sha256 6aed5db2a1ca405793f3dc7377f810c737204318bdd13c6546de72237a20c75e

/home/seunghyun200/fdtdx_results/exact_binary_pilot_full_Eb_226f70e6_20260824/
  FDTDX_FRESH_EXACT_BINARY_PILOT.json
    sha256 1eb08ad366ea740aa7e37e9f3930b1e57b3e3d21c19c8c9a292f72d62e9eb5ca
  FDTDX_FRESH_EXACT_BINARY_PILOT_FIELDS.npz
    sha256 934f296016fa9c8629ef21eb058fbc6a87da960ddc46002aa67bac3521ae422e

/home/seunghyun200/fdtdx_results/exact_binary_matrix_4336a547_20260824/
  FDTDX_FRESH_EXACT_BINARY_MATRIX.json
    sha256 06e69f15e292ef29b6515282332b01d1e88c8348cfa965a951d3c1c3e98a431b
```

## What remains blocked

This completed endpoint control matrix does not certify time convergence, z
convergence, x/y convergence, PML/domain convergence, a nontrivial design
shape, an adjoint, thermal/electrical remapping, PTE current, or an optimizer.
The next allowed step is to choose and hash a nontrivial exact-binary reference
mask, then build a fail-closed time/z/x-y/PML/domain convergence hierarchy
around it. Empty/full endpoints remain controls, not a production mesh
certificate.
