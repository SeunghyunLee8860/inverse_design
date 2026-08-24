# Fresh FDTDX exact-binary material pilot

Status: **EA_EMPTY_AND_FULL_VALIDATED_EB_PENDING_NOT_A_CONVERGENCE_CERTIFICATE**

This is a parallel forensic/rebuild track for the historical FDTDX route. It
does not modify or replace the separate Lumerical work. The purpose of this
pilot is to establish that the pinned FDTDX forward model can propagate an
exact ordinary-Au endpoint, the anisotropic TaIrTe4 stack, and a closed energy
balance before any mesh ladder, adjoint, thermal/electrical solve, or optimizer
is allowed.

## Locked input contract

- source-pair certificate SHA-256:
  `cc86457678ba50becff8ec44408f7f519a8fd3ae44abedc248082eefeee28ee6`
- repository commit used for both solves:
  `08a6f60e2ea499aa76e1ceea323f2473bf4a7411`
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

## Validated E-parallel-a controls

Because `x = crystal b` and `y = crystal a`, `Ea` uses the explicit source
polarization vector `[0, 1, 0]`.

| Metric | exact empty | exact full design window |
|---|---:|---:|
| Au design solid cells / 6400 | 0 | 6400 |
| unscaled Au Q | 0 W | 1.9249669439077155e-14 W |
| unscaled TaIrTe4 Q | 4.696773955206817e-13 W | 4.308509697646610e-14 W |
| unscaled total Q | 4.696773955206817e-13 W | 6.233476641554326e-14 W |
| closed phasor inward power | 4.699883060857446e-13 W | 6.249901719204351e-14 W |
| Q/closed-phasor symmetric relative error | 0.0661528% | 0.262805% |
| Q/closed-TD symmetric relative error | 0.0654843% | 0.255895% |
| maximum previous/late complex-E NRMSE | 3.98749e-5 | 2.47814e-3 |
| previous/late spatial-Q NRMSE | 2.47474e-5 | 1.77276e-3 |
| previous/late total-Q relative change | 3.28968e-6 | 7.64724e-5 |
| absorbed fraction of all-air source reference | 0.249543 | 0.0331190 |
| Maxwell solve runtime | 27.7767 s | 27.8168 s |

All material readback, passivity, exact-binary, finite-value, nonnegative-Q,
field/Q stationarity, TD/phasor agreement, and Q/closed-surface gates passed.
The empty case has exactly zero Au Q. The full case has strictly positive Au Q
and all 6,400 design cells read back at the one ordinary-Au endpoint.

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
```

## What remains blocked

This partial control matrix does not certify polarization parity, time
convergence, z convergence, x/y convergence, PML/domain convergence, a design
shape, an adjoint, thermal/electrical remapping, PTE current, or an optimizer.
The next allowed cases are matching exact-empty and exact-full `Eb` controls,
each in a new immutable output directory. Only after the four-case matrix is
valid may a deliberately chosen exact-binary shape enter a mesh/time ladder.
