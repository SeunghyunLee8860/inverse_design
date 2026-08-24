# Lumerical 4-um Au spatial-Drude carrier checkpoint

Status: `MATERIAL_READBACK_PASSED__R1P2_GPU_BLOCKED__AD_FD_PENDING`

## Result obtained without a Maxwell solve

`22_probe_lumerical_4um_au_np_drude.py` opened only the installed Lumerical
FDTD material/design session. It configured an `Index perturbation` material
with the built-in Drude `np density` conversion and a vacuum base. The unit
electron-density endpoint was fitted to the frozen Ordal Au value at 4 um.

| quantity | value |
|---|---:|
| frozen Au epsilon | `-830.370000 + 127.160000 i` |
| direct Lumerical epsilon readback | `-830.370066 + 127.160026 i` |
| Lumerical FDTD-fit epsilon readback | `-830.370066 + 127.160026 i` |
| relative complex-epsilon error | `8.4914e-8` |
| electron density at `f_Au=1` | `5.9283726e22 cm^-3` |
| electron mobility | `24.418819 cm^2/(V s)` |
| plasma frequency | `1.3735968e16 rad/s` |
| damping frequency | `7.2027236e13 rad/s` |

The analytic material law is

```text
epsilon(f_Au, omega)
  = 1 - f_Au * omega_p^2 / (omega^2 + i*gamma*omega),
0 <= f_Au <= 1.
```

It is causal and passive for the full relaxation interval, equals vacuum at
`f_Au=0`, and equals the frozen dispersive Au endpoint at `f_Au=1`. At the
single target frequency it is equivalent to linear epsilon interpolation,
but its time-domain implementation is a Drude pole rather than a constant
gray `importnk` value.

This is the first tested Lumerical representation in this campaign that has
all three required *material-level* properties: a spatial scalar carrier,
causal loss, and an accurate exact-Au endpoint. It has not yet passed a field
solve or an adjoint test.

## Why the installed B200 route is still blocked

The installed product is `2026 R1.2`, build `4522`. Its `fdtd-engine` contains
the explicit rejection:

```text
GPU simulation does not support ... grid attribute types of ... 'np density'.
```

Ansys added np-density grid-attribute support to GPU FDTD in **2026 R1.3**.
Therefore the current R1.2 installation cannot execute this carrier on the
user-required B200. This is a version/capability blocker, not a HEAT or CHARGE
license issue. No HEAT/CHARGE solver participates in this route.

Official references:

- [2026 R1.3 release notes: np density in FDTD GPU](https://optics.ansys.com/hc/en-us/articles/53916763140499-2026-R1-3-Release-Notes)
- [np density and temperature index perturbation object](https://optics.ansys.com/hc/en-us/articles/360034901753-np-Density-and-Temperature-Index-Perturbation-Simulation-object)

## How this supports inverse design

The optimizer does not send a binary mask at every iteration. It sends the
one filtered/projected `f_Au` field as electron density

```text
n_e(x,y) = f_Au(x,y) * 5.9283726e22 cm^-3.
```

The same `f_Au` array is passed to the custom CUDA thermal and electrical
material maps. Beta continuation drives `f_Au` toward 0/1. Final candidates
are thresholded, repaired for 500-nm solid/void DFM, and reevaluated in
Lumerical with exact ordinary dispersive Au geometry rather than the carrier.

This solves the conceptual conflict between topology relaxation and exact
binary promotion. It does **not** yet solve the derivative. The production
gradient requires a custom fixed-grid optical contraction for the Drude pole
and the direct Au-loss term, followed by central same-step AD-FD. Bundled
LumOpt's real-index topology gradient is not used as a substitute.

## Next fail-closed sequence

1. Install Lumerical 2026 R1.3 or newer on the B200 host and record its exact
   `VERSION` and GPU engine log.
2. Repeat this no-solve material readback in that installation.
3. Run uniform `f_Au=0`, `0.5`, and `1` GPU forward stability/readback/Q
   controls. Cross-check `f_Au=1` against an ordinary exact dispersive-Au
   object under the identical stack, source, mesh, and time window.
4. Run a nonuniform fixed-grid carrier control and validate the optical
   material gradient against independently re-solved central FD over at least
   three step sizes and multiple directions.
5. Only then connect the optical gradient to the custom CUDA thermal and
   electrical adjoints and validate the full latent-variable PTE gradient.

The material probe can be reproduced with:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  22_probe_lumerical_4um_au_np_drude.py
```

It deliberately reports `maxwell_solve_run=false`, `gpu_engine_acquired=false`,
and `heat_or_charge_license_assumed=false`.
