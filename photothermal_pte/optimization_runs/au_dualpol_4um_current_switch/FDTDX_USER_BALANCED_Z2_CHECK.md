# User-balanced full-domain-z factor-2 check

## Outcome

The first refinement around the user-requested baseline is complete.  Both
z2 source-only and exact-binary L500 material cases pass all of their internal
stationarity, energy-closure, material-readback, and provenance gates.  The
cross-mesh comparison nevertheless **fails** five spatial-convergence gates.
The requested 5-nm thin-stack z mesh is therefore not selected, and no
inverse-design or adjoint run is authorized.

This check uses code commit `2ccc9215` and patched FDTDX commit
`6cc0e97252ee0b95de5016e8db1a5b414177efa4`.  The full project suite is
`446 passed, 7 subtests passed`.  Lumerical was not used or modified.

## Refinement contract

x/y is byte-identical to the user baseline.  Every z segment is refined by a
factor of two with all physical interfaces and PML thicknesses fixed:

- SiO2, TaIrTe4, Au: `2.5 nm`
- non-PML air: `25 nm`
- resolved Si exception: `25.375 nm`
- z PML: 16 cells per face at `100 nm`
- grid: `186 x 186 x 300 = 10,378,800` Yee cells
- time: 24 periods, four-period windows, Courant `0.5`, `76,849` steps

CPU placement and exact all-air readback pass before the GPU runs.

## Runtime and internal validity

Ea/Eb were run concurrently on verified-idle physical B200 GPUs 6/7 after
each launcher's compute-process check.  Existing Lumerical jobs on GPUs 0/4
were untouched.

- source total time: `121.661 s` Ea, `121.776 s` Eb
- material total time: `121.958 s` Ea, `121.934 s` Eb
- material cold compile+forward: `94.721 s` Ea, `94.699 s` Eb
- z2 source powers are identical: `1.8834807845313772e-12 W`
- material total Q: `4.542555120497780e-13 W` Ea and
  `7.894118106334004e-13 W` Eb
- every per-case failed-gate list is empty

This is forward-only timing, not an adjoint or optimization-iteration time.

## Baseline-to-z2 comparison

The existing full-z comparison implementation was applied without changing
its gates.  It uses component-Yee volumes, conservative physical-z overlap,
fine-to-coarse complex-field interpolation without extrapolation, and an
identical fixed physical tangential-field probe.

| metric | measured worst case | limit | result |
| --- | ---: | ---: | --- |
| source power relative change | `0.062998%` | `0.5%` | pass |
| Q/closed-flux error | `0.098233%` | `2%` | pass |
| refined-case stationarity E NRMSE | `0.009718%` | `0.5%` | pass |
| total Q relative change | `6.256648%` | `1%` | fail |
| material/component Q max change | `13.478506%` | `2%` | fail |
| fixed-probe complex-E NRMSE | `3.880560%` | `2%` | fail |
| conservative 3-D Q NRMSE | `10.279118%` | `5%` | fail |
| material-region complex-E max NRMSE | `62.937483%` | `5%` | fail |

Per polarization, total-Q change is `2.096749%` for Ea and `6.256648%` for
Eb.  Conservative-Q NRMSE is `5.133554%` and `10.279118%`; material-region
field NRMSE is `50.882152%` and `62.937483%`.

This two-level result is not a final convergence certificate.  It is strong
evidence that the user baseline cannot be treated as z-converged.

## Artifact ledger

z2 source root:

```text
/home/seunghyun200/fdtdx_results/user_balanced_z2_source_2ccc9215/
```

- source pair SHA-256:
  `7e4c736cc8847b7edd5ab7d784cba06c656cc12863c2f4e132104cd0b207cc7b`
- Ea/Eb source report SHA-256:
  `5f1893faf7a158528c52053b024445b611e71cfbc88993b4b30f0877372bf29d`,
  `dda1c841b6375ab2750e539b5f090258f2b30e46d22bd702258c3d23aaf97b63`
- Ea/Eb source raw SHA-256:
  `ee98f1d499ebe6b263573a88c6695a18ad63c586095d4ab1542f589a34c5e932`,
  `ea72d765e087f7395cc7a006f10efe45666a57b7af52bf77fb8321ffae4fc347`

z2 material root:

```text
/home/seunghyun200/fdtdx_results/user_balanced_z2_material_2ccc9215/
```

- Ea/Eb report SHA-256:
  `3a0b86ffc331879361690f7aec178da8954e1f7d76433b1d9d9fa633c164fde1`,
  `6ed571808dc5b3ab59b0ea27517b07307fcad69a045fb97f9aa9f3c86d841d17`
- Ea/Eb raw SHA-256:
  `dd610f1b4b90503f2b82cf89e5368c57fcb714f1ec30a98d6879ebe71693ec5a`,
  `a0ffc021317bc244ae4ab2aa70345133c72565eecfb6ec9b2520ae0d18787b59`

Raw artifacts remain outside Git.

## Next gate

Do not start x/y refinement, thermal/electrical coupling, gray optimization,
or an adjoint.  First create a hash-bound comparison artifact that revalidates
all baseline/z2 report and raw bytes.  Only then consider z4.  Linear measured
scaling suggests roughly four minutes per z4 forward, but it must be timed by
source-only Ea/Eb first and stopped if the actual cost approaches the user's
practical limit.  A z2-to-z4 tail comparison is required before any z level
can be considered for selection.
