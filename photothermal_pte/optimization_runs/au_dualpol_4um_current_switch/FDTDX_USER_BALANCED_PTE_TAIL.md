# User-balanced z2-to-z4 downstream PTE tail

## Decision

The byte-bound frozen-Q thermal/electrical comparison is valid, but its
predeclared aggregate gate is blocked.  Ea passes every downstream metric.
Eb passes source, temperature-field, gradient, current-density, signed-current,
and sign-stability gates, but its TaIrTe4 peak-temperature change is `2.0781%`
against the fixed `2.0%` limit.  The limit was not relaxed after seeing the
result.

Consequently, z2 (`2.5 nm` thin-stack z pitch) is numerically stable to within
`2.32%` for the PTE current diagnostic, but it is **not** selected as a strict
converged optical or production multiphysics mesh.  z8 is still excluded by
the measured FDTDX runtime scaling and the user's per-iteration feasibility
constraint.

The current exact-binary L500 geometry also fails the intended sign-switching
behavior in this diagnostic: Ea and Eb currents are both positive at both
optical z levels.  This is not an actual-electrode prediction; the electrical
operator uses the existing floating-Au model with `psi=0/1` on the left/right
flake edges.

Lumerical was not used, launched, modified, or interpreted.

## What was compared

No Maxwell solve was rerun.  The certified z2 and z4 exact-binary native Yee
Q arrays were loaded by byte hash, separately for Ea and Eb.  Each was:

1. scaled only by its source-pair common incident-power normalization;
2. conservatively remapped to the same thermal mesh;
3. solved with the same thermal material/interface/boundary assumptions; and
4. passed through the same 100-nm floating-Au electrical weighting operator.

The common downstream mesh/domain is:

- thermal shape: `548 x 548 x 72 = 21,622,272` unknowns;
- thermal x/y refinement factor: `2` over the prior explicit grid;
- thermal z refinement factor: `2`;
- domain: `+/-48 um` laterally, `30 um` Si depth, `3 um` top air;
- comparison TaIrTe4 temperature/current grid: `160 x 160`, `100 nm` pitch;
- electrical terminals: virtual left/right flake-edge weighting boundaries;
- actual metal electrodes present: false;
- electrical mesh converged: false.

The prior thermal-domain-size certificate revalidates by SHA-256, but the
device-specific thermal boundary/interface/material uncertainty remains open.
In particular, this result does not undo the previously measured dominant
TaIrTe4-SiO2 contact-conductance sensitivity.

## Measured result

| metric | Ea z2-to-z4 | Eb z2-to-z4 | limit | result |
| --- | ---: | ---: | ---: | --- |
| mapped absorbed power | `0.6723%` | `1.7494%` | `2%` | pass/pass |
| mapped source x-y NRMSE | `0.6921%` | `1.9465%` | `5%` | pass/pass |
| Ta maximum temperature | `0.6543%` | `2.0781%` | `2%` | pass/**fail** |
| Ta mean temperature | `0.6100%` | `1.6758%` | `2%` | pass/pass |
| Ta temperature-field NRMSE | `0.6075%` | `1.7093%` | `2%` | pass/pass |
| Ta gradient L2 magnitude | `0.6212%` | `1.6162%` | `5%` | pass/pass |
| Ta gradient-vector NRMSE | `0.6441%` | `1.9140%` | `5%` | pass/pass |
| signed PTE current | `0.7195%` | `2.3198%` | `5%` | pass/pass |
| PTE current-density NRMSE | `0.6350%` | `1.9319%` | `5%` | pass/pass |

The electrical weighting field is identical to roundoff between z2 and z4,
as required for an optical-Q-only comparison.  Thermal residuals are below
`9.6e-10`, energy-balance errors below `1.1e-11`, electrical explicit
residuals below `8.9e-11`, and current-density integration reproduces the
reported current.

| level | Ea current | Eb current | opposite signs? |
| --- | ---: | ---: | --- |
| z2 (`2.5 nm`) | `+6.11455 nA` | `+6.36632 nA` | no |
| z4 (`1.25 nm`) | `+6.07055 nA` | `+6.21863 nA` | no |

Each case took `21.6--23.2 s` total.  Ea/Eb ran concurrently on physical B200
GPUs 6/7 after an empty compute-process check, so each two-polarization pair
took about 23 seconds wall time.  Other users' Lumerical processes on GPUs
0/1/2/4 were not touched.

## Artifact ledger

Runner/certificate code commit: `101ad8ac`.

Case root:

```text
/home/seunghyun200/fdtdx_results/user_balanced_pte_tail_101ad8ac/
```

Report SHA-256 values:

- z2 Ea: `ac05a38c7b73919516d8125a8eb6b16e53c6453281a5814a59f0d68aed0ca55a`
- z2 Eb: `161244669ba8495897f695a1d9cc3a80740e5b4ef9f74aec2d8f150d8f564cd4`
- z4 Ea: `da19cd5c04d5a75a4082fd0742368b3c84679075e23c5ec4e343c6e243106953`
- z4 Eb: `c89c7aff75422e8551ce1f2be9b7f69196dd65dc0f20d970de0fd8bc4ecfe29c`

Raw downstream NPZ SHA-256 values:

- z2 Ea: `d4fde018cf22276f3795cead71d197d5bb0e9974a7bc8e6c6e40bb83da1ff662`
- z2 Eb: `78f62ad92488c5b0684a64bc03b3f9d20fa39ecbbdedf312814935116b64385f`
- z4 Ea: `99a5b56eed5aac11f0609a9d01d7e870825c8cc829d8859c39a3200cdeec8958`
- z4 Eb: `e5e14a346731fe920866230559ae626f394dd136c027062d8f672497ff242bc9`

Certificate:

```text
/home/seunghyun200/fdtdx_results/user_balanced_pte_tail_certificate_101ad8ac/FDTDX_USER_BALANCED_PTE_TAIL_CERTIFICATE.json
```

Certificate SHA-256:
`4c2215734cb394d8b338eba3bbdcc7f21b8f79fbe55b543396d824919e5b1002`.
All four case reports/raw files, the blocked optical tail certificate, runner
commit, common downstream mesh, and clean-repository provenance revalidate.
Raw JSON/NPZ artifacts remain outside Git.

## Consequence and next action

- Do not launch z8, the FDTDX adjoint, or the historical optimizer.
- Do not describe z2 as strictly converged; describe it only as a downstream
  current/gradient-stable diagnostic whose aggregate certificate is blocked by
  the Eb peak-temperature gate.
- Do not use the same-sign current values as experimental predictions because
  actual electrodes/contact geometry and electrical mesh convergence are open.
- Continue the code/paper audit at the gray-Au design law.  One projected
  geometry must feed optical, thermal, and electrical physics consistently;
  the historical optical `rho^3` / thermal/electrical mismatched laws cannot be
  resumed without a paper-supported, endpoint-consistent constitutive model
  and AD-FD validation.
