# Full-SiO2 Q/flux offline audit

Status: `DIAGNOSED_FULL_SIO2_Q_FLUX_CANCELLATION_UNRESOLVED`

No new FDTD, thermal, PTE, adjoint, or optimization run was made.  The saved
5-nm-oxide FSP was reopened read-only to compare four quadrature paths and
Lumerical's own `Pabs_total` against the same six-face balance.

## Face-flux cancellation

- `P_six = 1.277063833821e-11 W`.
- `sum(abs(P_face))/abs(P_six) = 7.885903844`.
- z-face contribution to that factor: `7.885829401`.
- lateral-face contribution: `7.444246092e-05`.

| Q path | power (W) | closure versus P_six |
|---|---:|---:|
| common_trapezoid_W | 1.261110455146e-11 | 1.249223277% |
| common_bounded_W | 1.261110455146e-11 | 1.249223277% |
| native_trapezoid_W | 1.261111355884e-11 | 1.249152745% |
| native_bounded_W | 1.261111540521e-11 | 1.249138287% |
| Lumerical_Pabs_total_W | 1.261111530072e-11 | 1.249139105% |


The requested Q bounds and `finite_pabs_adv` object readback agree exactly,
but they do **not** match the independently realized six-face bounds within
`1e-15 m`.  The x/y maximum mismatch is `50 nm`: the Q object spans
`[-27.05, 27.05] um`, whereas the realized face coordinates span
`[-27.00, 27.10] um`.  The z bounds agree to roundoff.  This mismatch is now
reported fail-closed and must not be called a matched control volume.

The four lateral face powers together are only `7.444e-5` of `|P_six|`, so
this 50-nm lateral displacement cannot explain the 1.249% closure.  The
balance instead subtracts `+4.39677e-11 W` at z-min and `-5.67393e-11 W` at
z-max; their absolute sum is `7.88583` times the small net flux.
The maximum independently read E/index component-coordinate mismatch is
`6.776e-21 m`.

Every old volume-Q path retains approximately the same 1.249% closure.
Changing the Python quadrature or oxide z mesh is therefore not a justified
fix.  The requested one-run follow-up snapped Pabs and all six faces to native
mesh planes and tightened auto-shutoff to `1e-6`.  It exposed a stronger
problem: the old `1e-5` case stopped before a late-time field rise, while the
strict run later diverged at `1.62632 ps`.  The log does not by itself assign
that rise to the source or instability.  No final strict Q/flux exists.
See `FULL_SIO2_STRICT_TIME_REPORT.md` and its time-trace plot.
