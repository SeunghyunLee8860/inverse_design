# Au sharp-interface P_Q adjoint diagnostic

Status: `BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_UNVALIDATED`

The 25 nm exact-binary Au baseline forward solve passed: `P_Q =
1.551979775376e-15 W`, six-face closure is
`0.138667%`, and the adjoint
auto-shutoff is below `1e-5`. The FieldRegion source round trip is exact and
the forward/adjoint coordinate mismatch is zero.

The numerical AD--FD gate nevertheless fails decisively. The strong central
FD is `-2.904123473676e-17 W/um`, while the candidate boundary result is
`4.079992905639e-12 W/um`. Their signs do not agree
and the magnitude ratio is `1.404897e+05`.

The candidate explicitly included both the bundled v261 tangential-E/normal-D
field-mediated boundary kernel and a moving-domain absorption trace. The
surface quadrature itself converges (`0.434578%`
on its final refinement), so quadrature resolution is not the explanation.
Instead, the continuous pointwise inside-Au loss trace is incompatible with
the discrete conformal-Yee `P_Q` objective at a sharp lossy-metal edge. It is
therefore rejected; no fit, normalization, sign change, or gradient rescaling
is applied.

The alternative density route is also not available: the uniform rho=1 Au
`importnk2` endpoint diverged. Consequently neither current Au representation
has a certified optical gradient, and no Au/TaIrTe4 thermal, electrical, PTE,
or optimization run is permitted from this checkpoint.

The next isolated diagnostic should use a fixed external field observable to
certify the boundary kernel independently of an explicit moving material-loss
integral. A production PTE formulation then needs a solver-consistent
conformal material derivative or a boundary-fitted discretization; the rejected
continuous trace must not be reused.
