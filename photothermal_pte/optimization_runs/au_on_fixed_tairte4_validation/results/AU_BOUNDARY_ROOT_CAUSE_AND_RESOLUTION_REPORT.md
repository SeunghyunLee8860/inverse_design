# Au optical-boundary root cause and resolution

Status: `BLOCKED_AU_BOUNDARY_ADJOINT_UNRESOLVED_AFTER_SMOOTH_CONTROL`

## Conclusion

The failing gate is not a PML, GPU, source-normalization, Yee-coordinate, or
thermal-boundary problem. It is localized to the exact high-contrast lossy-Au
interface on the conformal Yee mesh. The original rectangular control had a converged central
FD plateau (`0.153585%`) but its
continuous boundary quadrature changed by
`38.417161%` and missed the
strong FD by `6.769092%`.

At 801 vertical-face samples, the two `y=+-10 um` endpoints supplied
`83.718282%` of the stored
tangential-E proxy. Removing those endpoints leaves a smooth-face interior
whose 201-to-6401 change is only
`0.00462274%`.
Moving the y corners out to `+-18 um` did not solve the full 3D problem: the
extruded film still has top/bottom rims, and the full y-z surface rule changed
by `19.752223%`.

The independent solver-discrete conformal `d-epsilon` route was also tested.
All component coordinates match the electric-field grid to
`6.776264e-21 m`,
but its final step change is
`100.430180%` and its strong FD error is
`68.128315%`. Therefore
coordinate repair alone does not make a sharp metal boundary differentiable
on this conformal mesh.

## Controlled remedy

The replacement control uses exact binary scalar Au (`n=12.1+69.2i`) with a
smooth closed in-plane ellipse represented by 512 counter-clockwise vertices.
No gray Au, clipping, fitting, normalization, or gradient rescaling is used.
The boundary is integrated with endpoint-free Gauss-Legendre nodes, and an
analytic geometry test independently verifies that its normal shape velocity
recovers `dA/da`.

For this control, the FD step change is
`0.336608%`, the final quadrature
change is `1.325295%`, and
the strong-direction AD-FD error is
`108.687865%`.
The result is `FAILED_AU_SMOOTH_ELLIPSE_BOUNDARY_KERNEL_ADFD`. The AD sign is opposite to the converged
FD direction. Thus the smooth control disproves the narrower hypothesis that
the failure is caused only by in-plane corners. Sharp corners amplify the
boundary trace, but removing them does not make the continuous interface
kernel a solver-consistent discrete derivative.

No Au optical shape derivative is promoted by this checkpoint. The same five
forward solves give total-`P_Q` FD derivatives of
`-4.978400515013e-18` and `-6.433856000594e-18 W/um`; they change by
`22.621823%`, while the strong perturbation changes baseline
`P_Q` by only `0.020656%`. This is not a usable direct-
absorption derivative certificate. Production Au PTE optimization is still
prohibited because the direct moving-Au spatial absorption contribution has
not passed AD-FD. No thermal, electrical, PTE, or optimization solve is
included in this report.
