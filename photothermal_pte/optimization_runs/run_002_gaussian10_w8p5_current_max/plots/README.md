# Plots

The source-only target-plane audit, design-domain contract, uniform
complex-material representation comparison, production-candidate geometry,
native component-Q forward diagnostics, and production component-Yee mapping
tests are present. The material-intersection power partition is shown
separately from the full optical control-volume P_Q, and the deposited 3D
thermal source has its own lateral/depth diagnostic. Optimization-history
plots remain absent because
optimization is still fail-closed.

`production_combined_adfd_smoke.png` records the first production
physical-density full-chain directional smoke, gate margins, and the
forward/adjoint mesh-parity fix. It is not an optimization-history plot.

`production_design_window_selection.png` records the immutable combined-gradient
L1 coverage of the reviewed candidates and the promoted centered 18.6 µm
window. It is an offline window-selection certificate, not an optimizer plot.

`production_finite_filter_projection.png` shows the latent, finite filtered,
and projected fields, a boundary impulse with no opposite-edge wrap, the
mapping-only FD step sweep, and JVP/VJP dot errors.

`selected_production_optical_chain.png` summarizes the realized selected-grid
mesh, native component absorbed powers, and component-J mapping/transpose
errors from the actual 18.6 µm, 373×373 optical environment.
