# Paper-IR offline Q, thermal, and remap controls

Status: `PARTIAL_OFFLINE_PAPER_IR_VALIDATION_BLOCKED_PLANAR_Q_AND_FIG3HI`

No new FDTD solve was run. The saved 1.2/4 ps artifacts were read only.
Thermal FVM calculations used the analytic source. No PTE, adjoint,
gradient, or optimization calculation ran.

## Saved 1.2 ps versus 4 ps Q

The observable-Q subgate passes for diagnostic heat-source use:

- P_Q: `8.701460132991e-17` to
  `8.701470836178e-17 W`
- relative P_Q change: `0.000123%`
- normalized spatial-Q NRMSE:
  `0.000738%`
- spatial Pearson correlation:
  `0.999999999965`
- centroid shift: `2.878632e-12 m`
- hotspot shift: `0.000000e+00 m`

The FDTD gate remains failed: final auto-shutoff is
`1.810760e-05` at 1.2 ps and `1.809820e-05` at
4 ps, both above `1.0e-05`. Observable convergence does not
promote this artifact to production Q.

## Paper-like Figure 3F/G thermal control

The source is the analytic 11-um Gaussian--Beer--Lambert law on a 130-nm
TaIrTe4 y<=x half-plane. The thermal model is the paper-reduced Robin model:
bottom `G=7.37e6 W/(m2 K)`, top air `G=1 W/(m2 K)`, insulating lateral
material edge, and paper anisotropic kappa.

At 200/100/50 nm, the robust exact-edge x-gradient b/a ratios are
`1.440200`,
`1.440259`, and
`1.440266`. Thus the requested
`|grad T|_b > |grad T|_a` trend is reproduced. The 100-to-50 nm robust-x
changes are `0.454%`
and `0.454%`.

This is not a blanket local-maximum convergence claim. Raw max-dT/dx changes
by `8.979%` and
`9.138%`, while Tmax
changes by `1.306%` and
`1.304%`. Those diagnostics remain
unresolved.

## Yee-like remap control

The exact cut-cell analytic source was compared with the same law sampled on
a 33.898-nm Yee-like Cartesian layout and passed through the current
conservative remap. This was not a Maxwell solve and is not claimed to be the
exact v261 Yee mesh.

- worst Q_T NRMSE:
  `0.494%`
- worst T-field NRMSE: `0.040%`
- worst gradient-field NRMSE:
  `0.334%`
- worst primary-metric change:
  `0.299%`
- worst raw-cell peak change:
  `1.891%`
  (diagnostic only)

No clipping, smoothing, gain, global rescaling, tiling, or source deletion
was used. The earlier centre-sampled diagonal control was wrong because a
cell cut by y=x was filled completely; the published control uses the exact
half-cell measure.

## Three-source decomposition and Figure 3H/I

The required edge-free planar TaIrTe4-stack Q artifact is absent.
The saved empty-stack case contains no TaIrTe4; the finite-centre case is a
digitized Device-A polygon with edges. Neither is relabeled as planar. The
saved straight-edge Q is also a legacy `epsilon_c=16` diagnostic with
exactly zero Qz, not the production `epsilon_c=epsilon_b` material closure.

Therefore the analytic/planar/edge decomposition is
`BLOCKED_PLANAR_STACK_Q_ARTIFACT_UNAVAILABLE`. The available distributions are plotted only for
provenance; they do not establish the requested causal decomposition.
Following the approved order, Figure 3H/I was not started after this
blocker. Doing so requires either an explicitly approved new planar-stack
FDTD artifact or a revised order/contract.
