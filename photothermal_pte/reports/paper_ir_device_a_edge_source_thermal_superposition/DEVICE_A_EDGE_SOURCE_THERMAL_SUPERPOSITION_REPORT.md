# Device-A free-edge Q causal thermal-current split

Status: `VALIDATED_DEVICE_A_FREE_EDGE_Q_CAUSAL_CURRENT_SPLIT`

One immutable explicit-3D thermal matrix was assembled. For each saved
same-position polarization case, the material-overlap source was split as
`Q_full=Q_free-edge+Q_remainder`. Only the free-edge source was newly solved;
the complementary temperature was inferred as `T_full-T_free-edge` and
independently checked against its matrix right-hand side.

| d (um) | full a-b (nA) | free-edge-Q a-b (nA) | remainder-Q a-b (nA) | edge/full difference | edge-source Ib/Ia | remainder-source Ib/Ia |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.566641 | 3.449318 | -0.882677 | 1.343904 | 0.709581 | 1.478420 |
| 3 | 2.236814 | 2.719125 | -0.482311 | 1.215624 | 0.763893 | 1.220926 |
| 5 | 1.908797 | 1.788827 | 0.119970 | 0.937149 | 0.817929 | 0.951358 |


Values of `edge/full difference` above one mean that free-edge Q produces
more than the entire observed `a-b` difference and the remainder source
partially cancels it. This occurs at `d=1,3 um`. At `d=5 um`, free-edge Q
still produces `93.7%` of the difference and the remainder supplies `6.3%`.
This is a causal linear-operator statement, unlike the preceding
co-localization diagnostic. No equal-power normalization was used.

All complementary source powers, temperatures, currents, and `a-b`
differences close at the reported gates. The full saved field, new edge solve,
and inferred remainder all pass linear residual `<1e-8`; edge solves pass
energy balance `<1%`. No Q clipping, smoothing, gain, rescaling, tiling,
nearest relocation, or deletion occurred: both complementary sources are
retained and reconstruct the immutable full source.

No FDTD, weighting-potential solve, adjoint, AD-FD, or optimization was run.
Raw input NPZ files remain external and SHA-pinned. Derived 3D edge
temperatures are intentionally not serialized because they are reproducible
intermediate arrays; the code and immutable inputs reproduce them.

This attribution is conditional on the present digitized Device-A geometry,
Maxwell Q, thermal operator, and weighting field. It does not by itself prove
that the edge-localized Maxwell Q is physically correct; mesh, exact CAD,
contact/metal thermalization, and beam-contract uncertainties remain.
