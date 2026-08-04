# Weighted adjoint-source collocation diagnostic

Status: `VALIDATED_WEIGHTED_ADJOINT_SOURCE_COLLOCATION_DIAGNOSTIC`

This is a single strong-direction diagnostic. It is not the final
multi-direction combined physical-density AD--FD certificate.

## Preserved Stage 10 failure

The original Stage 10 raw result remains immutable and diagnostic. Its
selected-step combined errors are 2.14938% for the 4 um thermal footprint and
2.88520% for the 6 um footprint. No empirical normalization or gradient
rescaling was applied.

The component split localizes the error:

| path | 4 um, h=0.005 | 6 um, h=0.005 |
|---|---:|---:|
| thermal-material only | 1.78582e-6 | 1.29945e-5 |
| old spatially weighted optical path | 2.74461% | 2.80498% |

The independent nonuniform scalar-\(P_Q\) material-gradient control gives
0.204900% and 0.209005% at \(h=0.01\) and \(0.005\). Thus the component
material Jacobian is not responsible for the previous 2.8% error.

## Corrected source collocation

Raw monitor component arrays share an array shape but live at
\(x+\delta_x\), \(y+\delta_y\), or \(z+\delta_z\). A v261 FieldRegion accepts
one common rectilinear coordinate tuple. Assigning the already-staggered
component values directly to that common tuple is therefore incorrect for a
spatially varying adjoint source.

The corrected operator explicitly solves the component-wise linear
common-FieldRegion-to-native-Yee placement. The source grid is extended by
one zero-padded cell on each positive axis so that the last nonzero native
sample is retained. No source sample in the original support is deleted.

| source construction | h=0.01 | h=0.005 |
|---|---:|---:|
| naive common-grid interpolation | 1.86860% | 1.86974% |
| exact component collocation | 0.797340% | 0.798496% |

The exact-collocation diagnostic passes the strong-direction 1% gate.
Maximum collocation reconstruction error is \(1.2413\times10^{-16}\), native
Q mapping transpose error is \(8.2510\times10^{-16}\), and maximum
forward/adjoint coordinate mismatch is \(4.2352\times10^{-22}\) m.

This does not yet validate the full combined gradient. The next mandatory
gate uses adjoint-aligned, central-localized, design-edge-localized,
smooth/asymmetric, and fixed-seed-random directions at
\(h=0.01,0.005,0.0025\), followed by the requested angle and normalized
multi-direction tests.

Gray-law sensitivity, latent/filter/projection AD--FD, and optimization were
not run.
