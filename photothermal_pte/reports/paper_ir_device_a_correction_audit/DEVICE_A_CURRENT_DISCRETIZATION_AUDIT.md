# Device-A current-discretization audit

Status: `DIAGNOSTIC_CURRENT_QUADRATURE_DISAGREEMENT_PRESERVED`

No new Maxwell or thermal solve was run. The immutable saved temperature and
weighting-potential arrays were reintegrated with three separately named
quadratures.

| method | abs(Ia)/abs(Ib) |
|---|---:|
| legacy mixed centred/one-sided cell gradient | 1.410409918 |
| strict four-neighbour cell mask | 1.389083373 |
| common internal-face bilinear | 1.402701047 |
| digitized paper target | 0.836589698 |

The strict scheme satisfies the requested masking rule but removes physical
boundary volume. The internal-face scheme collocates T and psi differences on
the same faces, but it omits exterior half-control-volume/contact quadrature.
Neither diagnostic is silently promoted as production. If all three retain a
ratio above one, the old polarization reversal is not caused only by the
legacy one-sided gradient implementation.
