# Device-A beam-position terminal-current sensitivity

Status: `ROBUST_MAXWELL_REVERSAL`

All promoted comparisons use the immutable s0 thermal contract: 60-um
lateral domain, 20-um Si depth, 100-nm core cells, and 10-nm TaIrTe4 cells.
The earlier 48-um analytic run is preserved but excluded.

| signed s (um) | Maxwell Ia isolated (nA) | Maxwell Ib isolated (nA) | Maxwell ratio isolated | Maxwell ratio perfect | analytic ratio |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 10.7567 | 7.08781 | 1.517631 | 1.562067 | 0.682023 |
| 3.0 | 8.07262 | 4.99032 | 1.617656 | 1.638590 | 0.682351 |
| 4.0 | 4.94061 | 2.60236 | 1.898514 | 1.842354 | 0.683384 |

The paper digitization gives `0.836590 ± 0.008526`.
The analytic source already inputs the larger b-polarized TMM absorption and
is a control, not a paper reproduction. No empirical current normalization,
polarization matching, Q clipping, smoothing, gain, or rescaling was used.

All displayed gradient and local-integrand maps use the strict four-neighbour
mask requested by the user: a cell is hidden if any of ±x or ±y lies outside
the TaIrTe4 mask. Temperature and Q maps retain the full physical flake.
