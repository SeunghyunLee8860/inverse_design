# Robust objective and projection audit

Status: `CODE_PATH_CORRECTED_REVALIDATION_REQUIRED`

## Historical invalidation

The historical robust continuation constrained only the dilated `eta=0.35`
and eroded `eta=0.65` projections.  It omitted nominal `eta=0.50` from the
current epigraph.  The historical gray-law diagnostic showed the consequence:
under O3/TE1, both offset projections had the requested signs while nominal
`Ib=+8.386359 nA` had the wrong sign.

Grayness was also constrained only for nominal `eta=0.50`.  The dilated
projection still contained 382 cells with `0.01 < rho < 0.99` and had a much
larger global grayness metric, so the single nominal inequality did not make
all robust realizations binary.

## Corrected contract

The robust evaluator now has three density projections:

```text
eta = 0.35, 0.50, 0.65
```

For every eta it imposes both signed current inequalities (`Ia >= t` and
`-Ib >= t`) and one grayness inequality.  This produces six current plus
three grayness constraints.  The scenario list and count are centralized in
`robust_contract.py` and covered by preflight tests.

The evaluator records every scenario density and grayness gradient.  Resume
rejects manifests without this exact robust contract.  A continuation stage
is now fail-closed unless all nine inequalities are feasible and the returned
worst-case current and epigraph are both positive.

No robust optimization may be run until the shared material law, spatial/time
mesh, and combined-gradient certificates are reissued.

The superseding Lumerical nodal route now has one objective-level development
certificate. Script 39 combines hash-bound Ea/Eb beta-4 latent AD-FD results
in the exact epigraph form and passes one common direction: balanced-objective
relative error `7.779e-5`, with constraint errors `7.779e-5` and
`1.4748e-4`. It performs no solver call and does not enable LD_MMA. The common
unoptimized point has `I_Ea=-8.334 nA` and `I_Eb=-15.591 nA`, so the requested
opposite-sign state is not yet present.
An independent second common direction now also passes: balanced-objective
error `7.011e-5` and Ea/Eb constraint errors `7.011e-5`/`4.093e-5`.
Direction 2 subsequently passed with balanced-objective error `9.981e-5` and
Ea/Eb constraint errors `9.981e-5`/`1.890e-4`.
Direction 3 also passed with balanced-objective error `9.494e-5` and Ea/Eb
constraint errors `9.494e-5`/`1.322e-4`. All four planned common directions
now pass on the RTX development mesh. A fail-closed Lumerical evaluation
driver, formal mesh-convergence/B200 repetition, and actual optimization
remain open.
The user selected CV0 `2.5/50 nm` as the bounded-cost development mesh. One
complete common beta-4 latent direction now passes on that exact mesh: Ea/Eb
current AD-FD errors are `8.515e-5`/`1.3047e-4`; the active signed balanced
objective error is `8.515e-5`, and both constraint errors are below 1%. The
baseline remains non-switching (`I_Ea=-8.70019 nA`, `I_Eb=-16.8637 nA`), so
this validates the objective/constraint gradient but is not an optimization
result. The
existing four-direction family remains additional evidence for `5/50 nm`
staircase only.
