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
