# Production latent-mapping VJP report

**Status: `VALIDATED_LATENT_MAPPING_VJP`**

This certificate isolates the production mapping:

`latent 240 x 240`

`-> periodic conic filter`

`-> nominal tanh projection (eta=0.5)`

`-> exact periodic fencepost`

`-> z extrusion`

`-> physical density 241 x 241 x 13`.

It verifies the literal vector-Jacobian product used to return a
physical-density adjoint to the optimization variable. It does not claim that
the complete latent-to-Maxwell-to-thermal objective has passed a v261 finite
difference.

## Production metadata

- mapping version: `periodic_constrained_mapping/v1`;
- period: `6 um`;
- latent spacing: `25 nm x 25 nm`;
- filter radius: `0.5 um`;
- nominal projection threshold: `eta=0.5`;
- robust thresholds: dilated `0.25`, eroded `0.75`;
- isolation gap: `0`;
- physical z layers: `13`, extruded from the same two-dimensional density.

The covector was the completed fixed-K thermal/PTE physical-density gradient
from the preceding Maxwell certificate. Its unscaled L2 norm was
`1.8239208653824576e-21`. It was normalized before this dot test to prevent
loss of floating-point significance. That scalar normalization does not
change the mapping Jacobian or relative AD--FD error.

## AD--FD results

The deterministic latent direction used `h=1e-4`. The gate was relative error
below `1e-5`.

| beta | AD directional | central FD | relative error |
|---:|---:|---:|---:|
| 4 | -1.9566198820915603e-2 | -1.9566198776743704e-2 | 2.25756158270105e-9 |
| 8 | -3.6631784494927454e-2 | -3.663178442359083e-2 | 1.9473969147676157e-9 |
| 32 | -1.0967091912009065e-1 | -1.0967091898628212e-1 | 1.2200912946775672e-9 |

For every beta:

- x periodic-fencepost maximum error: exactly `0`;
- y periodic-fencepost maximum error: exactly `0`;
- z-extrusion maximum error: exactly `0`.

## Exact claim boundary

Validated here:

- periodic-filter transpose;
- tanh-projection chain rule;
- duplicate-fencepost accumulation in the transpose;
- accumulation over all 13 extruded z layers;
- the production `autograd.tensor_jacobian_product` path.

Still pending:

`LATENT_TO_MAXWELL_OBJECTIVE_V261_CENTRAL_FD`.

That next test must evaluate a latent baseline whose mapped physical density
is used consistently in the baseline, plus, minus, and adjoint solver runs.
The earlier sinusoidal physical-density certificate cannot be reused as a
latent baseline because it varies with z, whereas the production mapping
extrudes one two-dimensional field.
