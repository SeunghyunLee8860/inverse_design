# Run 004 one-point GPU pilot audit

Status: `PAUSED_AFTER_ONE_POINT_JOINT_PROGRESS_AUDIT`

The pilot started from the immutable original beta=2 latent density. One
candidate was accepted. The following candidate was interrupted before
completion and is excluded from accepted history.

## Accepted g001 result

| metric | baseline | g001 | change |
|---|---:|---:|---:|
| FOM (A/W) | 9.775174754e-8 | 1.024991878e-7 | +4.8566% |
| smooth solid | 1.192175674e-3 | 1.092145945e-3 | -8.390% |
| smooth void | 2.563430042e-5 | 2.976605352e-5 | +16.118% |
| exact solid bad cells | 158 | 96 | -62 |
| exact void bad cells | 0 | 1 | +1 |
| rho RMS change | — | 0.003117 | — |
| rho maximum change | — | 0.003283 | — |

This was genuine joint progress, not a constraint-only update: FOM increased,
the dominant smooth solid violation decreased, and the total exact bad-cell
count fell from 158 to 97. The latent move was small enough that no premature
box saturation occurred.

## Why the pilot was paused

After g001, the void metric was already 99.22% of its beta=2 cap. Offline
prescreen rejected moves 0.0025, 0.00125, and 0.000625. The next feasible
proposal used move 0.0003125 and changed exact counts only from 96/1 to 95/0.
Launching full forward/adjoint physics for repeated moves of this scale would
recreate the Run 003 failure mode: expensive incremental constraint repair.

The g002 solver attempt was therefore interrupted and is not accepted.

## Physics gates and cost for g001

- one GPU Maxwell forward and one GPU Maxwell adjoint;
- one CUDA thermal forward and one CUDA thermal adjoint;
- total preparation wall time: 730.50 s;
- forward Maxwell wall time: 229.94 s;
- optical closure: 5.373e-6;
- Q mapping error: 0;
- thermal residual: 9.548e-11;
- thermal energy-balance error: 3.835e-14;
- forward/adjoint auto-shutoff: 8.970e-8 / 9.990e-8.

## Required policy correction before continuation

The early-beta rule must permit a meaningful solid/void topology trade while
preventing total morphology degradation. A proposed correction is to use a
less restrictive early void envelope together with exact total-bad-cell
nonincrease and strict FOM preservation, then activate phase-specific trust
caps only after the design is substantially less gray. This is a material
contract change and must be reviewed before another GPU launch.
