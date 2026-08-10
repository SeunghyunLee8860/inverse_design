# Run 005: low-beta topology-exploration pilot

Run 005 restarted from the immutable original beta=2 density. Its first phase
was deliberately limited to one fresh GPU Maxwell/CUDA thermal-PTE update with
`move=0.01`.
Run 003 and Run 004 remain unchanged checkpoints.

This pilot tests whether useful topology/FOM motion is possible without using
the discontinuous exact bad-cell count as a low-beta optimization veto:

- differentiable 500 nm solid/void disk constraints remain active;
- both smooth caps must be satisfied;
- a smooth-feasible step may lose at most 0.2% actual FOM;
- exact DRC is diagnostic at beta=2;
- only a greater-than-50% and greater-than-25-cell exact-count increase halts
  the pilot as a catastrophic guard;
- no smaller-move line search was allowed in the one-point experiment;
- beta promotion and the full continuation are prohibited.

That point was reviewed and the bounded extension is authorized through five
total accepted beta=2 updates. Its fixed reprojected caps are `1.0e-3` solid
and `4.5e-5` void. Offline smooth-feasibility trials may use only moves
`0.01`, `0.005`, and `0.0025`; a solver-backed rejection stops the run instead
of launching a smaller GPU retry. Every later beta cap must first be calibrated
by reprojecting the accepted checkpoint; Run 005 cannot silently continue to
beta=4.

Accepted g002 added another 16.43% FOM gain. Because it consumed 99.62% of the
first extension void cap, all next moves were rejected offline and the process
stopped before another Maxwell solve. The remaining pilot resumes from g002
with a second fixed reprojected epoch: `8.2e-4` solid and `5.3e-5` void,
corresponding to current occupancies 0.899/0.846.

Accepted g003 then added 14.22% FOM, for a cumulative 58.89% gain from the
immutable baseline. The second void cap again blocked every next move offline.
To avoid turning beta=2 into repeated cap repair, the final two authorized
points now share one unchanged exploration envelope: `1.0e-3` solid and
`1.0e-4` void. This is deliberately loose only at beta=2; it does not authorize
beta=4, and it cannot be loosened again during this five-point pilot.

The first three accepted updates all improved FOM materially while the smooth
constraints remained feasible. This is evidence for healthy low-beta topology
motion, not evidence that beta=2 or the complete constrained optimization has
converged.

## Five-point beta=2 result

The bounded pilot completed five accepted `move=0.01` updates and then stopped
automatically. The solver-backed FOM rose monotonically from
`9.775174754357e-8` to `1.943798604096e-7 A/W`, a cumulative `+98.8505%`.
The final smooth solid/void values are `5.082822e-4` and `8.089554e-5`, both
inside the fixed `1.0e-3` / `1.0e-4` envelope. Exact 500 nm bad cells are still
`42/6`; at beta=2 they remain diagnostic, not a claim of final manufacturable
geometry. The density is also deliberately gray (`gray fraction=1.0`,
`binarization metric=0.9317`).

This demonstrates useful low-beta exploration without constraint-only
micro-repair. It does not authorize beta=4 automatically. The next checkpoint
must reproject g005 at beta=4, quantify the projection shock in FOM and both
constraints, and approve a new fixed beta=4 cap before another GPU solve.

## Full continuation from g005

The full continuation keeps g005 immutable and first resumes beta=2 until the
real four-update FOM/density plateau passes. It does not promote beta merely
because the five-point pilot ended. Optimized stages are `2, 4, 8, 16, 32, 64,
128`, each with a bounded update budget. Beta values `256` through `8192` are
solver-free projection sharpening only and are reached only after exact 500 nm
solid and void violations are both zero.

Each beta after 2 receives one cap calibrated from its incoming checkpoint;
the cap cannot change inside that stage. Exact thresholded morphology is a hard
nonincrease gate from beta=32. Budget exhaustion or two consecutive minimum
moves advances to the next projection stage instead of launching dozens of
micro-repair iterations or loosening a cap. Beta 32/64/128 use deliberately
tight incoming cap occupancies of 1.10/1.15/1.20 and accept an infeasible
restoration step only when normalized smooth violation falls by at least 1%.

Completion requires gray fraction and mean `4*rho*(1-rho)` both below `0.001`,
zero exact solid/void violations, and a fresh thresholded-binary GPU
Maxwell/CUDA thermal-PTE evaluation. Reaching a beta value alone is not
completion.

## Post-g007 beta=2 cap epoch

The solver-backed g006 and g007 updates improved FOM by `+5.0607%` and
`+4.8293%`; beta=2 therefore had not reached the approved plateau. At g007,
the smooth solid/void values were `4.677681717e-4` and `9.971852083e-5`.
The old exploration envelope placed the void phase at 99.72% occupancy, and
the next `0.005` and `0.0025` proposals were both rejected offline (zero new
Maxwell/thermal solves). Moving to beta=4 or shrinking below the authorized
minimum move would have converted an active topology search into premature
binarization or constraint micro-repair.

The accepted g007 checkpoint is reprojected once into a new fixed beta=2 cap
epoch: `5.50e-4` solid and `1.11e-4` void, with incoming occupancies 85.05%
and 89.84%. The pair is immutable for the remainder of beta=2; it is not
fitted to a solver result and cannot follow individual candidates. The failed
g008-v5 proposals remain raw diagnostic provenance and are archived by the v6
cap-contract mismatch before deterministic reproposal.

## Final beta=2 topology-search epoch after g009

The v6 epoch accepted g008 and g009 at the minimum move `0.0025`.  Their
solver-backed FOM gains were still `+2.3077%` and `+2.2582%`, so treating the
two small moves as an optical/PTE plateau would be false.  They occurred
because the void constraint reached the fixed envelope, not because the
objective or density stopped moving.  The anti-microrepair guard therefore
stopped exactly as intended after g009.

The accepted g009 checkpoint starts one final bounded beta=2 topology-search
epoch with caps `6.00e-4` solid and `2.00e-4` void.  Incoming occupancies are
75.46% and 55.74%.  MMA asymptote memory is reset at g009 and the move ceiling
returns to `0.01`; pre-g009 minimum moves do not throttle this new epoch.  This
pair is immutable through the remaining beta=2 budget and cannot be loosened
again.  Exact DRC remains diagnostic at beta=2, while all later stages retain
phase-wise nonincrease and the final zero-bad-cell requirement.

```bash
CUDA_VISIBLE_DEVICES=2 /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  run_optimization.py --gpu 2 --constraint-device cuda:0
```
