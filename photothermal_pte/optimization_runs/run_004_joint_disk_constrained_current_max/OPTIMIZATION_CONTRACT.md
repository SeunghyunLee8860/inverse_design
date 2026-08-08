# Run 004 corrected optimization contract

## Immutable physics

Run 004 preserves the certified Run 002 physical chain: 10 um scalar Gaussian
illumination with target-plane waist 8.5 um; finite nonperiodic 18.6 x 18.6 um
design window on a 373 x 373, 50 nm nodal grid; air/complex-Kitamura-SiO2
single-wavelength endpoints; 1 um design height; signed `+I_PTE/P_incident`
objective; GPU Maxwell forward/adjoint and CUDA float64 thermal/PTE
forward/adjoint. The finite conic filter radius remains 500 nm.

## Restart

The trajectory starts from the immutable original beta=2 latent field
`d3617baf54d54e735feba9d85c439ee77bcdf5ddaeec47e12c812bf036b2c87e`.
Run 003 g095 and the partial g096 evaluation are not used as initial states.

## Joint manufacturability from iteration zero

Both phases use the differentiable soft disk-opening constraint with a 250 nm
radius Euclidean disk, corresponding to the requested 500 nm solid and void
feature diameter. This is used at every beta, beginning with the first beta=2
MMA proposal. The former Zhou p=8 surrogate is not part of Run 004.

Fixed stage caps are:

| beta | solid cap | void cap |
|---:|---:|---:|
| 2 | 1.25e-3 | 3.00e-5 |
| 4 | 1.00e-3 | 2.50e-5 |
| 8 | 7.50e-4 | 2.00e-5 |
| 16 | 5.00e-4 | 1.50e-5 |
| 32 | 2.50e-4 | 1.00e-5 |
| 64 | 1.00e-4 | 7.50e-6 |
| 128 | 5.00e-5 | 5.00e-6 |
| 256 | 2.50e-5 | 2.50e-6 |
| 512 | 1.50e-5 | 1.50e-6 |

The beta=2 caps are close to the original-state disk metrics rather than the
Run 003 loose 0.04/0.04 caps. At beta=2--16 MMA uses these fixed caps directly,
so topology can change anywhere inside the already narrow feasible envelope.
From beta=32, each phase receives at most 1% relative or 1e-7 absolute local
slack and can never exceed its fixed cap.

## Continuation and anti-waste gates

- beta schedule: 2, 4, 8, 16, 32, 64, 128, 256, 512;
- initial MMA move ceiling: 0.01; minimum adaptive ceiling: 0.0025;
- minimum accepted updates: 8 at beta=2, 6 at later beta;
- maximum accepted updates: 20 per beta;
- promotion requires fixed-cap feasibility and a four-update joint plateau:
  maximum relative FOM change below 0.5%, rho RMS change below 0.5%, and rho
  maximum change below 3%;
- after six accepted updates, the driver halts if net FOM gain is below 0.2%
  and exact bad-cell reduction is below 2%, unless the stage already passes;
- a feasible candidate must stay feasible and retain at least 99.5% of actual
  FOM; an infeasible candidate must reduce normalized violation by at least 1%
  and retain at least 98% of FOM.

The first launch is deliberately limited to three newly accepted GPU-backed
updates. The full continuation is resumed only after those three points show
joint FOM/manufacturability progress and no premature latent saturation.

Exact thresholded 500 nm solid/void morphology is recorded at every state. Its
nondifferentiable nonincrease veto begins at beta=32; before that point the
smooth disk constraints guide topology without freezing threshold crossings.
The final gate still requires exactly zero bad solid and zero bad void cells.

## Prohibited shortcuts

No CPU FDTD or CPU thermal fallback, post-hoc binary repair, density clipping,
empirical gradient normalization, objective rescaling, or skipped fresh final
binary GPU/CUDA validation is allowed.
