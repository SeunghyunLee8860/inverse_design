# Run050 exact-feasible hard-constraint recovery

The original Run050 continuation reached 854.7535 nA at 285 uW but retained
326 exact 500 nm violations (88 solid and 238 void). Increasing the scalar DFM
penalty through beta=1024 did not change the design: the late-stage RMS latent
step fell to approximately 7.5e-16.

An exact binary cleanup removed every violation, but its best fresh GPU result
was 655.4406 nA, a 23.318% loss. This is a real objective/geometry tradeoff and
must not be hidden by gradient rescaling.

This recovery therefore:

1. uses the exact-feasible solid-first binary candidate as a topology target;
2. inverts the fixed 300 nm conic filter so the *filtered* beta=8 physical
   design still has zero exact violations;
3. starts a fresh NLopt LD_MMA state at beta=8;
4. supplies solid and void opening residuals as two separate inequalities;
5. removes the scalar morphology penalty from the objective; and
6. continues factor-two beta projection only after each LD_MMA stage.

The inverse-filter seed construction is solver-free. Production objective and
gradient evaluations remain the unchanged GPU Maxwell, CUDA thermal/electrical,
and Maxwell-adjoint chain.
