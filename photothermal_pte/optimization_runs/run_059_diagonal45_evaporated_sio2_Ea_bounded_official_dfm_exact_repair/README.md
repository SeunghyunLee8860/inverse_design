# Run 059: +45-degree contacts, evaporated SiO2, E||a

This run repeats the bounded official-DFM/exact-repair inverse-design
protocol of Run 057 with the electrical terminal pair rotated to the +45
degree axis.

- The TaIrTe4 flake remains exactly 24 um x 24 um. It is not enlarged or
  rotated.
- The optimization grid remains 20 um x 24 um at 100 nm spacing, matching
  Run 057/058.
- The low terminal is the southwest corner and the high terminal is the
  northeast corner. Each equipotential overlap is 2 um deep along the +45
  degree terminal axis.
- Terminal-overlap TaIrTe4 nodes are locked to rho=1 in latent continuation,
  optical/thermal/electrical solves, gradients, official DFM, and exact repair.
- Lumerical axes remain x=b, y=a, z=c. This run uses E||a.
- The TaIrTe4/SiO2 interface scenario is `evaporated`, as in Run 057/058.
- One physical GPU is exposed. The optimizer uses NLopt LD_MMA with beta
  continuation 1 through 128 and exact 500 nm binary repair/evaluation.

Raw solver artifacts are written under
`/data/seunghyun/tairte4/artifacts/tairte4_diagonal_45_contact_anchored/`.
Published checkpoints and the final report are written to `results_v2/`.
