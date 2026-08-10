# Run 005 bounded beta=2 pilot audit

Status: `PAUSED_AFTER_RUN005_BOUNDED_BETA2_GPU_PILOT`

## Outcome

Run 005 completed exactly five accepted beta=2 MMA updates. Every accepted move
was 0.01 and every fresh solver-backed evaluation increased the signed
`+I_PTE/P_incident` objective. The pilot then stopped automatically and did not
promote beta.

| point | FOM (A/W) | gain from previous | solid C | void C | exact solid/void |
|---|---:|---:|---:|---:|---:|
| baseline | 9.775174754357e-8 | -- | 1.192176e-3 | 2.563430e-5 | 158 / 0 |
| g001 | 1.167963771133e-7 | +19.4826% | 9.047577e-4 | 3.816873e-5 | 44 / 2 |
| g002 | 1.359883308119e-7 | +16.4320% | 7.371437e-4 | 4.482911e-5 | 43 / 2 |
| g003 | 1.553223792767e-7 | +14.2174% | 6.430949e-4 | 5.244080e-5 | 40 / 6 |
| g004 | 1.747899299708e-7 | +12.5336% | 5.673841e-4 | 6.396462e-5 | 41 / 1 |
| g005 | 1.943798604096e-7 | +11.2077% | 5.082822e-4 | 8.089554e-5 | 42 / 6 |

The cumulative FOM gain is `+98.8505%`. This is substantial objective and
topology motion, not a sequence of constraint-only repairs.

## Constraint behavior

The differentiable 500 nm solid and void constraints were active in every MMA
step. g004 and g005 used one unchanged beta=2 exploration envelope:
`C_solid <= 1.0e-3` and `C_void <= 1.0e-4`. Final occupancies are 0.5083 and
0.8090. Exact thresholded DRC was diagnostic below beta=32, with a catastrophic
growth guard, and was never used as a monotone low-beta veto.

The driver demonstrated the anti-waste behavior twice before the final cap
epoch: when a cap blocked moves 0.01, 0.005, and 0.0025, all three proposals
were rejected offline with zero Maxwell and zero thermal solves. No move below
0.0025 was attempted. No solver-backed rejection was followed by a smaller GPU
retry.

## Final numerical gates

| gate | g005 value | requirement | result |
|---|---:|---:|---|
| optical six-face closure | 3.0413e-7 | < 5e-3 | pass |
| Q mapping power error | 1.8082e-16 | conservative | pass |
| Q pullback transpose error | 8.6143e-16 | < 1e-12 | pass |
| thermal linear residual | 9.5099e-11 | < 1e-8 | pass |
| thermal energy balance | 1.1039e-12 | < 1e-2 | pass |
| forward auto-shutoff | 8.9072e-8 | < 1e-5 | pass |
| adjoint auto-shutoff | 9.3353e-8 | < 1e-5 | pass |
| forward/adjoint coordinate mismatch | 0 m | exact common grid | pass |

g004 and g005 end-to-end evaluation times were 748.606 s and 747.127 s.
Each used one GPU Maxwell forward, one GPU Maxwell adjoint, one CUDA thermal
forward, and one CUDA thermal adjoint solve. CPU FDTD fallback, CPU thermal
fallback, empirical normalization, gradient rescaling, and post-hoc density
repair were not used.

## Binarization status and next gate

This is not a finished manufacturable design. At g005 the physical density
range is `[0.2882, 0.7107]`, gray fraction is 1.0, binarization metric is
0.9317, and exact bad cells are 42 solid / 6 void. These values are expected for
the deliberately exploratory beta=2 stage.

The next step is not another unrestricted optimization run. It is a solver-free
reprojection of the immutable g005 checkpoint at beta=4, reporting projection
shock, FOM-surrogate implications, smooth solid/void constraints, exact DRC,
and a proposed fixed beta=4 cap. Only after that audit is reviewed may fresh
beta=4 GPU solves begin. The final binary gate remains zero solid and zero void
bad cells.

## Provenance

- g005 checkpoint: `checkpoints/run005_b002_s005_g005_accepted_mma.npz`
  (`ba24655d86f0c0a550458cde94a969b8093918af38b23b813b7d400f89a39a19`)
- g005 raw evaluation NPZ:
  `/data/seunghyun/tairte4/raw_artifacts/run005_lowbeta_topology_pilot_20260808/b0002_s005_g005_retry0_evaluation/selected_full_latent_adjoint_preparation.npz`
  (`634e6378d73823b187e45190b65dea69c9028cf91b42906c7a820b0e3065a5ab`)
- Full external paths, sizes, and SHA-256 values are recorded in
  `../manifests/RAW_ARTIFACT_MANIFEST.json`.
