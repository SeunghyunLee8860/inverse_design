# Run 005 one-point low-beta topology pilot audit

Status: `PAUSED_AFTER_RUN005_ONE_POINT_GPU_PILOT`

## Decision

The approved beta=2, `move=0.01` point is a healthy joint topology/FOM step. It
is not constraint-only repair: the actual GPU/CUDA objective increased strongly,
the smooth solid constraint improved, the void constraint stayed below its
explicit pilot cap, and the discontinuous exact audit improved even though it
was not used as a low-beta veto.

This result authorizes review of a 3--5 accepted-update beta=2 pilot. It does not
authorize beta promotion, full continuation, or optimization completion.

## Baseline versus accepted point

| metric | baseline | accepted g001 | change |
|---|---:|---:|---:|
| FOM (A/W) | 9.775174754357e-8 | 1.167963771133e-7 | +19.4826487% |
| smooth solid | 1.192175674370e-3 | 9.047577467959e-4 | -24.1086892% |
| smooth void | 2.563430041771e-5 | 3.816872716968e-5 | +48.8970892% |
| solid cap occupancy | 94.6171% | 71.8062% | -22.8109 points |
| void cap occupancy | 51.2686% | 76.3375% | +25.0689 points |
| exact solid/void bad cells | 158 / 0 | 44 / 2 | total 158 -> 46 |
| physical-rho RMS change | 0 | 0.0124517 | diagnostic |
| physical-rho max change | 0 | 0.0131299 | diagnostic |
| binarization metric | 0.971210 | 0.965453 | -0.592751% |

The accepted latent move was exactly 0.01. No smaller retry was proposed or
evaluated. Exact DRC was diagnostic only; its catastrophic limit was 237 cells,
and the proposal produced 46.

## Physics and runtime

- GPU: NVIDIA RTX 6000 Ada, device 2
- forward FDTD solver time: 195.097 s
- adjoint FDTD solver time: 284.822 s
- end-to-end evaluation wall time from driver events: 739.768 s
- optical closure: 4.7383485e-6
- Q mapping error: 1.7869088e-16
- Q pullback transpose error: 3.8306910e-16
- thermal residual: 9.7435481e-11
- thermal energy balance: 1.2477984e-12
- forward/adjoint auto-shutoff: 9.19675e-8 / 8.42705e-8
- component-coordinate mismatch: 0 m

## Cap calibration caveat

The initial solid occupancy, 0.9462, is in the recommended 0.8--0.95 interval.
The initial void occupancy, 0.5127, is deliberately lower for this bounded
experiment because the prior `move=0.01` topology proposal required more void
headroom. Therefore `5e-5` is a pilot-specific envelope, not a promoted cap
schedule. The 3--5 step pilot must monitor void-cap consumption, FOM, density
motion, and exact DRC together. Every later beta requires fresh checkpoint
reprojection and cap calibration.

## Provenance

The raw evaluation NPZ is not committed to Git. Its path is
`/data/seunghyun/tairte4/raw_artifacts/run005_lowbeta_topology_pilot_20260808/b0002_s001_g001_retry0_evaluation/selected_full_latent_adjoint_preparation.npz`,
size `7,264,178` bytes, SHA-256
`15e7a4cb0ccf335645a0a16bdc4c8eb91abd2ff77cb30b44572742ff9dd2aac7`.
Complete proposal, state, evaluation, and checkpoint provenance is in
`manifests/RAW_ARTIFACT_MANIFEST.json`.
