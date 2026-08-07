# Beta-continuation optimization contract

> **Stopped methodology diagnostic (2026-08-07).** The implementation below
> did not satisfy its own promotion language.  The supervisor executed only
> one nominal update per beta (two accepted beta=4 updates happened before the
> automated supervisor), while the beta=2 objective was still increasing by
> several percent per iteration.  It also replaced fixed inequality limits by
> a new 1% tighter cap at every proposal.  The beta=8 record is therefore a
> reprojected diagnostic baseline, not a converged beta=8 optimization stage.
> No binary design was promoted.  The corrected optimization restarts as Run
> 003 from the original beta=2 state.

This stage continues the five accepted beta=2 nominal iterations. It does not
replace the validated AD--FD certificates or reinterpret the gray pilot as a
fabrication-ready result.

## Algorithm

- Optimizer: stateful Svanberg MMA separable convex subproblems.
- Persistent state: previous two latent designs plus lower/upper asymptotes.
- Design variables: the existing 373 x 373 latent nodal field on a 50 nm grid.
- Density map: finite, nonperiodic 500 nm conic filter followed by a tanh
  projection.
- Planned beta schedule: 4, 8, 16, 32, 64, and 128, followed by powers of two
  up to 4096 only when the strict projected-binary gate remains open. A stage
  is not promoted merely because its nominal beta was reached.
- Objective scaling is fixed for the full run. There is no empirical gradient
  normalization or per-iteration rescaling.
- Each candidate is evaluated with one GPU Maxwell forward solve, one GPU
  Maxwell adjoint solve, one CUDA thermal forward solve, and one CUDA thermal
  adjoint solve. CPU FDTD fallback is prohibited.

## 500 nm solid and void constraints

The 500 nm conic filter is a sensitivity regularizer and is not, by itself, a
proof that both solid and void features are at least 500 nm. The optimizer also
uses differentiable Zhou solid/void indicators with the Wang thresholds

\[
\eta_e=0.75,\qquad \eta_d=0.25,
\]

which follow from a 500 nm requested feature and a 500 nm conic radius. Grayness
is constrained independently through

\[
G=\left\langle4\rho(1-\rho)\right\rangle.
\]

Every evaluated state is additionally thresholded at 0.5 and audited with a
radius-250-nm disk opening for both solid and void. The audit reports violating
cells but never repairs, clips, smooths, or otherwise edits the candidate.

## Per-evaluation figures

Every recorded state produces:

- latent, filtered, physical, and thresholded-binary structure maps;
- exact solid and void violation maps;
- optical and thermal physical-gradient maps;
- physical-density histogram and binarization metrics;
- iteration-versus-FOM, beta, grayness, and exact-DRC history.

## Final promotion gates

A high-beta state is not called fully binarized until all of the following hold:

- fraction with `0.01 < rho < 0.99` is below 0.1%;
- mean `4 rho (1-rho)` is below 0.001;
- exact 500 nm solid violation count is zero;
- exact 500 nm void violation count is zero;
- a separately thresholded binary design is rerun through the full GPU
  Maxwell/CUDA-thermal evaluation and retains all physics gates.

Until then the status remains
`RUNNING_BETA_CONTINUATION_WITH_500NM_SOLID_VOID_CONSTRAINTS`.
