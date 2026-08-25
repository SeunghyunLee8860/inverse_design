# FDTDX user-balanced 50-nm lateral audit

## Outcome

The requested 100-nm to 50-nm design/complete-flake lateral experiment is
complete. The byte-bound certificate is valid, but the lateral pair fails
with status VALIDATED_BLOCKED_FDTDX_USER_BALANCED_LATERAL_CONVERGENCE.

No optical mesh is selected and no optimizer is authorized.

This is forensic evidence for the historical user-balanced FDTDX route. It
is not a result from the newer FDTDX_PARITY_HANDOFF.md contract. In
particular, these runs used Courant 0.5, 24 periods, a 4-period late window,
the historical z2 grid, and an exact-binary endpoint. They must not be reused
as 40-period/Courant-0.25 parity physics or as a continuous-density
n-k-to-epsilon AD result.

## Hash-bound implementation

- Branch: agent/audit-fdtdx-au-dualpol-4um
- 50-nm case implementation: f9ea5e342173d4095513abbe370eabc8c6a11389
- corrected mesh-audit tolerance and run provenance:
  55a35cbd7c036502533588e4ca86639794d3ed7e
- lateral certificate implementation:
  3cfc284b3110fa59dc7aa48cd9a8d8c5f1be2b5d
- full regression suite: 484 passed, 7 subtests passed

The physical 80x80 L500 binary mask was held fixed. Only its Maxwell
representation was replicated exactly 2x2:

- coarse grid: 186 x 186 x 300, 100-nm design/flake x-y
- fine grid: 346 x 346 x 300, 50-nm design/flake x-y
- outer-air x-y: unchanged at 200 nm
- lateral PML: unchanged at 8 cells
- z edges: byte-identical
- fine Yee-cell count: 35,914,800
- fine mesh-contract SHA-256:
  35235086f6908448a2bf671b4e4c3a825320713251c495dd56f85daedd51d277

The first fine source pair under f9ea5e34 was discarded after report review
found that exact floating-point equality incorrectly marked the realized
200-nm outer-air pitch false. The comparison uses only the corrected
55a35cbd artifacts, for which every mesh invariant is true.

## Runtime and GPU use

Ea and Eb were run concurrently on verified-idle B200 GPUs 6 and 7. Existing
Lumerical and Python processes on other GPUs were not touched.

| fine 50-nm case | Ea total | Eb total | observed GPU memory per polarization |
|---|---:|---:|---:|
| all-air source | 257.30 s | 257.05 s | about 33.5 GB |
| exact-binary material | 261.44 s | 261.04 s | about 33.5 GB |

These are first-call compile plus one forward timings, not adjoint or
optimization-iteration timings.

## Artifacts outside Git

- fine source pair:
  /home/seunghyun200/fdtdx_results/user_balanced_lateral50_source_55a35cbd/FDTDX_USER_BALANCED_LATERAL50_SOURCE_PAIR.json
- fine source-pair SHA-256:
  4d963baabf67e15c009d3e1d7898c8682173c62c179d535babd85c99acc57fd4
- fine material report SHA-256, Ea:
  d2725cbf0df318e08ed8606eb6562f4bc32271b2e31d53ba3de3f63021faeb11
- fine material report SHA-256, Eb:
  6282a13fa96241575b243bc91a9ab5989ae1f1f4329f4bed0ccd58e7c4c57f63
- final certificate:
  /home/seunghyun200/fdtdx_results/user_balanced_lateral50_certificate_3cfc284b/FDTDX_USER_BALANCED_XY100_TO_XY50_CERTIFICATE.json
- certificate SHA-256:
  ac6644b0d95cbc8527b0e001626a532ae85af97222517690af45fdb88788d315

Raw NPZ files and run directories remain outside Git.

## Measured comparison

All field and Q comparisons below use physical Yee coordinates. Each mesh is
normalized to the same 285-uW incident-power reporting point. Fine Q is
restricted to coarse controls through exact x/y control-volume overlaps; the
maximum remap power error is 1.87e-16.

| metric | measured maximum | limit | result |
|---|---:|---:|---|
| raw source-power relative change | 74.997% | 0.5% | fail |
| common-power total-Q relative change | 0.2653% | 1% | pass |
| material/component-Q relative change | 45.360% | 2% | fail |
| target-plane tangential complex-E NRMSE | 1.4208% | 2% | pass |
| conservative spatial-Q L2 NRMSE | 14.035% | 5% | fail |
| material-region complex-E NRMSE | 139.722% | 5% | fail |
| Q/closed-flux relative residual | 0.01969% | 2% | pass |
| late-window field stationarity NRMSE | 0.02220% | 0.5% | pass |

Per polarization:

| quantity | Ea | Eb |
|---|---:|---:|
| normalized total Q, 100 nm | 68.7359 uW | 119.4503 uW |
| normalized total Q, 50 nm | 68.5536 uW | 119.3542 uW |
| total-Q relative change | 0.2653% | 0.08045% |
| maximum component-Q change | 30.561% | 45.360% |
| target-plane tangential E NRMSE | 0.8641% | 1.4208% |
| conservative spatial-Q L2 NRMSE | 8.8220% | 14.0349% |
| Au-region E NRMSE | 110.921% | 139.722% |
| TaIrTe4-region E NRMSE | 7.5095% | 4.6269% |

The fine/coarse unscaled source-power ratio is 0.25002956, within 0.01182%
of the cell-area ratio 1/4. This is strong diagnostic evidence that the
historical FDTDX source-amplitude convention changes with lateral sampling.
Common physical-power normalization removes the trivial fourfold Q scale,
but it does not repair the failed component, spatial-Q, or material-field
convergence gates.

## Interpretation and stop rule

- The runs and artifacts are internally valid.
- The 100-nm lateral Maxwell result is not converged against 50 nm.
- Similar normalized total absorption alone is insufficient evidence:
  component absorption, local Q, and material-region fields remain sensitive.
- One 100-to-50-nm pair could not select a production mesh even if it passed;
  a 50-to-25-nm confirmation would also be required.
- Do not launch the 25-nm historical ladder or any optimizer from this
  forensic route. The newer parity handoff deliberately treats FDTDX as a
  candidate generator and reserves final CV0/finer-mesh authority for
  Lumerical.
