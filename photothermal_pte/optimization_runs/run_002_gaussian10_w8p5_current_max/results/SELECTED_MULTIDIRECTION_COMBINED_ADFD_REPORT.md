# Selected multidirection combined physical-density AD–FD

Status: `VALIDATED_SELECTED_MULTIDIRECTION_COMBINED_PHYSICAL_RHO_ADFD`

The corrected component-wise Yee/thermal chain passes five independent physical-density directions. These are real centered FD reruns at `h=0.005`; no empirical normalization, FD-derived scale, or gradient rescaling is used.

| direction | AD (A) | FD (A) | relative error | result |
|---|---:|---:|---:|---|
| adjoint_aligned | 8.545357319273e-20 | 8.545598286818e-20 | 0.002820% | pass |
| smooth_asymmetric | 5.132354555196e-20 | 5.132269505153e-20 | 0.001657% | pass |
| central_localized | -9.726082353760e-22 | -9.728464395188e-22 | 0.024485% | pass |
| design_edge_localized | -2.788986539416e-21 | -2.789363480998e-21 | 0.013514% | pass |
| fixed_seed_random | 3.554523873701e-22 | 3.561808313329e-22 | 0.204515% | pass |

Worst directional error is `0.204515%`; worst normalized error is `0.002820%`. New cases used eight GPU Maxwell forward solves and eight CUDA thermal forward solves. No CPU FDTD or CPU thermal fallback was used.

This closes the combined *physical-density* gate. It does not yet certify the latent→finite-filter→projection chain and does not itself start optimization.
