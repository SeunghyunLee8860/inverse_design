# Extended full-latent AD–FD: ten perturbations

**Status: `VALIDATED_EXTENDED_FULL_LATENT_PERTURBATION_ADFD`**

The final five-direction certificate was preserved and five new latent
directions were evaluated with fresh centered Maxwell FD pairs at `h=0.005`:
uniform, x-antisymmetric, y-antisymmetric, diagonal-quadrupole, and
radial-ring. No clipping, gradient rescaling, or empirical normalization was
used.

| scenario | slope through zero | R2 | NRMSE | angle | worst individual error |
|---|---:|---:|---:|---:|---:|
| 4 µm | 1.00015826 | 0.999999953 | 0.02651% | 0.01218° | 0.30412% |
| 6 µm | 1.00013367 | 0.999999943 | 0.02597% | 0.01275° | 0.23668% |

The 4 and 6 µm plots are intentionally separate. Point labels 1–5 are the
original certificate directions and 6–10 are the new directions. The raw
FSP/NPZ files remain outside Git and are SHA-pinned in the manifest.

- [4 µm ten-direction AD–FD](figures/21_full_latent_adfd_4um_10directions.png)
- [6 µm ten-direction AD–FD](figures/22_full_latent_adfd_6um_10directions.png)
- [ten perturbation maps](figures/23_ten_latent_perturbation_maps.png)
