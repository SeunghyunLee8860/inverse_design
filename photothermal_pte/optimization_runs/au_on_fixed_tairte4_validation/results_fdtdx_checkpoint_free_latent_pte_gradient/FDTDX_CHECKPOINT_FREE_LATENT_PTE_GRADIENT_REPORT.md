# Checkpoint-free latent/filter/projection PTE gradient

Status: `VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_LATENT_PTE_GRADIENT`

The checkpoint-free combined physical-density gradient was pulled back through
the same finite conic filter and tanh projection as the frozen end-to-end
certificate. No FDTD, thermal, electrical, or finite-difference solve was
rerun, and no time history or checkpoint stack was used.

| metric | result |
|---|---:|
| physical-gradient vector error | 0.153893% |
| latent-gradient vector error | 0.135856% |
| latent-gradient norm error | 0.104034% |
| latent-gradient angle | 0.050034 deg |
| worst strong latent AD--FD error | 0.411825% |
| worst normalized directional error | 0.101659% |

The runtime-bearing Maxwell VJP is therefore one forward plus one adjoint solve
with zero checkpoints. Raw arrays remain outside Git and are SHA-256 pinned in
the manifest.
