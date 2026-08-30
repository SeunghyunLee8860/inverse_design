# Runs 047/048 exact-binary cleanup

Status: **COMPLETED_RUN047_048_EXACT_BINARY_ZERO_500NM_VIOLATION_PHYSICS**

Both selected designs are exactly `0/1`. The independent discrete 500 nm audit reports solid bad nodes = 0 and void bad nodes = 0. The raw continuous/gray checkpoints are preserved; this publication uses the separately repaired and freshly evaluated exact candidates.

| Run | pol. | source gray | source solid+void bad | final bad | continuous I (nA) | exact I (nA) | change | physical gates |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 047 | Ea | 2.820% | 167+161 | 0 | 48.7127 | 46.3981 | -4.751% | True |
| 048 | Eb | 2.814% | 255+64 | 0 | 22.5407 | 22.6064 | 0.291% | True |

Run047's exact structure loses more than the legacy 1% objective-preservation threshold. That is reported as a performance diagnostic, not hidden and not used to undo the geometry gate. Run048 preserves the objective within 1%.

The cleanup enforces the requested 500 nm solid/void rule but does **not** remove electrically disconnected islands. Connectivity was explicitly absent from these optimization contracts; adding it now would be a different design constraint and a different geometry.

Every exact candidate has a fresh v261 GPU Maxwell forward evaluation followed by the unchanged CUDA thermal/electrical path. The publisher independently verifies density/field SHA-256 values, reintegrates current, checks exact geometry, and produces Q, temperature, strict-centered gradient, weighting-potential, and current-contribution maps. No Q clipping, smoothing, gain, global rescaling, tiling, or source deletion is used.
