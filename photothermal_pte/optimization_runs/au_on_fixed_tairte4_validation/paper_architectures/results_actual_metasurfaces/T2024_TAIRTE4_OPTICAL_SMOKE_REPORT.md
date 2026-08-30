# 2024 inverse-T / TaIrTe4 optical smoke

Status: `VALIDATED_T2024_FIGURE_DIGITIZED_TAIRTE4_OPTICAL_SMOKE`

This is the first actual metasurface calculation in this folder. It is **not**
the earlier planar-backplane truncation control. It uses the 2024 paper's
MIR inverse-T concept, 1500 x 1000 nm unit cell, 4.75 um target, 35-nm Al2O3
spacer and no MIR passivation. The active graphene layer alone is replaced by
100-nm anisotropic TaIrTe4. The T arm vertices are digitized from Supplementary
Fig. 14 axes because numeric CAD vertices are not published.

## Solver contract

- Lumerical v261 `8.35.4522` GPU forward, x=b polarized.
- x/y periodic boundaries; z PML; normal-incidence plane wave.
- conformal variant 1; 10-nm x/y and 5-nm z structure mesh.
- native mesh: `[151, 101, 186]` (about
  `2,775,000` Yee cells before PML logging).
- TaIrTe4: x=epsilon_b, y=epsilon_a, z=epsilon_c=epsilon_b closure.
- Au: installed `Au (Gold) - CRC`; the paper does not state an Au dataset.
- Al2O3: lossless n=1.62 explicit optical closure, not a paper-certified dataset.

## Forward result

| Metric | Value |
|---|---:|
| GPU wall time | 27.476 s |
| source power per periodic cell | 1.972769480158e-15 W |
| P_Q (periodic pabs) | 4.236429786026e-16 W |
| absorbed flux | 4.242603101750e-16 W |
| closure | 0.145508% |
| reflection | 0.784941771 |
| final auto-shutoff | 5.668510e-07 |

All smoke gates passed: GPU completion, auto-shutoff below 1e-5, closure below
0.5%, finite Q and no negative Q cells. Qx/Qy/Qz remain on their own staggered
Yee coordinates; the plot does not pretend that equal array indices are common
physical positions.

No Q clipping, smoothing, gain, global rescaling or polarization matching was
used. No thermal, PTE, adjoint or optimization solve was run.

## Z architecture status

The 2022 paper publishes P1/P2/L1/L2/W1/W2/D for M1-M5, but the PDFs do not
provide machine-readable Z polygon vertices or a fixed junction/crossing angle.
The audit plot therefore shows only hatched dimension envelopes and explicitly
forbids them as Maxwell CAD. This is a topology-provenance blocker, not a claim
that the Z architecture cannot be simulated.
