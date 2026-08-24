# FDTDX frozen-Q thermal domain-size convergence

## Decision and boundary

The current frozen-Q prototype thermal operator passes the thermal domain-size
ladder for Ea and Eb at the previously selected x/y factor 2 and z factor 2.
The selected diagnostic domain is:

- lateral half-span: 48 um;
- Si substrate depth: 30 um;
- top-air height: 3 um;
- shape: `548 x 548 x 72` (`21,621,888` unknowns).

The axis-isolated lateral, substrate, and top-air ladders pass both successive
pairs.  The combined `48/30/3 um` to `64/40/4 um` tail also passes, so the
selection is not inferred only from independent one-axis changes.

This closes domain **size**, not the thermal boundary-value model.  Side and
bottom ambient Dirichlet conditions, the top convection coefficient, and the
Au-TaIrTe4, TaIrTe4-SiO2, and SiO2-Si interface conductances remain physical
assumptions with no uncertainty certificate.  The optical mesh and actual
device electrical geometry remain blocked.  No production multiphysics mesh
is selected and optimization remains forbidden.  No Lumerical file or job was
read, edited, launched, or reinterpreted.

## Provenance chain

Every case revalidates the prior thermal x/y certificate by bytes:

- prior certificate:
  `/home/seunghyun200/fdtdx_results/frozen_q_thermal_xy_certificate_65f2d44e/FDTDX_FROZEN_Q_THERMAL_XY_CERTIFICATE.json`
- prior SHA-256:
  `811c79ded3ba1b7cfe70d23f75bfad76c665566815e881645eee8e4cdbfae96f`
- prior selected x/y factor: 2
- prior selected z factor: 2
- prior domain/boundary convergence: false
- prior production mesh selected: false
- prior optimizer start allowed: false

Every case also revalidates the same blocked optical z32 certificate SHA-256
`079a6fbbb78aeab29d5e7460815f22208708a307f02572dc956f244433b9bb97`,
the exact-binary 375-cell Au mask, common 285-uW normalization, conservative Q
mapping, solver residual, energy balance, clean Git state, and exclusive
physical GPU ownership.

The new baseline reproduces the prior x/y-factor-2 Ta temperature, x/y
gradients, coordinates, and base source-power map exactly, array for array.
The baseline raw NPZ hashes are therefore exactly the prior selected x/y raw
hashes.  This prevents a silent change in Q, mesh, or observation operator.

## Domain construction and comparison rule

No existing mesh face moves.  Domain growth only appends coarse outer cells:

- lateral: 32 to 48 to 64 um half-span, using 4-um outer intervals;
- substrate: 20 to 30 to 40 um depth, using 10-um deep-Si intervals;
- top air: 2 to 3 to 4 um, using 0.5-um outer-air intervals.

Every interval is still subdivided by x/y factor 2 or z factor 2.  The Au,
TaIrTe4, SiO2, and near-interface cells are byte-identical across the ladder.
The exact-binary Au fraction is replicated only by the selected mesh factor;
no gray interpolation or `rho**3` law is introduced.

The source is compared on the common +/-32-um base window.  It is block-summed,
not averaged, and its total remains equal to the full-domain mapped source.
Temperature and gradient comparisons use the common 160 x 160 TaIrTe4 base
observation grid.

The limits were fixed before the cases ran:

- Ta temperature-map NRMSE: at most 1%;
- Ta maximum-temperature relative change: at most 1%;
- Ta mean-temperature relative change: at most 1%;
- combined x/y temperature-gradient NRMSE: at most 2%;
- base source-power-map NRMSE: at most `5e-12`;
- base coordinate tolerance: `2e-18 m`;
- both successive pairs for each axis and the combined ladder must pass for
  Ea and Eb.

## Convergence results

| ladder pair | pol. | T-map NRMSE | Tmax relative | Tmean relative | combined-gradient NRMSE | source NRMSE |
|---|:---:|---:|---:|---:|---:|---:|
| lateral 32 to 48 | Ea | 0.040009% | 0.008394% | 0.077595% | 0.001360% | 0 |
| lateral 32 to 48 | Eb | 0.040578% | 0.008652% | 0.078378% | 0.001391% | 0 |
| lateral 48 to 64 | Ea | 0.002725% | 0.000572% | 0.005285% | 0.000094% | 0 |
| lateral 48 to 64 | Eb | 0.002764% | 0.000589% | 0.005338% | 0.000096% | 0 |
| substrate 20 to 30 | Ea | 0.264779% | 0.062892% | 0.511945% | 0.008521% | 6.31e-17 |
| substrate 20 to 30 | Eb | 0.268590% | 0.064787% | 0.517189% | 0.008720% | 6.68e-17 |
| substrate 30 to 40 | Ea | 0.068267% | 0.016102% | 0.131887% | 0.001934% | 6.31e-17 |
| substrate 30 to 40 | Eb | 0.069248% | 0.016587% | 0.133234% | 0.001979% | 6.68e-17 |
| top air 2 to 3 | Ea | 0.030202% | 0.005145% | 0.017924% | 0.087845% | 0 |
| top air 2 to 3 | Eb | 0.032714% | 0.008356% | 0.019213% | 0.097432% | 0 |
| top air 3 to 4 | Ea | 0.015996% | 0.002702% | 0.010295% | 0.046804% | 0 |
| top air 3 to 4 | Eb | 0.017280% | 0.004426% | 0.011031% | 0.051452% | 0 |
| combined baseline to 48/30/3 | Ea | 0.391563% | 0.087230% | 0.754377% | 0.087741% | 6.31e-17 |
| combined baseline to 48/30/3 | Eb | 0.396753% | 0.086816% | 0.760960% | 0.097313% | 6.68e-17 |
| combined 48/30/3 to 64/40/4 | Ea | 0.196702% | 0.043152% | 0.377742% | 0.046800% | 6.31e-17 |
| combined 48/30/3 to 64/40/4 | Eb | 0.199271% | 0.042820% | 0.380942% | 0.051446% | 6.68e-17 |

All gates pass.  For every reported metric, the second isolated-axis tail is
smaller than the first.  The combined large-tail errors are also smaller than
the baseline-to-selected changes.

## Runtime and practical cost

Ea and Eb ran concurrently on physical GPUs 6 and 7.  Other users' Lumerical
jobs were observed on GPUs 0 and 4 before the run and were not touched.  The
18 solves completed in about 3 minutes 14 seconds of wall time.

| case | shape | unknowns | Ea total | Eb total | Ea/Eb PCG iterations |
|---|---:|---:|---:|---:|---:|
| baseline 32/20/2 | 532 x 532 x 66 | 18,679,584 | 18.55 s | 17.79 s | 3,750 / 3,750 |
| lateral 48 | 548 x 548 x 66 | 19,820,064 | 19.23 s | 19.48 s | 4,050 / 4,050 |
| lateral 64 | 564 x 564 x 66 | 20,994,336 | 20.68 s | 19.77 s | 4,275 / 4,200 |
| substrate 30 | 532 x 532 x 68 | 19,245,632 | 19.02 s | 20.08 s | 4,125 / 4,125 |
| substrate 40 | 532 x 532 x 70 | 19,811,680 | 19.98 s | 19.24 s | 4,425 / 4,400 |
| top air 3 | 532 x 532 x 70 | 19,811,680 | 19.04 s | 18.69 s | 3,750 / 3,725 |
| top air 4 | 532 x 532 x 74 | 20,943,776 | 20.44 s | 19.66 s | 3,750 / 3,725 |
| combined 48/30/3 | 548 x 548 x 72 | 21,621,888 | 21.07 s | 21.54 s | 4,475 / 4,475 |
| combined 64/40/4 | 564 x 564 x 78 | 24,811,488 | 25.42 s | 24.66 s | 5,175 / 5,150 |

All explicit relative residuals are below `1e-9`; all energy-balance relative
errors are below `7.5e-11`.  The selected diagnostic thermal pair is about 22
seconds cold wall time, far below the blocked optical z32 forward time of
about 18.5 minutes per polarization.

At the selected diagnostic domain, the frozen-Q TaIrTe4 results are:

| pol. | base Tmax rise | base mean rise | native Tmax rise |
|:---:|---:|---:|---:|
| Ea | 0.98908633 K | 0.11481721 K | 0.98939164 K |
| Eb | 1.64713481 K | 0.19503319 K | 1.64830533 K |

These are properties of an optically unconverged frozen Q field and the
current assumed rectangular thermal geometry.  They are not validated device
temperature or current predictions.

## External artifacts

Case root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_domain_a7d5a52a/`

- clean runner commit:
  `a7d5a52a7c0eac50156748db082aee031510be14`
- runner SHA-256:
  `98ac8e11d52ce567f48e898838d3bf10ce2bb0c49b51586028540f678f5d176f`

| case | pol. | report SHA-256 | raw NPZ SHA-256 |
|---|:---:|---|---|
| baseline | Ea | `8c106060a7bd05b2463351e1113119230eb6823cc27d5d7113eaa4b8cffccbab` | `2bf49a265c2e3eaa8015d8e4b6b19318d4541bbbcc9b2d10c81c0736b92f5fcf` |
| baseline | Eb | `a4d12c3233c339e2553cf171be37f4b2c0e39f0a2f5a39513b4743521c472fea` | `b8907b383df2ca420bbe63bbb90565cd50bfadee2a557d5f31daf6cdeb30d5d5` |
| lateral 48 | Ea | `3eb80e3d8f5b458cf7f3fa9ef3e9c0a7a91d9c4b3b122fb78d21d2ab5d3ea1a8` | `ec3fdbfd414138d92ff05240f0d32e1ce51a5d2ca7e6a99688e868d366b59175` |
| lateral 48 | Eb | `0053fc7f72691328890e9928a0c1a963b0effd2023b47d2c2b09855c17587f19` | `14ab92d58b3402ec4944bd7f072012b84ffc02cb28f5e6944454791fe9afd9b7` |
| lateral 64 | Ea | `18cc6d8771311d300582767bf12451333670db5b99b4c9e1e58d71073c82c133` | `6dda91cb40dbb38779daed7b33baa95e405cc63ea7618a4c538bbf769f04fe76` |
| lateral 64 | Eb | `4c3da24c3b0a12a686bd488a270b0bd3147fdb58f27f0b3833e730c8820da328` | `2a93e18befcd375d65c8b32e6158d67b7a9b26bcee10b6fbc0d3722b2b580f72` |
| substrate 30 | Ea | `e30658b2e0e15e624b77777c2fe41f9e4b71e409d80f1fc1a3dc08945eaa5759` | `276a022f44e456878b6320285da139a1efb4cb67b3d8ff2a97d0c9a4c573a11e` |
| substrate 30 | Eb | `a4ab600cb5bba725019cf2f8e9d06c76b3d18eff694f6044c8d88389b11a7618` | `d04b27eef61979055cca389fb2b17992519b19da4a9395d8b93c6f4af84506e6` |
| substrate 40 | Ea | `b2b2a20a02d00a1c9a1110d2ca8d5b19cbe2afa2b37afa7228bc063651565f40` | `f08e62c497b04384a656067f9b72fb46b35172dd2baf5b6b581fdb4cae3a0514` |
| substrate 40 | Eb | `b9fa93e3ade0607553a8ac5e0150ac0fc5f0cb3a385320d5733269fbe4eaa598` | `fa4955ab8293d27b981759f3eed1113f115af21c0cac6f90d718d1a5adff1128` |
| top air 3 | Ea | `8f3bac87a122b5f3aba1c07400ca23640c98d25f171caed9dcb1074920f87a6e` | `08ca6b1c2d786e3bf7cd5db7b20efcc474e06b875ddb66a2dbc6e9a2f74fcb09` |
| top air 3 | Eb | `175a98e8dbbf6876bcc770862da605e596e3916d3feb3a470383276b8a6ac4fc` | `aafb55e5185420f8b10bed3104d0fb1ce1d84befc5dae12ac93fea3992bb4d17` |
| top air 4 | Ea | `0193caaf794ecf493618c4003e930dcfa7f32a66b4e9ea66ec3acaac18b1f58e` | `12686df4f63cd344591e6ff8387a70bfef14c371de4aa5bda140e396bd080bda` |
| top air 4 | Eb | `dcb9328ddb1a62e119ac823516230c03c08bc382d2955b12ce7384f3bc9b2dca` | `3c18054538b54ea4d3f8c6f7bf1c6723d0876740eaaf10d4a33a30d2311b8b46` |
| combined 48/30/3 | Ea | `eedc9487fab6e910d1371ed55b66152160d27221f9fb893968791962942524e7` | `4ae6078ed264764171e78ae5c1d462366f974b2a7468d863c0829a9dced0eb25` |
| combined 48/30/3 | Eb | `28679c9b417d83e5c25aaa97c6cc5c5b47db15e23947eb5fb7cca0ef7e949c71` | `9bc3dd16dee24bed32ff8e939a97c635827f5eedaa8c3a263db7fb7b36b6609b` |
| combined 64/40/4 | Ea | `8c215909c7e1e2a8a76a7fa14f723b5034ad434ab4fb0ac858f25683e61748dd` | `c3d78a6d293149d3c1353db52c1b13fd00924cc38f3683425ec5c49fb7691407` |
| combined 64/40/4 | Eb | `e09e2482ba242a283ff98be92b2f01dc4f384fbdacce79c3b337aede7f770ba4` | `2a1ee6c322ce4c305bf08cc5882754c03a1c7631d211927c1cc96a023f13bbcb` |

Certificate root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_domain_certificate_a7d5a52a/`

- certificate: `FDTDX_FROZEN_Q_THERMAL_DOMAIN_CERTIFICATE.json`
- SHA-256:
  `2402be4a0b669c24acfaf5167cb9a5917edef65a2c2ab4a342fcc185c6bd4ef1`
- generator commit:
  `a7d5a52a7c0eac50156748db082aee031510be14`
- generator SHA-256:
  `10fe12f2e089b3b2f3427516e1202d3fb58a35445951056fde528061816698af`
- status:
  `VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_DOMAIN_SIZE_CONVERGENCE`
- ready: true
- production mesh selected: false
- optimizer start allowed: false

No raw NPZ, image, log, or iteration artifact is committed to Git.

## Remaining blockers and next actions

1. Keep x/y factor 2, z factor 2, and the 48/30/3-um domain fixed only for
   subsequent frozen-Q prototype diagnostics.
2. Separate domain-size convergence from boundary-model uncertainty.  Sweep
   side/bottom boundary models and distances where physically meaningful,
   top convection, and interface conductances without changing Q.
3. Obtain the real flake outline/thickness, crystal-axis angle, electrode and
   pad polygons, signed terminal assignment, substrate/oxide stack, and the
   intended electrical role of patterned Au.
4. Replace the current ideal full-edge rectangular terminal abstraction with
   the actual device and then run electrical pitch/contact/floor convergence.
5. Do not start inverse design until an independent Maxwell route, actual
   device thermal/electrical models, and complete coupled AD-FD all pass.
