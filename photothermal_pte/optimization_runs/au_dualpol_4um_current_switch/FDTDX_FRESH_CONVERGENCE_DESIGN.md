# Fresh FDTDX convergence design for exact-binary Au

Status: **L500 time/Courant controls validated; the full-domain-z ladder and z16 t32 extension are rejected; fine-dt float32 ADE transient precision is the blocker; no spatial-mesh claim and no optimizer permission**

This document defines the next FDTDX work after the four empty/full endpoint
controls.  It is deliberately independent of the Lumerical work in progress in
another session.

## What the endpoint matrix established—and did not establish

The validated four-case matrix established that the pinned FDTDX executable can
run the anchor grid for `Ea` and `Eb`, that empty and full masks remain exact
binary endpoints, and that the recorded optical energy/flux/stationarity gates
pass for those four controls.  Empty/full masks contain no fabricated edge,
minimum linewidth, minimum gap, or re-entrant corner.  They therefore cannot
certify the design-window x/y mesh, the external-domain mesh, the z-domain
extent, or a nontrivial antenna geometry.

The anchor is 196 x 196 x 160 Yee cells.  Its Au thickness is 50 nm and is
represented by eight z cells (6.25 nm per cell).  With the locked 4 um Au value
`n = 2.2 + 28.9i`, the amplitude skin depth is about 22.0 nm and the intensity
1/e depth is about 11.0 nm.  This is a useful scale comparison, not a convergence
certificate.  In particular, the anchor design-window pitch is still 100 nm.

## Why the new references are exact geometries

The relevant device papers use fabricated Au resonator or contact geometries and
vary physical dimensions, placement, or orientation.  They do not justify three
different fictitious Au densities for optical, thermal, and electrical physics:

- The 2022 full-Stokes PTE detector uses patterned 50 nm Au antennas and obtains
  the geometry by global/parameter optimization; absorption is evaluated from
  the physical lossy material field.
  [Nature Communications 13, 4408 (2022)](https://www.nature.com/articles/s41467-022-32309-w)
- The 2024 directional PTE work uses exact T, inverse-T, and propeller resonators
  and studies physical length, width, spacer, and rotation parameters.  It also
  reports sensitivity to rounded corners and fabrication-size deviations.
  [Nature Communications 15, 7117 (2024)](https://www.nature.com/articles/s41467-024-51599-w)
- The TaIrTe4 device analysis uses physical Au contacts and a device-specific
  weighting field; contact edges and crystal/electrode geometry affect the
  measured current.
  [Blevins et al., arXiv:2602.14959 (2026)](https://arxiv.org/html/2602.14959)

These papers motivate an exact-shape campaign.  They do **not** certify the
present FDTDX mesh, and this document does not treat them as mesh evidence.

The primary spatial reference is
`l_shape_4um_with_500nm_arms`: two 4 um by 500 nm arms joined at a corner.  It
contains straight edges, two orientations, outer corners, and a re-entrant
corner at the requested 500 nm DFM scale.  The separate
`parallel_bars_4um_by_500nm_with_500nm_gap` reference stresses the 500 nm void
gap.  Empty/full remain endpoint controls; x/y bars screen orientation bias.

## Ordered optical-only campaign

Every comparison uses both `Ea` and `Eb`, the same physical reference geometry,
raw complex fields, component-specific Yee dual volumes, separately refitted and
read-back dispersive materials, and a source-pair certificate for that exact
numerical contract.  Per-polarization power rescaling is forbidden.

1. **Time settling at the anchor spatial grid.** Hold Courant at 0.5 and compare
   16, 24, and 32 optical periods, retaining two successive comparisons.  The
   startup and phasor windows remain four periods.  This explicitly checks the
   long-time behavior that the endpoint control alone could not establish.
2. **Time-step convergence.** At the selected settled duration compare Courant
   factors 0.5, 0.375, 0.25, and 0.1875. Refit/read back ADE material
   parameters and regenerate the source pair at every level.
3. **One spatial axis at a time.** Compare two successive pairs on each ladder:
   full-domain z resolution; design-window x/y resolution; outer flake/gap x/y
   resolution; lateral PML x/y resolution; bottom Si buffer; top source-to-PML
   gap; lateral material-to-PML gap; lateral PML thickness; and z-PML thickness.
4. **PML sensitivity.** Repeat the selected spatial contract with CPML alpha
   scales 0.5, 1, and 2.
5. **Joint confirmation.** Assemble the individually selected levels and compare
   that joint mesh with a deliberately finer feasible confirmation mesh.  Axis
   sweeps are not assumed additive.
6. **Reference rechecks.** Recheck empty/full, x/y orientation bars, and the
   500 nm gap-stress mask on the selected contract.  A candidate design is not
   admitted until this stage passes.

## Completed L500 time-settling certificate

The first campaign stage is complete on the fixed `196 x 196 x 160` anchor at
Courant 0.5. All raw artifacts remain outside Git under
`/home/seunghyun200/fdtdx_results/l500_time_settling_01a8ad8a_20260824`.
The fail-closed certificate was generated from clean commit `5e376ce1` at
`time_settling_certificate_5e376ce1/FDTDX_FRESH_TIME_SETTLING_CERTIFICATE.json`;
its SHA-256 is
`20ab99b8488606475d2ed8d604d1810c9f3953176b68f42ed0689685ed505ab0`.
All 21 top-level gates and all five selection gates passed. The certificate is
explicitly not a mesh certificate and keeps optimizer start forbidden.

The canonical case-file SHA-256 values are:

- 16 periods: `8e0d3d547c5f35b23738046f5c6ff1e0f85c6c6cc05929dd7cd928bf7e6b232f`
- 24 periods: `7ea87f6ffdf1f9557dfae2fca9c65b710df95feb4da50b43f117dff8e14955d7`
- 32 periods: `d4f673e6574543aa5057cef9f0e8fcd6ae357b16ef577ba38ef9813f53c2c266`

The matching source-pair certificate SHA-256 values are:

- 16 periods: `de14efff04778a4fea7aeaaded88c53822087b864d77d17acfe3d97185692de1`
- 24 periods: `2e1edd6f05f8798e368d4f19463204bd31dff3ee5293d3e53680ffde77c0566b`
- 32 periods: `78439db4caa795c3c3e1d08d26c4cef02c61297f596e752bb197802f0ea08d06`

The 16-period material cases were correctly rejected as unsettled. `Ea` failed
complex-field stationarity at `1.1229036e-2`; `Eb` failed field stationarity at
`1.7813813e-2` and previous/late spatial Q at `5.5404220e-3`. Both 24- and
32-period Ea/Eb cases passed every internal material/energy/flux/stationarity
gate. Independent NPZ reload then produced these worst-polarization successive
metrics:

| Optical metric | Limit | 16 to 24 | 24 to 32 |
|---|---:|---:|---:|
| source power relative change | 5e-3 | 5.7604523e-7 | 3.4562714e-7 |
| Q/closed-flux relative | 2e-2 | 7.8704948e-4 | 7.7062780e-4 |
| fine-case complex-E stationarity | 5e-3 | 3.4493703e-4 | 2.4179021e-4 |
| total Q relative change | 1e-2 | 2.2200505e-5 | 1.5007209e-6 |
| material/Cartesian-component Q max change | 2e-2 | 1.4088296e-3 | 2.4009730e-4 |
| fixed 8 x 8 um probe complex-E NRMSE | 2e-2 | 3.7799424e-5 | 6.0565673e-6 |
| component-Yee-volume Q L2 NRMSE | 5e-2 | 7.5788175e-4 | 1.0268401e-4 |

The certificate independently re-hashes all three canonical cases, all three
source pairs, all six material reports and NPZs; reconstructs the exact
375-pixel L500 mask; reconstructs component-specific Yee volumes from grid
edges and placement; recomputes Au/TaIrTe4 powers and previous/late field and Q
metrics; and compares only the fixed physical `[-4,+4] um` probe at
`z=0.250 um`. Because this ladder holds the spatial grid exactly fixed, no
interpolation or conservative remap is necessary; exact common physical cells
are used.

The selected minimum settled duration is therefore **24 periods**, with
**32 periods** as the required confirmation. The completed Courant stage below
uses that 24-period selection. Optimization remains forbidden.

## Completed L500 Courant certificate

The second campaign stage is complete at 24 periods on the same fixed
`196 x 196 x 160` spatial grid. Raw artifacts remain outside Git under
`/home/seunghyun200/fdtdx_results/l500_courant_4d79a439_20260824`. The
fail-closed certificate was generated from clean commit `876cfff3` at
`courant_certificate_876cfff3/FDTDX_FRESH_COURANT_CERTIFICATE.json`; its
SHA-256 is
`7fd86bc8582d27002c226b6395a7d803f29ba98deda4abff00e60def9560a869`.
All top-level and selection gates passed. It is a time-step certificate only:
`is_mesh_certificate=false` and `optimizer_start_allowed=false`.

The canonical case-file SHA-256 values are:

- Courant 0.5: `7ea87f6ffdf1f9557dfae2fca9c65b710df95feb4da50b43f117dff8e14955d7`
- Courant 0.375: `9a75f96cc331d500ec9fb9eb63f6cdecef2d737063b06e4021d96cc919a8f5af`
- Courant 0.25: `6ecf2ccbd3b4b27b33eb7c9f70d788532197c3a4e66ef1f62eb5c1779454dffe`
- Courant 0.1875: `5f4aac85143bcd624f0a9f13bec90879fad9d45f1fc50ef0d10faea39c56eb60`

The matching source-pair certificate SHA-256 values are:

- Courant 0.5: `6fae7e958d22ae7aea580d9e74d94371bcc93e0934b2173a660136a7929cbd0f`
- Courant 0.375: `923a6d3814c8c1d4e8ecf00623f06becabc0a7248aa318656f90c9f2b1536863`
- Courant 0.25: `e4a6898b86db1d6418924c202272719e22a62c75adc71de7d46ea551f08c748b`
- Courant 0.1875: `f5e7d4239c772f92bee03930d4bd9fe9979b8f2e94266c105759a00a283da26b`

The first three levels were run at clean commit `4d79a439`; the 0.1875
extension was run at `14624869`. The certificate records this rather than
claiming one run commit. Its cross-commit audit proves that only the Courant and
time-settling certificate/test files changed between those commits; the material
runner hash, source-pair generator hash, material contract, pinned FDTDX source,
and runtime lock are identical, and source/material commits match per level.

| Optical metric | Limit | 0.5 to 0.375 | 0.375 to 0.25 | 0.25 to 0.1875 |
|---|---:|---:|---:|---:|
| source power relative change | 5e-3 | 8.0646341e-7 | 6.9125443e-7 | 1.9585542e-6 |
| Q/closed-flux relative | 2e-2 | 2.5001969e-3 | 2.5001969e-3 | 2.6334778e-3 |
| fine-case complex-E stationarity | 5e-3 | 4.5087853e-4 | 8.9994775e-4 | 1.3745917e-3 |
| total Q relative change | 1e-2 | 2.2341408e-3 | 4.0722149e-4 | 2.0097064e-4 |
| material/Cartesian-component Q max change | 2e-2 | **2.3390840e-2 fail** | 8.9096488e-4 | 8.0774269e-4 |
| fixed 8 x 8 um probe complex-E NRMSE | 2e-2 | 6.7899555e-4 | 5.0716420e-4 | 2.4243411e-4 |
| component-Yee-volume Q L2 NRMSE | 5e-2 | 5.4550187e-3 | 5.4008614e-4 | 8.1493446e-4 |

The coarse 0.5-to-0.375 pair remains explicitly rejected because the worst Au
Cartesian Q component changed by 2.339%, above the 2% gate. The next two
successive pairs both pass every declared gate, establishing the finer
asymptotic range. The selected Courant factor is therefore **0.25**, confirmed
by **0.1875**. The next allowed FDTDX action is the exact same L500 reference
on the full-domain-z resolution ladder at 24 periods and Courant 0.25.

## Rejected first L500 full-domain-z ladder

The first spatial stage was executed on clean raw-run commit `150a7592` at 24
periods and Courant 0.25. Only the full-domain z factor changed: `z2`, `z4`,
and `z8` used grids `196 x 196 x 80`, `196 x 196 x 160`, and
`196 x 196 x 320`. Their Au pitches were 12.5, 6.25, and 3.125 nm;
TaIrTe4 pitches were 10, 5, and 2.5 nm. Every physical Si, SiO2, TaIrTe4,
Au, air, source, and PML boundary remained fixed. Raw artifacts are external at
`/home/seunghyun200/fdtdx_results/l500_full_z_150a7592_20260824`.

The canonical case-file SHA-256 values are:

- z2: `52f43abd1355fbccda4d289acd68d59c53ec5f7679d710cae408a6b2ef12e7d0`
- z4: `6ecf2ccbd3b4b27b33eb7c9f70d788532197c3a4e66ef1f62eb5c1779454dffe`
- z8: `1d35f6e603c1983e5ed87a16e752d8e5d2ff971b95278d7e6429b31ba35b17c4`

The matching source-pair SHA-256 values are:

- z2: `bbeb96f07b3da5c1a933e46d3e9f66c4d621cc248f86a2e3830ec68c39f88282`
- z4: `b59834591bca878d4200b8a3841f524d7584b9bcd06516e935ffa0c464eccf18`
- z8: `1b3e91c9d444cda6fbfb89cf097cbd85ea8aadb260ba7e05ad80722a62b97d66`

The corrected fail-closed certificate was generated from clean commit
`7b687684` at
`full_z_certificate_7b687684/FDTDX_FRESH_FULL_Z_CERTIFICATE.json`; its
SHA-256 is
`319743a29b8dd4869c5d1feedf564850ff10e4c30fb1888fd28eb7ed8764036c`.
The correction applies the exact binary solver mask to the Au field comparison
so design-window air is excluded, and treats the observed float32 representation
of an exact 0.90 common-support fraction as numerical roundoff. All source, raw
NPZ schema/hash/grid-coordinate, placement, material, repository, CFL, end-time,
mask, conservative-remap, and optimizer-forbidden gates pass. The certificate
is blocked only by both physical z comparisons.

| Optical metric | Limit | z2 to z4 | z4 to z8 |
|---|---:|---:|---:|
| source power relative change | 5e-3 | 2.7183581e-3 | 6.7789117e-4 |
| Q/closed-flux relative | 2e-2 | 2.3665178e-3 | 5.5250911e-3 |
| fine-case complex-E stationarity | 5e-3 | 8.9994775e-4 | 1.7502175e-3 |
| total Q relative change | 1e-2 | **3.7549030e-2 fail** | **1.8457622e-2 fail** |
| material/Cartesian-component Q max change | 2e-2 | **7.8486582e-1 fail** | **3.7719972e-1 fail** |
| fixed 8 x 8 um tangential-probe complex-E NRMSE | 2e-2 | **1.3049050e-1 fail** | **6.8821593e-2 fail** |
| conservative component-Yee Q L2 NRMSE | 5e-2 | **2.6010590e-1 fail** | **1.3924790e-1 fail** |
| exact-material-region complex-E NRMSE | 5e-2 | **7.8508702e-1 fail** | **9.3868486e-1 fail** |

For z4 to z8, Ea total Q changes by 1.8458 percent while Eb changes by
0.6405 percent. The exact-Au field NRMSE is 71.58 percent for Ea and 93.87
percent for Eb; TaIrTe4 is 6.27 and 5.04 percent. Some large Cartesian relative
changes are attached to components carrying less than 0.05 percent of total Q,
so those diagnostics must be interpreted with their absolute power fractions.
That caveat does not rescue the ladder: total Q, fixed probe, conservative
spatial Q, and exact-Au field gates independently fail.

No z level is selected, `is_mesh_certificate=false`, and
`optimizer_start_allowed=false`. Do not proceed to x/y convergence, thermal or
electrical promotion, or optimization. The next FDTDX action is a finer
full-domain-z extension under the same exact L500, 24-period, Courant-0.25
contract; failed z2/z4/z8 results must remain visible rather than being relabeled.

## Blocked z16 extension: float32 ADE precision

The canonical z16 contract was created at commit `8766b3c6` with file SHA-256
`74fca414c3c82ce1031f0f688cab0c3a3d252de6ea66e2fceb22ee40c0493e3a`
and internal canonical SHA-256
`568617c82617b04b45753650933e44330aaa7a1bcde59822db56dceb94a801c6`.
It resolves a `196 x 196 x 640` grid, including 1.5625-nm Au and 1.25-nm
TaIrTe4 cells. The source-only Ea preflight stopped before any field solve with
`realized float32 ADE refit error 0.000117579 exceeds 1e-05`. The failure JSON
SHA-256 is
`0a302d01386faf0967b1626d1646fa082a5a52eb23c8768b3d09bad1e5cb4631`.
No z16 source pair, material solve, or mesh comparison was created under that
rejected single-pole extension.

`fdtdx_fresh_ade_precision_diagnostic.py` was committed at `a4cf66d5` and run
from that clean commit. Its external diagnostic is
`ade_precision_a4cf66d5/FDTDX_FRESH_ADE_PRECISION_DIAGNOSTIC.json` under the
same raw root; SHA-256
`bfa98e74b81eae816b888bfbe1b460f94d5cf407f4be4954742c91e2b540911c`.
It reproduces the JAX-x64-disabled float32 edge realization before computing
the rectilinear CFL step. The reproduced z8 step is exactly
`2.083469563193086e-18 s`; z16 is `1.0422198660912219e-18 s`.

| ADE carrier check | z8 relative error | z16 relative error | 1e-5 gate |
|---|---:|---:|:---:|
| current single-Drude phase search (0.8 to 1.2 gamma seed) | 1.7063374e-6 | 1.1757867e-4 | z16 fail |
| wide single-Drude realized-error scan (0.01 to 10 gamma seed) | 1.7063374e-6 | 2.2144332e-5 | z16 fail |
| stable positive-strength two-Drude numerical candidate | 7.2571180e-9 | 7.2571180e-9 | candidate only |

The wide Au scan shows that merely widening the current damping search does not
close z16, so the 1e-5 gate must not be relaxed and the failed source directory
must not be overwritten. A full-tensor follow-up was committed at `ecc33c22`.
Its clean-commit external JSON is
`ade_precision_ecc33c22/FDTDX_FRESH_FULL_MATERIAL_ADE_PRECISION_DIAGNOSTIC.json`,
SHA-256
`cb15e83073887fc0b7bd328f81b1b5463087024d98277bd740027bd82a412741`.
It proves the first Au exception was not the only pending failure:

| material axis | z16 current single-pole error | z16 pass | z32 current single-pole error | z32 pass | stable two-pole candidate error |
|---|---:|:---:|---:|:---:|---:|
| Au | 1.1757867e-4 | no | 1.1757867e-4 | no | 7.2571180e-9 |
| TaIrTe4 a | 2.7593129e-5 | no | 4.0745430e-5 | no | 2.1516201e-8 |
| TaIrTe4 b | 2.5155545e-6 | yes | 2.3850798e-5 | no | 2.1030075e-8 |
| TaIrTe4 c | 2.5155545e-6 | yes | 2.3850798e-5 | no | 2.1030075e-8 |

The canonical candidate-law generator was then committed at `f959a9ef` and run
from that clean commit. The z32 case is `196 x 196 x 1280` (49,172,480 Yee
cells), with case-file SHA-256
`33398486f542fa0f1c7b063011e61992f7830b7cd36c25c8d6863c553aa3fbf4`
and internal case SHA-256
`a10b3a9fa2757d51b51ee0726566feb75c5516a1ed3728dcc423ebc8b14d3125`.
The external candidate-law files under `two_pole_material_f959a9ef` are:

| z factor | material-law file SHA-256 | internal material-law SHA-256 |
|---:|---|---|
| 8 | `6352e58e0b3b2449f5316948adb3247bfc9c71547cbb2252a8beba69571d67bc` | `700be9659c9031a5ed69fd6b9c3eca637b508ebdef8b61d7082c3ac644438ba7` |
| 16 | `558eae569446993096081320c1f6e9439ee78ef799c8aeb0b0af8810a72e6fb2` | `5fee80f9295483fd5022eafb6bcabaa2e96934a908a45fe4581712c1cb91f7b5` |
| 32 | `302ab4e8991b55d0fb17c2ff5332b156fb29401ac04e026e0394f7e6c1fbcd1d` | `ee428f0c2ec4402ae397d208e2c6773221f55980579fc160c9c7996d256a2ed4` |

Each file binds the numerical case, original material table, algorithm source,
and pinned FDTDX update/dispersion hashes. All remain `candidate_only=true`.

The pinned-library coefficient preflight was committed at `7504045c`. It
reconstructed every law, instantiated its physical Drude/Lorentz poles through
pinned FDTDX, and obtained bit-exact float32 c1/c2/c3 with zero c4 for all four
material axes. Clean external preflight file SHA-256 values are:

- z8: `1b892395e5d989dcb12a679d0d0c19389d2017cff026326670fd9a074cf0aeb2`
- z16: `aa91d260271982f2bf3c4aba523cda6127a18ab6ddac50514628fbe8f5f59a9f`
- z32: `4f5e3da15bcbc571fd8c9d98bc30ca4be4f5b5ac8b3266efe78a01d93d9202b6`

This is a pinned coefficient-generator certificate only, not a placed solver
array, time-domain material, field, or mesh certificate.

The candidate law then passed a second, stronger placement gate at clean code
commit `011e0d36`. `fdtdx_4um_model.build_model` keeps its historical
single-pole path as the default and accepts the two-pole law only through an
explicit argument. It recomputes the law self-hash, checks exact dt and target
epsilon, instantiates the physical poles, reruns pinned FDTDX coefficient
generation, and requires bit-exact float32 endpoints before object placement.
The exact-binary material path now fills and reads back every pole rather than
only pole zero.

External zero-time-step placed-array reports are under
`two_pole_solver_array_011e0d36` with SHA-256 values:

| z factor | grid | placed-array JSON SHA-256 |
|---:|---|---|
| 8 | `196 x 196 x 320` | `126ac0aa053cc31c576700f1527e8a6f9a9d1dbdbda433bf7b38df13f272ec5c` |
| 16 | `196 x 196 x 640` | `28887e54bf29b51f818b962e0072fb774cb496e9bf19cfcf4c0bf858c9d0465c` |
| 32 | `196 x 196 x 1280` | `adfa0e0332bc487df31296b7116ea757f501a408f73e77939cf989ec68c74266` |

All reports pass exact case/law/dt/PML/mesh binding. The c1/c2/c3 arrays have
leading shape `(2,3)` for two poles and three Cartesian components; c4 is
absent. Au uses exact binary air/Au endpoints on every pole, TaIrTe4 preserves
`Ex->b`, `Ey->a`, and `Ez->c` on every pole, and Si/SiO2 inverse permittivity
readback passes. Every realized epsilon is passive and within `3.1e-8` relative
error of its original target.

A distinct candidate-bound source/pair path was committed at `e722ba73` and
first exercised at z8. Both all-air polarizations pass every source gate. Ea and
Eb incident powers are `1.8834239723e-12` and `1.8834237555e-12 W`, giving a
relative mismatch of `1.1513098e-7` against the `5e-3` limit. The pair uses one
common power scale and one common field-amplitude scale; polarization-specific
normalization remains forbidden.

External z8 artifact SHA-256 values under `two_pole_forward_e722ba73/z8` are:

| artifact | SHA-256 |
|---|---|
| Ea source report | `30cbc8b18c5aaaa289994b5bafe2c7b8821983aff31b7f97e7c9647d4b113901` |
| Ea source NPZ | `7e586000eb4a5681011062f9fe78e972120a8b0bb9b05c3eda55fd24f326d133` |
| Eb source report | `49788884d2ce62660cfba923a123940426211f8394fe98c6103a7058782bf459` |
| Eb source NPZ | `93cff39dbeaf200f0db43987c077b700ae1713774c71fc480bc7c982f8e393e1` |
| original `e722ba73` law-bound source pair | `d5196cc7c715260e5c0436ccc59ca258c22173ae9adc69340405dc5e1e05a582` |
| current pair after `7c527b6d` generator hardening | `28c84ac1b2c21cb6d6db537248f90101ca2add6b37aef49c002da0b7b214fa64` |

The candidate exact-binary material runner was added at `7c527b6d`; it accepts
only the current candidate source-pair status, reconstructs the case/law, reruns
placed material readback, and is forward-only. Both z8 polarizations pass. The
raw report/NPZ SHA-256 values are Ea
`215e8eaf37788623dd4c0f98f9e6661f462b49618867cfd572d8ccf10fb978c8` /
`61ffeec5dd24d2bcc85f5bf83e8bd692af90ec7789a062797b21b6ea216d9cea`
and Eb
`b29f9832f601079737f59a2953bcf7332f4eb23948e6a7ddaaceee55a6615c02` /
`54a586db06fa9deaa058bae48d3510e381db3b3824140c571dee56e92fc2b8c4`.

The independent pair verifier at `b489bbdc` reopens both 101 MiB NPZ files,
reconstructs the canonical 375-cell L mask, rejects gray/rho-power material
laws, recomputes component-Yee-volume Q, and revalidates every case, law,
source-pair, implementation, and raw hash. The clean certificate SHA-256 is
`ac71dd8e786ee3ca59d86f233f3d20b980bf3796abbc3a2b30167d9bae78b3d5`.
All gates pass. Common-285-uW total absorption is `67.3793 uW` for Ea and
`116.4644 uW` for Eb; the Eb/Ea ratio is `1.72849`. Maximum field stationarity
NRMSE is `1.3853e-3` and `1.5225e-3`, and closed-flux Q mismatch is below
`0.52%`.

This establishes one z8 two-pole optical material pair, not z convergence and
not PTE current. The independent z16 source-only pair is also complete under
`two_pole_forward_e722ba73/z16`: Ea report/NPZ hashes are
`9b6ced4e90c912a4ce00b99803b63645d4e85ec1a10a8f7c90ca6d9747298695` /
`40a2f60caa2462f71c91a59a699f1e212204a49e654b0f97a6e2391b4bd28632`,
and Eb hashes are
`3e9d1d07f68c53908e52678fd42ae4068286cece7f3ba4542c7bff6b1465045d` /
`836a425b661c3aee8ec949cf7569ce4f55a1b2b389766fcb7eac5a86d734dd72`.
The clean pair certificate SHA-256 is
`7cfcd8280cf63194aa53f328661613dd942e2b0da0f4045f2b4d2b8f881c7d35`.
Its source-power mismatch is exactly `0.0`; both field-stationarity NRMSE values
are below `2.98e-6`.

The 24-period z16 material Ea/Eb runs were subsequently completed and both are
blocked before mesh comparison. Ea report/NPZ SHA-256 values are
`62f30a998513636a6f04b47bc01a04fe03d0350e5bb785660b08d665a45a4bc6` /
`5c9091fab42e58dd226b010a9cea7a783da4f7e12344cd41f10e24261374eb1e`;
Eb values are
`403837ce5de0db120820013a5bbb09ff0146782b61a392c2b62966d8f8da939b` /
`033247f55ceafb50530c2dad5bdb09751f02a477ed034c59d3d2f7f92fa6566e`.
The current blocked-pair certificate SHA-256 is
`b590872c263b70d10e3355d936d41a8a74ef4e502a5957c5716054fa8c4f7b0c`,
generated at clean `eefae409`.

Ea field stationarity is `1.8074e-2`; Eb is `1.9032e-2`, both above `5e-3`.
Eb spatial Q stationarity is also `5.2720e-3`, just above `5e-3`. Both closure
mismatches remain below `0.54%`, all material/source/canonical/raw gates pass,
and total Q is stable. Component analysis identifies unsettled z-normal field
rather than the driven in-plane component. This is a 24-period settling
blocker, not a material or energy-conservation failure.

The next step is a separately hashed longer-time z16 case and its own source
pair; the existing 24-period pair cannot be reused. No z8-to-z16 comparison is
accepted, and z32 still requires independent source and material pairs.
Comparing old single-pole z8 directly with a new two-pole field level is
forbidden. Adjoint, thermal/electrical, current-sign, and optimizer stages
remain blocked.

The 32-period z16 settling case and zero-step solver-array preflight are now
complete at code commit `84461793`. The case file/internal hashes are
`6476b57bd577bcba0106e42c85ceb1707256384ff2d6a41824e3a2a3de47ba2f` /
`0c30a5c68efb3b4a79fbd248db104919a439a539e5eff84e5b10f8bfbd6ab07f`;
the material-law file/internal hashes are
`717f5ed3d24c33ebd4f870b108a4b0c618e87aabc7144991207c18db9e0ced31` /
`d4d140b09e624c5140f72778865fc9df60f8a79c2c0690de2b4d01ebf008cd70`.
The grid remains `196 x 196 x 640`; the realized t32 duration is `409,666`
steps. The t24/t32 material-axis payloads are exactly identical, hash
`445b5bf65eae93c5778edc8ee98b7abae4117bebec40937abdcaa494d08bb7aa`.

The placed-array preflight file SHA-256 is
`e0e992e1fdaf4edfcb9f96842759ed1b2410b4f293516bd5067dc15021ab2a1b`;
it is `ready=true`, has no failed checks, and has internal payload hash
`be6a3c2112ec84b7f4ffa274fa68731b62d2159782ebe6816b4cddcde04f982b`.
This closed only the t32 contract/readback gate; the t24 source pair remained
invalid for this case. The fresh t32 all-air pair was then completed at clean
commit `1c7cd8ee`. Ea report/NPZ SHA-256 values are
`beefc073c1f0403010858502ea42de452308e554f7847d3283173e495e5eef66` /
`20b54b16d9f9a634a3a97a0a938e8cadc3e1d4f343496500b8160a228d481573`;
Eb values are
`0d565cfee7c435f46f8891c26dbd296bae04531b88dbe4170bb5af11e0026e9e` /
`50862803d45551c9d508d939ab77427557aaab5147b1576b5588f6a50911eaa4`.
The source-pair certificate SHA-256 is
`278dff85e307042d1b7d004316ac74be010fb40593179b65948bbdc878c4b7bc`.
Both source cases pass all gates with identical incident power
`1.883720176371062e-12 W`, zero pair mismatch, and field stationarity below
`4.02e-6`.

The matching t32 material runs were completed at clean `b662b07b` and both
failed. Ea report/NPZ SHA-256 values are
`337c7d6e07b8fa7da9cd8394a89c524ebbccef0a4bb00d0b0d1d69aecde965c0` /
`2184cdb2a263f59ce96a2acbe3f4a654461482ef938756def5d30f8c66e42275`;
Eb values are
`cdc4f1b1baa55c9153b314d8bfff7f35c851a235be7976f0efcc3bd12ae22317` /
`3ad02aee9af1214fdf27fc3aa519160945e7bc7d42f387245d7c477047c5be89`.
The blocked pair certificate SHA-256 is
`999a28f273c15ef86d43e77112ca877a1c449ea14710a90f561389c94abc757e`.
Ea field stationarity is `1.6197%`; Eb field and spatial-Q stationarity are
`2.6502%` and `0.6583%`. All raw/canonical/source/material/closure gates pass.
The 16--32-period Au-window changes are non-monotonic, so t40/t48 is not a
valid next action.

The CPU scalar-recurrence gate added at `2edb38d8` reproduces the failure
without FDTD. z8/t24 passes (certificate
`48e6780c39f4256eac0ba116460bc937dc62c34069bcf5f7be86e2122e70c4ce`),
while z16/t32 and z32/t24 fail (certificates
`426c067f4971edddd2435134d714efe2e20e6b78492c15207d3c8a83e4b3b191` and
`3397023337a48bc843eb28de38d82860a8567e9581b3c05a16eaae5c367176b4`).
For z16 Au, float32 window drift is `1.713%`; identical coefficients with a
float64 state settle to `4.93e-10`, and their late response differs by
`3.069%`. The cancellation-condition estimate is `1.66e7`. Therefore the
one-frequency carrier fit/root-radius checks were necessary but insufficient;
fine-dt repeated float32 ADE updates are the present root blocker.

The measured cost is `260 s/solve` for z8/t24, about `862 s/solve` for
z16/t24, and about `1145 s/solve` for z16/t32. Independent polarizations may
run on two separately verified-idle GPUs, but a single solve remains
single-GPU and forward/adjoint remains sequential. z16/t32 therefore has a
lower bound near `38 min/optimization iteration` even with two GPUs and is
validation-only. No finer mesh, longer time, adjoint, thermal/electrical, or
optimizer step is allowed until the ADE recurrence is reformulated and passes
the CPU transient gate.

## Comparison and promotion rules

`Q` comparisons must conservatively restrict fine-grid cell-integrated
absorption (`q * component-specific Yee dual volume`) onto common physical
control volumes.  Complex `E` comparisons must interpolate each staggered Yee
component to the same fixed physical probe coordinates before computing an
NRMSE.  Array-index comparison across different nonuniform grids is forbidden.

The optical gates cover incident/source consistency, closed-surface flux versus
absorbed power, late-time stationarity, total and material-resolved absorption,
fixed-probe complex fields, and conservatively remapped absorption.  Thermal
temperature, PTE current, and current sign are downstream validation stages.
They require a separately closed physical device/contact/weighting-field
contract and are intentionally absent from the optical mesh certificate.

Passing this campaign would permit evaluation of exact-binary candidate
geometries.  It would still not validate a gray `rho`, `rho^3`, or independently
interpolated optical/thermal/electrical Au model, and it would not by itself
authorize restarting the historical optimizer.


## Increment-state implementation gate

The solver-free cancellation-resistant `(P, delta-P)` candidate at commit `05d8e9ba` passes z8/z16/z32 CPU precision gates, but it is not yet implemented in FDTDX. See `FDTDX_INCREMENT_STATE_CANDIDATE.md`. The z ladder remains closed until an isolated fork passes actual JAX-kernel forward, long-time float32, and checkpointed AD-FD controls. Old and new material-law results must never be mixed in a mesh comparison.
