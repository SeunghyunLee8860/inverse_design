# Au sampled-data MCM convergence findings

Date: 2026-08-24. These are local RTX 6000 Ada development results from
Lumerical v261 solver `8.35.4413`; they are not B200 promotion evidence.
Raw FSP/NPZ/JSON files remain outside Git under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.

## Controlled comparison

All rows used Ea, the same calibrated source hash, CV0, a 100 nm x/y flake
mesh, 5 nm thin-stack z mesh, 50 nm bulk/air/PML z limit, the same exact full
8 x 8 x 0.05 um Au rectangle, and the same Ordal input table. Only the maximum
number of Lumerical multi-coefficient-model terms allowed for Au changed.
TaIrTe4 and SiO2 retained their prior fitting settings.

| Au max coefficients | max Au fitted-epsilon error | max Au finite-dt error | native Q (W) | six-face absorbed power (W) | Q/flux closure | complex endpoint-field NRMSE vs MCM6 | gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 0.13966% | 0.14085% | 1.020401321e-15 | 1.020858744e-15 | 0.04481% | 1.462e-5 | pass |
| 6 | 0.09295% | 0.09413% | 1.019653272e-15 | 1.020565109e-15 | 0.08935% | 0 | pass |
| 8 | 0.09479% | 0.09596% | 1.020070199e-15 | 1.020419755e-15 | 0.03426% | 1.422e-5 | pass |
| 12 | 0.08139% | 0.08171% | 1.019715077e-15 | 1.020363462e-15 | 0.06354% | 1.568e-5 | pass |
| 16 | 0.08139% | 0.08171% | 1.019715077e-15 | 1.020363462e-15 | 0.06354% | 1.568e-5 | pass |
| 20 | 0.03711% | 0.03685% | 8.735757495e-16 | 1.233154449e-15 | 29.15926% | 3.5165e-2 | fail |

The 4--16 results form one field/Q plateau. The 20-coefficient fit has the
smallest pointwise readback error yet selects a different time-domain material
model and fails energy balance. Pointwise `getfdtdindex` and
`getnumericalpermittivity` agreement is therefore necessary but not sufficient.
The exact control must also pass native-Q versus a six-face Poynting balance.

The same 29% failure persisted across CV0, CV1, staircase, 5 to 2.5 nm z
refinement, and 1 ps/1e-7 to 2 ps/1e-9 decay settings. Exact-empty controls
remained near 0.015--0.016% closure. This excludes the closed box, background
stack, simple z resolution, mesh-refinement method, and material transient as
the root cause.

Ansys warns that excessive coefficients can make sampled-data fitting
sensitive to data features and recommends inspecting max coefficients,
passivity, stability, and the fit band:

- https://optics.ansys.com/hc/en-us/articles/360034915053
- https://optics.ansys.com/hc/en-us/articles/360034915033

## Selected setting and remaining endpoint issue

Au now defaults to six maximum MCM coefficients. Six is inside the measured
stable plateau and matches the ordinary Ansys default complexity. This is a
numerical material-model setting, not an optimization variable and not a
change to Ordal Au physics. The CLI retains `--au-max-coefficients` and
`--au-fit-tolerance` so every promotion rerun can reproduce the sweep.

The rho=1 `importnk2` carrier also passed Q/flux closure (0.04374%), but it is
not yet an exact endpoint match. Its realized Au Yee epsilon was approximately
`-846.877+130.992i`, while exact MCM6 realized
`-830.402+127.352i`. The resulting complex endpoint-field NRMSE versus MCM6
was 1.849%. Official `importnk2` accepts spatial n,k only, not a frequency
axis, so the continuous carrier still needs a mesh/time-specific realized-
epsilon calibration and an independent exact-binary MCM6 final reevaluation:

- https://optics.ansys.com/hc/en-us/articles/360034408694-importnk2-Script-command

Do not claim that the current imported rho=1 endpoint is exact Au, and do not
rescale Q or fields to hide the difference.
