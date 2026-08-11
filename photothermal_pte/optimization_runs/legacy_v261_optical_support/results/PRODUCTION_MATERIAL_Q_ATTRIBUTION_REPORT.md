# Production material-intersection Q attribution

Status: `VALIDATED_PRODUCTION_MATERIAL_INTERSECTION_Q_ATTRIBUTION`

The native component Yee-cell heat source was integrated only over the
literal volume shared with each physical thermal material. A cut-cell's full
power was **not** forced into TaIrTe4 or another nearest material. No clipping,
smoothing, gain, or global rescaling was used.

| partition | power (W) | fraction of full P_Q |
|:--|--:|--:|
| Si | 4.794583044368e-16 | 0.657066% |
| bottom SiO₂ | 6.458371292767e-15 | 8.850776% |
| physical TaIrTe₄ | 2.821624264833e-14 | 38.668518% |
| design SiO₂ | 3.693513958218e-14 | 50.617196% |
| artificial background | 7.530336520049e-18 | 0.010320% |
| air/cut-cell remainder | 8.728060400471e-16 | 1.196124% |

- full native P_Q: `7.296954820427e-14 W`
- native reintegrated P_Q: `7.296954820427e-14 W`
- material-attributed physical thermal source: `7.208921182771e-14 W`
  (`98.793556%`)
- reintegration error: `1.729732e-16`
- partition identity error: `0.000000e+00`

The artificial long-TaIrTe4 optical background outside the finite 32×32 µm
thermal flake contributes only
`0.010320%` of full
P_Q and is explicitly excluded from the physical thermal RHS. The
`1.196124%` air/cut-cell
remainder is reported rather than reassigned. Thus the thermal source is not
globally power-matched to the full optical control-volume P_Q.

This is an attribution gate only. It performs zero thermal, PTE, adjoint, or
optimization solves.
