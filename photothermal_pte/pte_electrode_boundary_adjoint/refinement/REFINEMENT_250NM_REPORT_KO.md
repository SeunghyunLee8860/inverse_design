# 0.25 µm refinement와 transition-width robustness

## 최종 판정

현재 결과를 두 문장으로 분리해야 한다.

```text
0.25 µm best-found electrode + transition-width robustness: PASS
0.5 -> 0.25 µm mesh convergence: NOT YET CONVERGED
```

즉 아래 geometry는 현재 계산한 것 중 가장 신뢰할 수 있는 0.25 µm best-found
design이다. 하지만 같은 geometry의 current가 0.5 µm와 0.25 µm 사이에서 corner
beam 기준 최대 13.0% 변했으므로, 이것을 최종 mesh-converged 수치라고 부르지는
않는다. 그 주장을 하려면 다음 thermal refinement가 한 번 더 필요하다.

## 1. 0.25 µm thermal field 재생성

기존 explicit conservative Cartesian FVM을 그대로 사용해 9개 Gaussian beam의
temperature field를 다시 생성했다.

- 구조: air / uniform rectangular TaIrTe4 / thermally-grown SiO2 / Si
- thermal cell shape: `122 x 122 x 36`
- electrical node shape: `97 x 97`
- 9개 beam 모두 별도 thermal solve
- 최대 linear residual: `9.88e-11`
- 최대 energy-balance relative error: `8.40e-12`
- 저장 field SHA-256 시작값: `d3f9c0547...`

`configs/per_beam_250nm.json`, `generate_250nm_fields.py`,
`per_beam_250nm_fields.npz`, `per_beam_250nm_thermal.json`에 설정과 provenance가
저장되어 있다.

## 2. 0.25 µm에서 Robin relaxation 재선정

0.5 µm에서 사용했던 `g=1e12 S/m2`를 그대로 가정하지 않았다. 0.25 µm model에서
Robin-to-hard 수렴과 adjoint-FD를 다시 확인해 production relaxation을

```text
g = 1e14 S/m2
contact discretization = nodal_lumped
```

로 정했다.

- validation geometry current relative error: `5.63e-4`
- weighting potential relative L2 error: `2.96e-3`
- 네 scaled 변수 adjoint-central-FD 최대 component error: `2.87e-5`
- 최종 transition audit 전체에서 최대 smooth-hard current error: `0.411%`

초기 diagnostic에서 썼던 `g=1e13`은 corner 후보와 좁은 transition에서
smooth-hard 차이가 1%를 넘었다. 따라서 `local_refinement_250nm.json`과 최초
`transition_robustness_250nm.json`은 문제를 발견한 diagnostic artifact이며 최종
ranking 결과가 아니다. 최종 결과는 `g=1e14` cross-seed 결과다.

## 3. 0.5 µm geometry transfer와 mesh effect 분해

0.5 µm plateau winner를 그대로 0.25 µm model에 옮겨 hard current를 평가했다.
추가로 temperature와 electrical discretization을 교차시켜 변화 원인을 분해했다.

- 최대 thermal-only change on 0.5 µm electrical mesh: `13.1097%`
- 최대 electrical-only change with interpolated 0.5 µm temperature: `0.2280%`
- 최대 combined same-geometry change: `12.9983%`

따라서 corner beam의 큰 차이는 electrode optimizer가 아니라 주로 0.5 µm thermal
discretization에서 온다. 이 때문에 0.5 µm 결과는 search plateau에는 도달했지만
mesh convergence에는 도달하지 않았다.

## 4. 0.25 µm local refinement와 cross-seed closure

각 beam에서 0.5 µm winner를 0.25 µm로 옮긴 뒤 `+I/-I` signed SLSQP를 별도로
실행했다. 모든 endpoint는 hard contact로 재평가했다. 첫 transition-width
audit에서는 width마다 서로 다른 basin을 선택해 `(-8,0)`과 `(+8,0)`에서 최대
3.91% spread가 나와 정직하게 `FAIL`을 기록했다.

이를 숨기거나 tolerance를 늘리지 않고 다음처럼 수정했다.

1. 모든 width에서 발견된 가장 강한 hard geometry를 공통 seed pool로 만든다.
2. terminal-swapped seed와 feasible perturbation을 함께 넣는다.
3. width `0.25, 0.50, 0.75, 1.00 µm`를 같은 pool에서 다시 최적화한다.
4. 어떤 width가 공통 seed보다 0.1% 이상 개선하면 그 geometry를 다시 모든
   width에 넣어 한 번 더 closure iteration을 수행한다.

첫 cross-seed sweep에서 네 corner beam은 0.180% 개선되어 두 번째 순환이
필요했다. 그 네 beam만 선택적으로 다시 돌렸고 모두 추가 개선 0, width spread
0으로 닫혔다. 최종 9-beam gate는 다음과 같다.

```text
status: PASS
transition widths: 0.25, 0.50, 0.75, 1.00 µm
maximum hard-current relative spread: 0.04866%
maximum improvement over latest common pool: 0.04868%
maximum symmetry-aligned geometry change: 0.2359 µm
maximum smooth-hard current error: 0.4107%
all SLSQP runs successful: yes
```

## 5. Boundary quadrature-order gate

orders `3,5,7,9`와 위 네 transition width의 Cartesian product를 9개 beam의 최종
geometry에서 검사했다. production contact는 Robin-to-hard limit을 맞추기 위해
`nodal_lumped`를 사용한다. 이 assembly는 boundary-node trapezoidal weight를
사용하고 Gaussian edge quadrature array를 의도적으로 사용하지 않는다.
따라서 이 gate는 asymptotic quadrature convergence가 아니라 production
implementation이 quadrature-order knob에 의존하지 않는다는 invariance test다.

총 144 case에서 order 9 기준으로

- current relative difference: `0`
- gradient scaled maximum difference: `0`
- weighting-potential relative L2 difference: `0`
- contact-integral relative difference: `0`

이어서 `PASS`다. consistent-edge quadrature의 수렴을 주장하는 결과는 아니며,
consistent-edge assembly는 앞서 hard-node limit과 일치하지 않아 production에서
제외했다.

## 6. 최종 0.25 µm best-found 결과

둘레 convention은 `s=0`이 bottom-left이고, bottom → right → top → left 순서로
증가하며 총 둘레는 96 µm이다. 표의 geometry는 `(c0,L0,c1,L1)`이고 모두 µm다.
전극은 top surface가 아니라 flake 외부 둘레에 붙는다. `corner=yes`인 contact는
둘레를 따라 인접한 두 side에 걸쳐 있는 L-shaped footprint다.

| beam (µm) | hard `|I|` (nA) | `(c0,L0,c1,L1)` (µm) | contact center sides | corner crossing |
|---:|---:|:---|:---|:---|
| (-8,-8) | 0.829863 | `(91.7035,8.3638,3.4427,6.1145)` | left / bottom | no / no |
| (-8, 0) | 0.677804 | `(85.4387,9.3907,0.6991,12.1300)` | left / bottom | no / yes |
| (-8,+8) | 0.829863 | `(68.5573,6.1145,76.2965,8.3638)` | top / left | no / no |
| ( 0,-8) | 0.833145 | `(12.9716,6.7924,27.2178,20.7000)` | bottom / right | no / yes |
| ( 0, 0) | 0.752775 | `(10.2525,8.9953,89.7452,20.7000)` | bottom / left | no / yes |
| ( 0,+8) | 0.833145 | `(59.0284,6.7924,44.7822,20.7000)` | top / right | no / yes |
| (+8,-8) | 0.829863 | `(20.5573,6.1145,28.2965,8.3638)` | bottom / right | no / no |
| (+8, 0) | 0.677804 | `(23.3086,12.1179,34.5588,9.3826)` | bottom / right | yes / no |
| (+8,+8) | 0.829863 | `(51.4427,6.1145,43.7035,8.3638)` | top / right | no / no |

두 electrode 길이는 독립적이며 실제 최적값에서도 대부분 다르다. Terminal
label을 바꾸면 signed current의 부호만 바뀌고 `|I|`는 같다.

## 7. 0.5 µm와 최종 0.25 µm 비교

| beam (µm) | best 0.5 (nA) | same geometry at 0.25 (nA) | final 0.25 (nA) | mesh change | 0.25 search uplift |
|---:|---:|---:|---:|---:|---:|
| (-8,-8) | 0.721305 | 0.815062 | 0.829863 | +12.998% | +1.816% |
| (-8, 0) | 0.647809 | 0.649224 | 0.677804 | +0.218% | +4.402% |
| (-8,+8) | 0.721305 | 0.815062 | 0.829863 | +12.998% | +1.816% |
| ( 0,-8) | 0.820996 | 0.829908 | 0.833145 | +1.086% | +0.390% |
| ( 0, 0) | 0.744135 | 0.745741 | 0.752775 | +0.216% | +0.943% |
| ( 0,+8) | 0.820996 | 0.829908 | 0.833145 | +1.086% | +0.390% |
| (+8,-8) | 0.721305 | 0.815062 | 0.829863 | +12.998% | +1.816% |
| (+8, 0) | 0.647809 | 0.649224 | 0.677804 | +0.218% | +4.402% |
| (+8,+8) | 0.721305 | 0.815062 | 0.829863 | +12.998% | +1.816% |

`(-8,0)`과 `(+8,0)`은 0.25 µm에서 더 좋은 다른 basin을 찾아 geometry 변화가
크다. Center beam도 geometry가 약 5 µm 이동했다. 따라서 0.25 µm 단계는 단순
재평가가 아니라 실제 local re-optimization과 cross-width basin audit까지 포함한다.

## 8. 현재 말할 수 있는 것과 아직 말할 수 없는 것

말할 수 있는 것:

- signed-current adjoint가 central FD와 일치한다.
- 0.25 µm에서 선택한 Robin relaxation이 hard contact에 충분히 가깝다.
- 각 beam별 electrode는 독립적으로 최적화되며 두 길이가 달라도 된다.
- 0.5 µm search는 48 starts/branch까지 plateau다.
- 0.25 µm best-found 후보는 transition width와 production quadrature-order knob에
  대해 선언한 tolerance를 통과했다.

아직 말할 수 없는 것:

- 수학적으로 증명된 global optimum
- 0.25 µm 결과의 mesh convergence
- contact resistance, finite metal geometry, metal heat sinking, contact doping을
  포함한 실제 fabricated device의 절대 전류 예측

다음 blocker는 adjoint나 SLSQP가 아니라 finer thermal mesh다. 0.125 µm 또는
적절한 adaptive thermal refinement에서 먼저 현재 0.25 µm geometry를 그대로
hard re-evaluate하고, corner current 변화가 허용 오차 안에 들어오는지 확인한 뒤
필요한 beam만 다시 local optimize하는 순서가 맞다.

## 9. 재현 순서

```bash
cd /home/seunghyun/tairte4/pte_electrode_boundary_adjoint

# 0.25 µm fields
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/generate_250nm_fields.py

# g selection + adjoint/FD
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/validate_250nm_relaxation.py

# transfer/local diagnostic and mesh decomposition
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/run_250nm_local_refinement.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/analyze_mesh_change.py

# transition-width audit, common-seed closure, and merge
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/run_transition_robustness_250nm.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/run_transition_cross_seed_250nm.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/run_transition_cross_seed_closure_250nm.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/merge_transition_closure_250nm.py

# quadrature invariance and final comparison
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/validate_boundary_quadrature_order_250nm.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python refinement/summarize_final_250nm.py
```

병렬 closure 실행 시 `--beam-index 0`, `2`, `6`, `8`을 각각 사용한다. 각 실행은
서로 다른 atomic checkpoint를 사용한다.
