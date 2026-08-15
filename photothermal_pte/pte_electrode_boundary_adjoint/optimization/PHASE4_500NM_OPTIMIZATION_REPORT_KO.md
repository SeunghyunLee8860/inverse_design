# Phase 4 — 0.5 µm beam별 signed SLSQP optimization

## 결론

```text
COMPLETED
9 beam x 2 signed branches x 12 starts = 216 SLSQP runs
216/216 SLSQP success
각 smooth 종료점을 hard electrode로 재계산한 뒤 기존 DE와 비교
새 SLSQP-derived hard 후보 승리: 4 beams
기존 DE hard 후보 승리: 5 beams
```

이 계산은 beam 평균 목적함수가 아니다. Gaussian 중심 9개마다 저장된 서로
다른 temperature field를 고정하고, 각각 완전히 독립적인 electrode optimization을
수행했다. 두 electrode의 길이도 독립 변수이므로 서로 같을 필요가 없다.

## Production 전에 통과시킨 두 gate

1. 실제 center-beam temperature에서 네 변수의 raw-current adjoint gradient가
   central finite difference와 일치했다. 최악의 component 상대오차는
   `4.58e-6`이고 central-FD 수렴차수는 `1.99992`, `2.00005`였다.
2. `nodal_lumped` Robin contact가 같은 0.5 µm mesh의 hard node contact로
   수렴했다. `g=1e18 S/m2`에서 current와 `psi` 상대 L2 오차는 각각
   `1.03e-5`, `2.10e-6`이다.

처음 사용한 consistent-edge Robin은 hard node model과 같은 fixed-mesh limit을
갖지 않았다. 이 경우 `g=1e18`에서도 current 오차가 `17.42%`였기 때문에
optimizer를 돌리지 않고 contact assembly를 nodal mass lumping으로 수정했다.
수정 후 adjoint-FD 검증도 처음부터 다시 통과시켰다.

## 실제 최적화 문제

TaIrTe4 외부 둘레 좌표를 `s`라 하고 총 둘레를 `P=96 um`라 했다. 최적화
변수는 다음처럼 무차원화했다.

```text
x = (u0, ell0, u1, ell1)
  = (c0/P, L0/P, c1/P, L1/P)
```

- `u0,u1`: box bound가 없는 lifted periodic center. 따라서 `0/P` seam은
  SLSQP 경계가 아니다.
- `ell0,ell1`: 각각 `1.0/96 <= ell <= 20.7/96`.
- nominal hard contact 사이 최소 perimeter gap: `0.5 um`.
- smooth contact: nodal-lumped Robin, `g=1e12 S/m2`, transition `0.75 um`.

각 beam에서 다음 두 minimization을 별도로 수행했다.

```text
branch b=+1: minimize -I/I_ref
branch b=-1: minimize +I/I_ref
```

즉 `I^2` 하나를 optimize하지 않았다. `I_ref=||q||_1`으로 objective와
gradient의 크기를 무차원화했고, geometry도 `P`로 scale했다.

한 objective evaluation의 순서는

```text
x -> smooth contact mask -> weighting potential psi
  -> I=q^T psi -> transpose adjoint -> dI/dx
```

이다. SLSQP가 같은 `x`에서 objective와 gradient를 연속 요청할 때 PDE를
두 번 풀지 않도록 evaluation cache도 넣었다.

## Smooth와 hard를 분리한 선택 규칙

SLSQP는 differentiable Robin current만 본다. 그러나 최종 ranking은 다음과 같다.

1. 각 signed branch를 12개 deterministic/asymmetric start에서 실행한다.
2. 종료한 24개 geometry를 전부 hard Dirichlet node contact로 다시 푼다.
3. feasible 후보 중 `abs(I_hard)`가 가장 큰 SLSQP-derived 후보를 고른다.
4. 그 값을 기존 DE geometry의 재계산된 `abs(I_hard)`와 비교한다.
5. 둘 중 큰 것을 해당 beam의 최종 0.5 µm 후보로 삼는다.

이 절차가 실제로 필요했다. Center beam에서는 smooth-derived 최선의 hard
current가 DE보다 `0.056%` 작았고, `(0,+-8) um` beam에서는 오히려 `8.10%`
컸다. Smooth objective의 순위와 hard staircase의 순위가 항상 같지 않다.

## 9-beam 결과

전류 단위는 nA이다. Ratio는
`best SLSQP-derived hard / legacy-DE hard`이다.

| beam center (um) | SLSQP→hard | legacy DE hard | ratio | 최종 선택 |
|---:|---:|---:|---:|:---|
| (-8,-8) | 0.712168 | 0.721305 | 0.987333 | legacy DE |
| (-8, 0) | 0.647809 | 0.645241 | 1.003979 | SLSQP→hard |
| (-8,+8) | 0.712168 | 0.721305 | 0.987333 | legacy DE |
| ( 0,-8) | 0.820996 | 0.759506 | 1.080959 | SLSQP→hard |
| ( 0, 0) | 0.743720 | 0.744135 | 0.999442 | legacy DE |
| ( 0,+8) | 0.820996 | 0.759506 | 1.080959 | SLSQP→hard |
| (+8,-8) | 0.712168 | 0.721305 | 0.987333 | legacy DE |
| (+8, 0) | 0.647809 | 0.645241 | 1.003979 | SLSQP→hard |
| (+8,+8) | 0.712168 | 0.721305 | 0.987333 | legacy DE |

반사 대칭인 beam끼리 결과가 수치적으로 일치한다. 이는 별도 symmetry
constraint를 걸어서 만든 결과가 아니라 각각 독립 실행한 결과다.

## 새 방법이 이긴 네 beam의 electrode geometry

둘레 좌표 convention은 다음과 같다.

```text
s=0: bottom-left corner
bottom: 0 -> 24 um, left to right
right: 24 -> 48 um, bottom to top
top: 48 -> 72 um, right to left
left: 72 -> 96 um, top to bottom
```

`(c0,L0,c1,L1)`은 um 단위이다. Electrode 0은 weighting potential 0,
electrode 1은 weighting potential 1인 terminal labeling이다. Label을 바꾸면
current sign만 바뀌며 `abs(I)`는 같다.

| beam center (um) | `(c0,L0,c1,L1)` (um) | signed `I_hard` (A) |
|---:|:---|---:|
| (-8,0) | `(4.1952,10.6534,20.3719,20.7000)` | `+6.47809e-10` |
| (+8,0) | `(19.8047,10.6532,3.6281,20.7000)` | `+6.47809e-10` |
| (0,-8) | `(92.6481,20.7000,11.0923,7.1885)` | `-8.20996e-10` |
| (0,+8) | `(75.3519,20.7000,60.9076,7.1886)` | `-8.20996e-10` |

일부 contact는 corner를 지나 두 외부 edge에 걸치는 L-shaped perimeter
footprint이다. 이것은 TaIrTe4 top surface 위에 electrode를 놓았다는 뜻이
아니다. 모든 contact는 flake의 외부 둘레에 있고, arc midpoint가 어느 side에
있는지와 footprint가 corner를 넘는지는 별개다. 기존 DE는 한 electrode를
한 side 안에 제한했기 때문에, 특히 `(0,+-8) um`의 약 8.1% 향상은 넓어진
perimeter design space에서 나온 결과다.

## 기존 DE가 최종 승자인 다섯 beam

Corner beam 네 개의 최종 geometry는 반사 관계이며, 예를 들어 `(-8,-8)`은

```text
electrode 0: left,  center=-7.5629 um, length=7.5062 um
electrode 1: bottom, center=-8.5641 um, length=5.1695 um
```

이다. Center beam `(0,0)`의 최종 geometry는

```text
electrode 0: right,  center=-0.7379 um, length=20.6645 um
electrode 1: bottom, center=+1.5899 um, length=8.5404 um
```

이고 `abs(I_hard)=7.441347e-10 A`이다. SLSQP-derived hard 후보와 차이는
0.056%뿐이지만, declared ranking rule에 따라 더 큰 기존 DE 값을 선택한다.

## Numerical audit

- SLSQP success: `216/216`
- 모든 후보 중 최소 separation constraint: `-6.70e-13`
  (허용 tolerance `1e-8`; roundoff 수준)
- 최대 smooth state relative residual: `4.92e-14`
- 최대 adjoint relative residual: `1.07e-13`
- 기존 DE current의 새 hard solver 재현 상대오차: 최대 `4.30e-16`

마지막 항목은 새 hard re-evaluation과 기존 DE 비교가 서로 다른 current
정의 때문에 생긴 차이가 아님을 확인한다.

## 파일과 재현 방법

```bash
cd /home/seunghyun/tairte4/pte_electrode_boundary_adjoint

# blocker gates
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python validation/phase3_gradient_check.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python validation/phase3_robin_hard_convergence.py

# center-beam pilot
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  optimization/run_center_beam_slsqp_multistart.py

# all nine beams
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  optimization/run_all_beams_slsqp_multistart.py
```

주요 결과 파일은 다음과 같다.

- `center_beam_slsqp_multistart.json/.png`
- `all_beams_slsqp_multistart.json/.png`
- 모든 run의 start, iteration history, smooth/hard current, geometry,
  constraint, residual, solver status가 JSON에 저장된다.

## 다음 단계

현재 결과는 0.5 µm fixed-mesh production run이다. 다음 robustness 단계는
0.25 µm에서 temperature/electrical field를 다시 구성하고, boundary
quadrature order 및 transition width convergence를 함께 확인하는 것이다.
특히 hard optimum은 node staircase에 민감하므로 최종 fabrication dimension을
확정하기 전에 0.25 µm hard re-evaluation과 작은 geometry perturbation 검사가
필요하다.
