# TaIrTe4 전극 inverse design: Phase 1–2 보고서

## 1. 정확히 어떤 문제를 푸는가

24 x 24 um², 두께 100 nm의 uniform TaIrTe4가 air/TaIrTe4/thermally-grown
SiO2/Si 구조 위에 있다고 둔다. Optical Maxwell solve는 하지 않는다. 각 빔 중심
`(xb,yb)`에 대해 Gaussian 열원으로 기존 explicit 3-D thermal solver를 한 번 풀고,
얻어진 TaIrTe4 두께 평균 `T_b(x,y)`를 전극 최적화 동안 고정한다.

중요한 목적함수 계약은 다음과 같다.

- 빔 위치마다 별도의 전극 최적화를 한다.
- 여러 빔의 mean current를 최대화하지 않는다.
- 전극 0과 1의 위치와 길이는 모두 독립 설계변수다.
- 따라서 같은 top edge에 두 전극이 오는 해도 실제 최대라면 허용하지만, top을
  미리 best라고 고정하지 않는다.
- 최종 판단값은 smooth relaxation 값이 아니라 기존 hard-Dirichlet FEM에서 다시
  계산한 `|Isc|`다.

이번 범위는 첨부 지시대로 Phase 1과 Phase 2다. 아직 “최적 전극을 찾았다”고
보고하는 단계가 아니다.

## 2. Phase 1: 기존 코드를 실제로 따라가 본 결과

### 2.1 전기 FEM

기존 `electrical.py`는 structured rectangle의 각 cell을 두 개로 나눈 P1 triangle
FEM이다. 0.5 um mesh에서는 49 x 49 = 2401 node, 4608 triangle이다. 미지수
`psi`는 node에 있고 triangle 안의 `grad(psi)`는 상수다.

각 element matrix는

```text
Ke = t Ae Be^T sigma Be
```

다. `t=100 nm`가 matrix에 정확히 한 번 들어간다. 실제 sparse matrix와 contact
elimination 후 reduced matrix를 측정했을 때 relative asymmetry는 모두 0이었다.
다만 향후 adjoint에서는 이 사실을 추정으로 사용하지 않고 항상 `K^T`를 푼다.

전극 0 node는 `psi=0`, 전극 1 node는 `psi=1`로 exact elimination한다. 나머지
경계에 별도 항을 넣지 않으므로 natural insulating Neumann 경계다.

### 2.2 PTE current discretization

thermal cell 온도를 TaIrTe4 두께 방향으로 가중 평균한 뒤 네 corner node로
평균한다. 각 triangle에서

```text
grad(T)e = sum_i Ti grad(Ni)
alpha = sigma S
jPTE,e = -alpha grad(T)e
Isc = -t sum_e Ae grad(psi)e^T alpha grad(T)e
```

를 사용한다. 이를 node vector `q`로 모으면 정확히

```text
Isc = q^T psi
```

가 된다. 실제 reconstruction relative error는 `4.17e-16`이었다. 두께만 두 배로
바꾼 독립 계산에서 `psi`는 그대로이고 current와 conductance는 모두 정확히 두
배가 되었다. 즉 thickness 누락이나 이중 계산은 발견되지 않았다.

terminal conductance는 unit terminal bias에 대해 `G=psi^T K psi`다. 전극 0/1을
바꾸면 `psi'=1-psi`, `I'=-I`가 각각 `7.85e-14` max error와 `9.87e-15`
relative error로 성립했다.

### 2.3 기존 center/length가 실제 geometry가 되는 방식

기존 최적화는 먼저 10개 side pair를 열거한다. 각 contact는 side 안에서 nominal
center/length를 만든 뒤, 실제 물리 계산에서는

```text
abs(boundary_node_tangent-center) <= length/2
```

인 node 정수 집합으로 바뀐다. 최소 두 node가 필요하다. 다른 side인 경우 contact
길이는 독립이고 corner clearance 안쪽에 머문다. 같은 side에서는 두 최소 길이와
최소 gap을 먼저 예약하고 남는 길이를 분배해 nominal overlap을 막는다. 마지막에는
node 집합 overlap도 다시 거부한다.

DE cache key는 continuous parameter가 아니라 정확히

```text
(tuple(contact0_node_ids), tuple(contact1_node_ids))
```

다. 그러므로 서로 다른 center/length라도 같은 node에 붙으면 같은 물리 문제다.

### 2.4 non-smoothness를 직접 측정한 결과

center와 length를 각각 0.025 um 간격으로 321점 sweep했다.

- center 321점 -> contact node set 33개
- length 321점 -> contact node set 9개
- `h=0.001, 0.005, 0.01, 0.05 um` 중앙차분 -> 같은 plateau 안이라 gradient 0
- 더 큰 `h` -> node가 한꺼번에 추가/삭제되며 jump/h 값만 생성

따라서 기존 hard-node objective는 nominal 변수에 대해 piecewise constant다.
기존 문제에 그대로 adjoint를 붙이면 거의 모든 곳에서 정확히 0이 나와 endpoint를
움직일 수 없다. 이 때문에 optimization 동안만 differentiable boundary
representation이 필요하다.

## 3. 논문 및 supplement와의 대조

로컬 Advanced Functional Materials 논문과 Supplement S5의 Table S2, Eq. S1–S7,
Blevins thesis를 대조했다.

일치하는 부분은 다음과 같다.

- `jloc=-sigma S grad(T)`와 `I=integral jloc.grad(psi)`의 부호/수축
- contact 0/1 Dirichlet, 나머지 sample boundary zero flux
- 최종 volumetric integration을 2-D thin sheet에서 `t*dA`로 환산한 것
- `x=b,y=a,z=c` 순서로 permutation하면 kappa, sigma, Seebeck 값이 Table S2와
  정확히 일치하는 것
- Gaussian lateral factor `exp(-2r²/w0²)`와
  `G_TaIrTe4-thermal-SiO2=7.37e6 W/m²/K`

하지만 “논문과 완전히 동일”이라고 말할 수 없는 부분도 확인했다.

- Supplement S7은 `laplacian(psi)=0`로 인쇄돼 있지만 코드는 anisotropic
  `div(sigma grad(psi))=0`을 푼다. 코드는 anisotropic continuity에 맞는 generalized
  reciprocal problem이지만, printed S7과 literal하게 동일하지는 않다.
- 논문은 Beer–Lambert z profile을 쓰지만 현재 optical-free 설정은 uniform z heat를
  쓴다.
- 8.5 um waist와 1 uW 전체 heat conversion은 material constant가 아니라 현 config다.
- 현재 air는 explicit layer로 풀고 그 먼 top boundary에 `h=10 W/m²/K`를 둔다.
  Table S2의 TaIrTe4-air interface `G=1 W/m²/K`를 별도 resistance로 넣은 것은 아니다.
- SiO2-Si `G=1.10e9 W/m²/K`는 현재 model의 추가 가정이며 Table S2 값은 아니다.

이번 작업에서는 thermal physics를 바꾸지 않는다. 위 차이는 숨기지 않고 provenance로
고정하며, 전극 방법 비교에서는 old/new 모두 같은 `T_b`를 사용한다.

## 4. Phase 2: 새 full-perimeter 설계변수

사각형 둘레 전체를 반시계 방향 periodic coordinate `s in [0,P)`, `P=96 um`로 둔다.

```text
s=0          bottom-left
0 -> 24      bottom: left -> right
24 -> 48     right: bottom -> top
48 -> 72     top: right -> left
72 -> 96     left: top -> bottom
```

설계변수는 미터 단위의

```text
p = (c0,L0,c1,L1)
```

다. 각 contact는 하나의 contiguous periodic arc이므로 corner를 지나도 segment가
쪼개지지 않는다. 이 parameterization은 side-pair enumeration을 없애며 두 길이를
같게 강제하지 않는다.

기존 0.5 um config와 공정 범위를 먼저 동일하게 비교하려면 `Lmin=1 um`,
`gap=0.5 um`, `Lmax=20.7 um`가 자연스럽다. 20.7 um는 기존 한 side의
`0.9*(24-2*0.5)`다. 다만 corner crossing 허용과 old corner clearance는 동시에
유지할 수 없다. 이것은 config에서 명시적으로 선택해야 하며 자동으로 물리를
바꾸지 않는다.

## 5. smooth compact contact와 forward equation

plain logistic은 perimeter 어디에서도 정확히 0이 아니다. 그래서 `g -> infinity`면
결국 둘레 전체가 contact가 되는 잘못된 hard limit를 갖는다. 이를 피하기 위해 arc
밖에서 정확히 0이고 endpoint 안쪽 transition에서만 변하는 compact C2 quintic
mask `mk(s;ck,Lk)`를 제안했다. 정의와 `dm/dc`, `dm/dL`의 닫힌식은
`derivation/PHASE2_FORMULATION.md`에 적었다.

최적화 중 경계식은

```text
n.sigma.grad(psi) = g m0(V0-psi) + g m1(V1-psi)
V0=0, V1=1
```

이다. 같은 P1 boundary-edge quadrature로 조립하면

```text
Bk = t g integral_boundary mk N^T N ds
bk = t g integral_boundary mk N ds
K(p) = Kbulk+B0+B1
f(p) = V0 b0+V1 b1 = b1
K(p) psi = f(p)
```

가 된다. `g`는 전기 contact conductance 단위 `S/m²`이며 thermal interface G와
무관하다. 측정 contact resistivity가 없는 현재에는 physical fit parameter가 아니라
hard contact로 가기 위한 continuation parameter다. Phase 3에서 `g` sweep과 matrix
conditioning을 보고 선택해야 한다.

## 6. production objective: `I²`가 아니라 두 signed branch

고정된 beam 온도는 고정 `q`를 만들고

```text
I = q^T psi
```

다. Production optimization에서는 `I²`를 사용하지 않는다. `I²`는 `I=0`에서
gradient가 사라지고 단위가 A²인 매우 작은 수가 되기 때문이다. 각 beam과 각
initialization에 대해 다음 두 문제를 독립적으로 푼다.

```text
b=+1: maximize  +I/Iref
b=-1: maximize  -I/Iref
```

`+` branch는 가장 큰 양의 current, `-` branch는 가장 작은 음의 current를 찾는다.
두 결과를 hard-Dirichlet로 다시 푼 뒤 `abs(Ihard)`가 큰 geometry를 선택한다.
`I²`와 그 gradient는 diagnostic으로만 남긴다.

`Iref`는 한 beam에서 design과 무관하게 고정해야 한다. 기본값은 `norm(q,1)`이며
필요하면 config에서 명시적으로 override한다. optimizer에는 dimensionless response만
보내지만 결과에는 raw ampere 값을 항상 저장한다.

## 7. 정확한 signed-current discrete adjoint

상태식을 미분하면

```text
K dpsi/dpi = dfi/dpi - dKi/dpi psi
```

이고 current adjoint를

```text
K^T lambdaI = q
```

로 정의하면 각 설계변수의 exact discrete current gradient는

```text
dI/dpi = lambdaI^T (df/dpi - dK/dpi psi)
```

다. contact `k` 변수에 대해서는

```text
dK/dpi = t g integral (dmk/dpi) N^T N ds
df/dpi = t g Vk integral (dmk/dpi) N ds
```

이며 forward와 derivative에 반드시 같은 quadrature를 쓴다. 이 유도는 sigma가
anisotropic이어도 변하지 않는다. 현재 K가 실제 symmetric인 것은 확인했지만,
adjoint 구현은 일반성을 위해 `K^T`를 푼다. SciPy가 최소화하는 branch objective는

```text
phib = -b I/Iref
dphib/dpi = -b/Iref * dI/dpi
```

다. 한 번 계산한 `dI/dp`로 양쪽 branch를 만들 수 있다.

전극을 교환하면 K는 그대로이고 `psi'=1-psi`, `I'=-I`여야 한다. 따라서 terminal
swap은 `+` branch와 `-` branch를 서로 바꾼다. Phase 3에서 gradient permutation까지
포함해 시험한다.

## 8. nondimensionalization과 0/P seam 제거

SLSQP에 미터 단위 `(c0,L0,c1,L1)`나 A² objective를 그대로 주지 않는다. 변수는

```text
x=(u0,l0,u1,l1)=(c0/P,L0/P,c1/P,L1/P)
```

로 모두 perimeter `P`로 나눈다. 따라서 chain rule은

```text
dI/dxi = P dI/dpi
dphib/dxi = -b P/Iref * dI/dpi
```

다. 변수, objective, constraint Jacobian이 모두 order-one dimensionless scale이 된다.

중심 `u0,u1`에는 `[0,1]` box bound를 두지 않는다. 대신 실수 전체에 사는 lifted
periodic variable로 두고 mask와 constraint가 `sin(2pi u)`, `cos(2pi u)`를 통해
정확히 주기적이게 한다. 따라서 SLSQP line search는 `0/1` seam을 자유롭게 통과하며
artificial barrier가 없다. `u mod 1` wrapping은 결과 출력, clustering, hard-contact
변환 시에만 수행한다. 길이 `l0,l1`에만 physical min/max box bound를 둔다.

## 9. geometry constraint와 optimizer 선택 원칙

두 periodic arc의 최소 gap을 smooth하게 강제하려면

```text
r = pi*(l0+l1+2gap/P)
Delta = 2pi*(u1-u0)
hsep = cos(r)-cos(Delta) >= 0
hpack = 1-l0-l1-2gap/P >= 0
```

를 사용한다. 두 식 모두 analytic Jacobian이 있다. 길이 box bound와 이 두 smooth
constraint만 남으므로, gradient가 검증된 후 첫 후보는 SLSQP다. penalty weight가
필요 없고 네 변수 문제에 충분히 가볍다. constraint residual이나 line search가
불안하면 trust-constr를 독립 확인용으로 쓴다.

SLSQP 한 번으로 global optimum을 주장하지 않는다. perimeter에 분산한 deterministic
starts, old-DE 해를 변환한 starts, electrode-swap starts를 모두 사용하고, periodic 및
swap-equivalent 해를 canonicalize한 뒤 objective cluster와 분포를 보고한다.

## 10. 다음 Phase 3에서 통과해야 하는 gate

다음 항목을 통과하기 전에는 production optimizer를 실행하지 않는다.

1. 두 signed branch와 네 scaled 변수 모두에서 여러 `h`의 central FD 대 adjoint
   gradient error curve
2. 1.0/0.5/0.25 um bulk mesh convergence
3. boundary Gaussian quadrature order convergence. 우선 `3,5,7,9`를 비교하고
   current와 네 gradient가 tolerance 안에서 멈출 때까지 늘림
4. smoothing transition width convergence. 우선 `epsilon/h=2,1,0.5`를 비교하고,
   각 epsilon에서 quadrature가 transition을 충분히 resolve하는지 함께 확인
5. `g` 증가에 따른 hard-Dirichlet `psi`와 `Isc` 수렴 및 conditioning
6. electrode swap, x/y reflection, zero-`grad(T)` tests
7. old DE geometry의 hard node set과 `Isc` exact equivalence
8. smooth optimum을 hard node로 projection한 뒤 성능 유지 여부

quadrature order와 transition width는 따로 통과 판정을 내리지 않는다. 너무 좁은
transition을 적은 quadrature point로 적분하면 매끄럽지만 틀린 gradient가 나올 수
있다. 따라서 `(mesh,quadrature,epsilon,hFD)` 조합별로 raw `I`, 네 adjoint 값, 네 FD
값, relative error를 같은 표에 기록한다.

그 다음에만 beam마다 multi-start optimization을 하고, old DE best와 위치/길이,
realized contact, current, weighting solve 수, runtime을 비교한다.
