# 0.5 µm systematic multistart plateau audit

## 판정

```text
PLATEAU_PASS
start budget per signed branch: 12 -> 24 -> 48
total signed SLSQP runs: 864
SLSQP success: 864/864
24 -> 48에서 9개 beam 모두 hard |I| 개선량: 0
미리 정한 plateau tolerance: 0.1%
96 starts는 실행할 근거가 없어 중단
```

이 audit의 목적은 0.5 µm 결과 차이가 mesh 때문인지, 단순히 SLSQP가 local
optimum에 갇힌 것인지 분리하는 것이었다. 각 Gaussian beam은 독립 문제이며,
각 문제에서 `+I`와 `-I`를 별도 branch로 풀었다. `I^2`를 production objective로
사용하지 않았다.

## Nested start 구성

각 budget은 앞 budget의 정확한 prefix이다. 첫 seed에는 다음을 반드시 포함했다.

1. 기존 DE best geometry
2. 기존 DE geometry의 terminal-swapped geometry
3. 앞선 12-start SLSQP campaign의 hard incumbent
4. 그 incumbent의 terminal-swapped geometry
5. 나머지는 seed `20260815`의 deterministic scrambled Sobol feasible design

따라서 `best(12)`, `best(24)`, `best(48)`은 서로 다른 random run의 비교가
아니라 동일한 nested search sequence의 prefix 비교다. 각 start의 초기 hard
geometry도 후보에 포함했고, 모든 SLSQP endpoint도 hard contact로 변환해
재계산했다. 최종 순위는 오직 hard-electrode `abs(I)`로 정했다.

## Plateau 결과

단위는 nA이다.

| beam center (µm) | best(12) | best(24) | best(48) | 24→48 gain |
|---:|---:|---:|---:|---:|
| (-8,-8) | 0.721305329 | 0.721305329 | 0.721305329 | 0 |
| (-8, 0) | 0.647809155 | 0.647809155 | 0.647809155 | 0 |
| (-8,+8) | 0.721305329 | 0.721305329 | 0.721305329 | 0 |
| ( 0,-8) | 0.820995549 | 0.820995549 | 0.820995549 | 0 |
| ( 0, 0) | 0.744134701 | 0.744134701 | 0.744134701 | 0 |
| ( 0,+8) | 0.820995549 | 0.820995549 | 0.820995549 | 0 |
| (+8,-8) | 0.721305329 | 0.721305329 | 0.721305329 | 0 |
| (+8, 0) | 0.647809155 | 0.647809155 | 0.647809155 | 0 |
| (+8,+8) | 0.721305329 | 0.721305329 | 0.721305329 | 0 |

`12 -> 24`와 `24 -> 48`에서 모든 beam의 best hard current가 그대로였다.
미리 선언한 stopping rule은 마지막 budget 증가에서 최대 개선량이 0.1% 이하인
것이므로 48 starts/branch에서 plateau로 판정했다. 96은 결과를 보고 임의로
생략한 것이 아니라 이 stopping rule에 따라 실행하지 않았다.

## 의미와 한계

이 결과는 0.5 µm design space에서 사용한 seed pool과 SLSQP가 더 이상 알려진
해를 개선하지 못했다는 강한 search audit이다. 수학적인 global-optimum 증명은
아니다. 특히 이후 0.25 µm thermal field에서 전류와 일부 optimum basin이
변했으므로, 0.5 µm plateau를 mesh-convergence 증거로 해석하면 안 된다.

재현 파일:

- `run_500nm_search_plateau.py`
- `search_plateau_checkpoint.json`
- `search_plateau_results.json`
- `search_plateau.png`

실행:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  optimization/run_500nm_search_plateau.py --max-starts-per-branch 48
```
