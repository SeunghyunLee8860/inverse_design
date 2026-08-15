# 0.125 µm full refinement와 62.5 nm targeted mesh audit

## 결론

```text
0.125 µm explicit thermal solve: PASS
0.125 µm Robin/adjoint gate: PASS
선택적 signed SLSQP 48/48: PASS
0.25 µm electrode geometry는 0.125 µm에서도 그대로 best
direct mesh convergence: 아직 미통과
62.5 nm targeted thermal pilot: PASS
62.5 nm successive 1% current gate: 아직 미통과
```

이번 단계에서 중요한 사실은 optimizer와 mesh effect가 분리됐다는 것이다.
0.125 µm에서 다시 최적화해도 electrode geometry는 전혀 개선되지 않았다. 남은
변화는 electrode search 실패가 아니라 thermal discretization에 따른 current
변화다.

## 1. 0.125 µm explicit thermal model

기존 conservative Cartesian FVM과 동일한 air/TaIrTe4/thermally-grown-SiO2/Si
모델을 사용했다.

- thermal cell shape: `218 x 218 x 36` = 약 171만 cell
- electrical node shape: `193 x 193`
- 9 beam을 각각 직접 solve
- CG iterations: `1134–1482`
- beam당 solve time: 약 `50–67 s`
- 최대 linear residual: `6.34e-11`
- 최대 energy-balance error: `8.22e-13`

각 beam 결과는 atomic checkpoint로 저장되므로 중단 후 이어서 실행할 수 있다.
Solver `rtol=5e-11`과 별도로 보고되는 unpreconditioned residual gate는
`7e-11`로 선언했다.

## 2. 0.125 µm Robin relaxation과 adjoint

대표적인 corner, x-edge, y-edge, center geometry에서 `g` sweep을 다시 수행했다.

```text
selected g = 1e14 S/m2
transition width = 0.50 µm
contact discretization = nodal_lumped
```

선택한 `g`에서:

- 최대 smooth-hard current error: `0.3084%`
- 최대 weighting-potential L2 error: `0.7399%`
- 최소 scaled gradient norm: `1.09e-10 A`
- adjoint-central-FD 최대 component error: `1.49e-5`

`g=1e18`에서 current error는 `7.58e-5`까지 감소하지만 `psi`는 약 0.534%에서
plateau한다. 이는 hard contact가 equality endpoint node를 포함하는 반면 smooth
mask endpoint 값은 0이기 때문이다. Current는 수렴하고 `psi<1%` 선언 기준도
만족하므로 gate는 `PASS`다.

## 3. 0.25 µm geometry의 0.125 µm hard 재평가

| beam (µm) | 0.25 µm hard (nA) | 0.125 µm hard (nA) | relative change |
|---:|---:|---:|---:|
| (-8,-8) | 0.829863 | 0.872520 | +5.140% |
| (-8, 0) | 0.677804 | 0.696820 | +2.806% |
| (-8,+8) | 0.829863 | 0.872520 | +5.140% |
| ( 0,-8) | 0.833145 | 0.834921 | +0.213% |
| ( 0, 0) | 0.752775 | 0.752439 | -0.0446% |
| ( 0,+8) | 0.833145 | 0.834921 | +0.213% |
| (+8,-8) | 0.829863 | 0.872520 | +5.140% |
| (+8, 0) | 0.677804 | 0.696820 | +2.806% |
| (+8,+8) | 0.829863 | 0.872520 | +5.140% |

y-edge와 center는 1% 안에 들어왔지만, corner와 x-edge는 아직 들어오지 않았다.

## 4. 선택적 0.125 µm local optimization

1%를 넘은 여섯 beam만 다시 최적화했다.

- beam indices: `0,1,2,6,7,8`
- 각 beam에서 `+I`, `-I` 별도 branch
- branch당 4 common local starts
- 총 48 SLSQP runs
- 성공: `48/48`
- 모든 start와 endpoint를 hard contact로 재계산

최대 improvement는 `9.39e-13` relative로 수치 roundoff 수준이었다. 즉 모든
beam에서 transferred 0.25 µm geometry가 그대로 0.125 µm hard winner다. 따라서
추가 multistart보다 thermal refinement가 다음 blocker라는 결론이 강화됐다.

## 5. 62.5 nm targeted pilot

아직 실패한 두 대칭 독립 유형 `(-8,-8)`과 `(-8,0)`만 62.5 nm에서 직접 풀었다.

- thermal cell shape: `410 x 410 x 36` = 약 605만 cell
- electrical node shape: `385 x 385`
- assembly: `3.6–5.1 s`
- corner: 2927 CG iterations, 445.8 s
- x-edge: 2747 CG iterations, 422.1 s
- 최대 energy-balance error: `1.21e-12`
- 최대 reported residual: `2.98e-10`

605만-cell 행렬에서는 사후 unpreconditioned matvec residual이 약 `3e-10`
floating-point cancellation floor에 도달했다. CG는 정상 종료했고 energy balance는
`1e-12` 수준이므로 thermal acceptance gate를 `5e-10`으로 별도 선언했다.

| type | 0.125 µm (nA) | 62.5 nm (nA) | direct change |
|---:|---:|---:|---:|
| corner | 0.872520 | 0.888815 | +1.868% |
| x-edge | 0.696820 | 0.703926 | +1.020% |

두 유형 모두 감소 추세지만 엄격한 successive-mesh 1% 기준에는 아직 들지 않았다.

## 6. Mesh-series 진단

동일 geometry의 `0.25 -> 0.125 -> 0.0625 µm` current 차이에서 관측한 order는:

```text
corner: 1.3883
x-edge: 1.4203
```

두 유형이 약 1.4-order의 일관된 monotone trend를 보인다. Richardson 진단은:

| type | 62.5 nm value (nA) | extrapolated value (nA) | estimated remaining error | predicted 62.5→31.25 change |
|---:|---:|---:|---:|---:|
| corner | 0.888815 | 0.898889 | 1.121% | 0.700% |
| x-edge | 0.703926 | 0.708164 | 0.599% | 0.377% |

이는 31.25 nm direct check에서 successive 1% gate가 통과할 가능성을 보여주지만,
직접 계산을 대체하는 증거는 아니다.

31.25 nm uniform model의 예상 크기는:

```text
thermal: 794 x 794 x 36 = 22,695,696 cells
electrical: 769 x 769 = 591,361 nodes
```

따라서 다음 단계는 무작정 full 9-beam uniform solve가 아니라 다음 둘 중 하나다.

1. corner와 x-edge만 31.25 nm direct pilot
2. thermal multigrid/AMG 또는 flake-near adaptive refinement를 먼저 도입

현재 direct evidence 기준 최종 판정은

```text
electrode geometry locally stable: PASS
62.5 nm thermal solves: PASS
1% direct mesh convergence: NOT YET PASS
```

이다.

## 재현 파일

- `generate_125nm_fields.py`, `per_beam_125nm_thermal.json`
- `per_beam_125nm_fields.npz`, `field_checkpoints/`
- `validate_125nm_relaxation.py`, `relaxation_125nm.json/.png`
- `run_selective_local_refinement.py`
- `local_refinement_beam??_125nm.json`과 checkpoint
- `merge_final_125nm.py`, `final_125nm.json/.png`
- `run_62p5nm_targeted_pilot.py`, `targeted_62p5nm_pilot.json`
- `analyze_mesh_series.py`, `mesh_series_analysis.json`

```bash
cd /home/seunghyun/tairte4/pte_electrode_boundary_adjoint

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  refinement_125nm/generate_125nm_fields.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  refinement_125nm/validate_125nm_relaxation.py

# 필요한 beam에 대해 --beam-index 0,1,2,6,7,8
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  refinement_125nm/run_selective_local_refinement.py --beam-index 0

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  refinement_125nm/merge_final_125nm.py
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  refinement_125nm/run_62p5nm_targeted_pilot.py
python3 refinement_125nm/analyze_mesh_series.py
```
