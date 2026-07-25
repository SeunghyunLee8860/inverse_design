# volume_current_inverse_design

TaIrTe₄ 박막 광자 **역설계**(inverse design) — FieldRegion 체적전류 adjoint FDTD로
coherent FOM을 최대화하고, **500 nm 최소 feature/gap**을 실제로 강제하는 자립형 번들.

이 폴더 하나로 두 변형을 모두 돌립니다:

| 모드 | 명령 | 의미 |
|---|---|---|
| **연결형** | `./run_inverse_design.sh connected` | air border 없음 (연결된 박막) |
| **고립형** | `./run_inverse_design.sh isolated`  | 각 셀 주위 **500 nm air moat** → 고립 섬 |

---

## 1. 목적함수 / 물리

- **FOM**: `F = Fx + Fy`, `Fx = |∫_flake E_x·conj(E_z) dV|²`, `Fy = |∫_flake E_y·conj(E_z) dV|²`
- **분석 파장 4 µm**, **광대역 source (3–6 µm 펄스)** 로 쏘고 DFT로 4 µm만 추출 (FDTD 표준)
- 6 µm 주기(x/y), 241×241×13 physical 격자, z-불변 압출
- 재료: TaIrTe₄ 이방성 (measured `bundle/perm_data.txt`, 2.7–13.2 µm fit)

## 2. 500 nm를 "실제로" 거는 방법 (옛 방식과의 차이)

옛 코드(단일 conic+tanh)는 500 nm를 **못 걸었습니다**(실측 150 nm). 이 번들은 4축을 바꿨습니다:

1. **240×240 unique periodic latent** + endpoint-free periodic conic filter (off-by-one 제거)
2. **3-field 매핑**(erode/nominal/dilate) + **Zhou 2015 solid/void 길이척도 제약** (`geometric_constraints.py`)
3. **NLopt MMA** 제약 최적화 (Adam → MMA; `run_constrained_inverse_design.py`)
4. **solver-safe affine 층** — rho가 0/1 rail에 닿아도 Jacobian probe 안전 → **beta 데드락 제거**

**성공 판정** = "beta가 N 도달"이 **아니라**:
- 배열이 정확히 `{0,1}`
- **독립 DRC**(`geometry_drc.py`, disk-covering local thickness)로 500 nm solid/void(고립형은 gap도) 통과
- 그 exact 이진 마스크의 **exact FDTD FOM** 계산됨
→ 셋 다 만족해야 `SUCCESS.json` 생성. 미달이면 SUCCESS 없음 + nonzero exit.

## 3. 요구 환경

- **Ansys Lumerical 2026 R1.02 (v261)** + 네트워크 라이선스 + 유휴 GPU (상용 — 번들 불가)
  - lumapi 경로: `VC_LUMERICAL_ROOT=/path/to/opt/lumerical/v261` (기본 탐색 경로도 있음)
- Python 3 + `pip install -r requirements.txt` (numpy, scipy, autograd, **nlopt**)

## 4. 실행

```bash
pip install -r requirements.txt
export VC_LUMERICAL_ROOT=/path/to/opt/lumerical/v261      # lumapi 위치

# 연결형 (GPU 1)
GPU="GPU 1" ./run_inverse_design.sh connected

# 고립형 (GPU 3) — 다른 GPU면 동시에 가능
GPU="GPU 3" ./run_inverse_design.sh isolated
```
산출물: `runs/<mode>_500nm_*/` — `history.jsonl`, `checkpoints/best_feasible.npz`,
`final_design.npz`, `final_projection/{geometry_drc.json, exact_binary_fom.json,
final_mask_*.npz, SUCCESS.json}`.

## 5. Lumerical 없이 지금 바로 검증 (핵심 수치)

```bash
python -m pytest inverse_design/tests/ -q          # 39 passed
```
검증 항목(실측): periodic filter(roll 2e-16, VJP 3e-7), 3-field ordering(erode≤nominal≤dilate),
uniform 0.5→0.5(편향 0), 길이척도 제약 gradient(~1e-7), **DRC 19px→FAIL / 21px→PASS,
대각 수직폭 인식, seam·island-gap**, solver-safe affine(rail probe∈[0,1]).

DRC를 임의 이진 마스크에 직접:
```bash
python inverse_design/geometry_drc.py mask.npz --min-solid-um 0.5 --min-void-um 0.5
```

## 6. 파일 지도

```
run_inverse_design.sh            두 모드 런처 (connected|isolated)
requirements.txt                 numpy scipy autograd nlopt
IMPLEMENTATION_STATUS.md         명세 0~18 섹션별 구현·검증 대응표
eqc_lib.py                       모델 로더 + Lumerical 런타임
volume_current_evaluator.py      forward+adjoint FDTD, FOM/gradient, solver-safe 층
volume_current_{adjoint_core,colored_jacobian,yee_metric}.py, collocated_coherent_fom.py
bundle/
  tairte4_volume_model.py        지오메트리/소스/메시/재료, 매핑 선택
  periodic_filter.py             endpoint-free periodic conic filter
  periodic_constrained_mapping.py 3-field 240-unique 매핑 (+ 고립 옵션)
  symmetric_mapping.py           [LEGACY] 옛 매핑 (재현용)
  perm_data.txt                  TaIrTe4 이방성 유전율 (measured)
  msopt/                         forward simulator + filter/threshold helper
inverse_design/
  run_constrained_inverse_design.py  NLopt MMA 제약 최적화 (메인)
  geometric_constraints.py       미분가능 solid/void 길이척도 제약
  geometry_drc.py                독립 periodic 이진 DRC (500 nm 판정자)
  final_projection.py            exact binary → DRC → exact FDTD 게이트
  evaluate_design.py, mapping_diagnostics.py
  tests/                         비-FDTD 검증 (39개)
tests/test_full_chain_filter_project_adfd.py   Lumerical AD/FD (없으면 skip)
```

## 7. 주의

- 이 번들의 `bundle/perm_data.txt`(측정 데이터)와 `bundle/msopt/`(랩 helper)는 연구 자산입니다.
- 실행 전 **빈 GPU 확인**(`nvidia-smi`), 바쁜 GPU 강행 금지.
- 상세는 `IMPLEMENTATION_STATUS.md` 참고 (무엇이 Lumerical 없이 검증됐고, 무엇이 본 실행에서 검증 필요한지).
