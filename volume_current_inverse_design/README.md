# volume_current_inverse_design

TaIrTe₄ 박막 광자 **역설계**(inverse design) — FieldRegion 체적전류 adjoint FDTD로
coherent FOM을 최대화하고 **500 nm 최소 feature/gap**을 실제로 강제하는 자립형 번들.
한 폴더로 두 변형을 모두 돌립니다:

| 모드 | 명령 | 의미 |
|---|---|---|
| **연결형** | `./run_inverse_design.sh connected` | air border 없음 (연결된 박막) |
| **고립형** | `./run_inverse_design.sh isolated`  | 각 셀 주위 **500 nm air moat** → 고립 섬 |

> 2026-07-25 코드리뷰 반영: DRC/제약/런처/finalizer/provenance 수정. 자세한 대응은
> `IMPLEMENTATION_STATUS.md`의 "코드리뷰(2026-07-25) 대응" 참고.

---

## 1. 목적함수 / 물리
- **FOM**: `F = Fx + Fy`, `Fx = |∫_flake E_x·conj(E_z) dV|²`, `Fy = |∫_flake E_y·conj(E_z) dV|²`
- **분석 파장 4 µm**, **광대역 source (3–6 µm 펄스)** 로 쏘고 DFT로 4 µm만 추출 (FDTD 표준)
- 6 µm 주기(x/y), physical 241×241×13, **unique latent 240×240**, 25 nm in-plane, z-불변 압출
- 재료: TaIrTe₄ 이방성 (measured `bundle/perm_data.txt`, 2.7–13.2 µm fit)

## 2. 500 nm를 "실제로" 거는 방법
옛 코드(단일 conic+tanh)는 500 nm를 못 걸었습니다(실측 150 nm). 4축을 바꿨습니다:
1. **240×240 unique periodic latent** + endpoint-free periodic conic filter
2. **3-field 매핑**(erode/nominal/dilate) + **Zhou 2015 solid/void 길이척도 제약**
   (전역 mean이 아니라 **power-mean**으로 집계 → 국소 위반 희석 안 됨)
3. **NLopt MMA** 제약 최적화
4. **solver-safe affine 층** — rho가 0/1 rail에 닿아도 안전 → beta 데드락 제거

스테이지 스케줄(2026-08-07 재작업): `beta[:maxeval[:min_evals]]` 스테이지별 예산
(기본 `2:40:12,...` — **β=2에서 topology를 충분히 탐색**), β<8 은 제약 없는 **warm-up**
(순수 FOM 상승), 사다리 후 gray>5%면 β-doubling **binarization polish** 자동 추가.
노브: `VC_CONSTRAINT_START_BETA`, `VC_GRAY_TOL`, `VC_BETA_CAP`, `VC_POLISH_MAXEVAL`.

**최종 판정자 = 독립 DRC** (`geometry_drc.py`): **opposing-boundary(local linewidth)** 측정 —
사각형 corner를 linewidth로 오판하지 않고(큰 island는 PASS), 얇은 bar/finger/ring/대각/seam은
정확히 잡음. all-solid/all-void 같은 trivial phase는 FAIL.

**성공 판정** = "beta N 도달"이 **아니라**: 배열이 정확히 `{0,1}` + DRC 통과(500 nm solid/void,
고립형은 gap도) + exact 이진 마스크의 **exact FDTD FOM 계산** → 셋 다 만족해야 `SUCCESS.json`.
DRC 실패/코드해시 불일치/미-feasible → SUCCESS 없음 + nonzero exit (실패를 성공으로 오인 불가).

## 3. 요구 환경
- **Ansys Lumerical 2026 R1.02 (v261)** + 라이선스 + 유휴 GPU (상용 — 번들 불가;
  `VC_LUMERICAL_ROOT=/path/to/opt/lumerical/v261`)
- Python 3 + `pip install -r requirements.txt` (numpy, scipy, autograd, **nlopt**)

## 4. 실행
```bash
pip install -r requirements.txt
export VC_LUMERICAL_ROOT=/path/to/opt/lumerical/v261
GPU="GPU 1" ./run_inverse_design.sh connected      # 연결형
GPU="GPU 3" ./run_inverse_design.sh isolated       # 고립형 (다른 GPU면 동시)
```
산출물: `runs/<mode>_500nm/` — `history.jsonl`, `attempts/attempt_XXXX/{contract.json,stop.json}`,
`checkpoints/best_feasible.npz`, `final_design.npz`,
`final_projection/{geometry_drc.json, exact_binary_fom.json, final_mask_*.npz, SUCCESS.json}`.

## 5. Lumerical 없이 지금 검증
```bash
python -m pytest inverse_design/tests/ -q          # 46 passed
python inverse_design/geometry_drc.py mask.npz --min-solid-um 0.5 --min-void-um 0.5
```
검증 항목(실측): periodic filter(roll 2e-16, VJP 3e-7), 3-field ordering, uniform 0.5→0.5,
길이척도 제약 gradient(~1e-7) + **worst-case 비희석/얇은-설계 판별**, **DRC adversarial fixtures**
(trivial FAIL, 2µm 사각형 PASS, 얇은 bar/finger/대각/seam/island-gap FAIL, 19px→FAIL/21px→PASS),
solver-safe affine(rail probe∈[0,1]), resume hash provenance.

full-chain AD/FD(`tests/test_full_chain_filter_project_adfd.py`)는 safe-vs-safe / exact-vs-exact 로
비교하며 Lumerical 없으면 자동 skip.

## 6. 파일 지도
```
run_inverse_design.sh            두 모드 런처 (실패 전파, SUCCESS 게이트)
requirements.txt                 numpy scipy autograd nlopt
IMPLEMENTATION_STATUS.md         명세 0~18 + 코드리뷰 대응표
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
  geometric_constraints.py       미분가능 solid/void 길이척도 제약 (power-mean)
  geometry_drc.py                독립 periodic 이진 DRC (opposing-boundary, 500 nm 판정)
  final_projection.py            exact binary → DRC → exact FDTD (transaction + provenance)
  evaluate_design.py, mapping_diagnostics.py
  tests/                         비-FDTD 검증 (46개)
tests/test_full_chain_filter_project_adfd.py   Lumerical AD/FD (없으면 skip)
```

## 7. 주의
- `bundle/perm_data.txt`(측정 데이터), `bundle/msopt/`(랩 helper)는 연구 자산입니다.
- 본 실행 전 **빈 GPU 확인**(`nvidia-smi`), 바쁜 GPU 강행 금지.
- **아직 Lumerical로 인증 필요(Go/No-Go)**: broadband+CV1 mesh 계약의 EM AD/FD, full-chain Fx/Fy/sum.
  자세한 미결·주의는 `IMPLEMENTATION_STATUS.md` 참고.
