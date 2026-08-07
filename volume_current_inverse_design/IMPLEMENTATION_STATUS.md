# 500 nm 이진 설계 수정 — 구현 상태 (섹션별 대응표)

작성: 2026-07-24, 코드리뷰 반영 갱신 2026-07-25. 명세 "TaIrTe4 500 nm 이진 설계: 전체 수정 명세"(0~18)에 대응.

## 스테이지 스케줄링 재작업 (2026-08-07) — 저-beta 탐색 부족 / 제약 과지배 / 미이진화
실런 관찰 3건(β=2가 topology를 못 찾고 넘어감; 제약이 저-beta FOM 상승을 막음; 종료 시 gray 잔존)에 대응.
`pytest inverse_design/tests/` = **94 passed** (신규 `test_stage_scheduling.py` 14개 포함).

- **스테이지별 예산**: `--beta-schedule`가 `beta[:maxeval[:min_evals]]`를 지원.
  기본 `2:40:12,4:16,8:16,16:12,32:10,64:10` — β=2에 큰 예산 + 높은 최소평가수(조기진급은 진짜
  plateau일 때만). 총 상한 104 evals (구 72). 평문 `2,4,8`은 기존처럼 `MAXEVAL` fallback.
- **constraint warm-up**: `VC_CONSTRAINT_START_BETA`(기본 8) 미만 스테이지는 길이척도 제약을
  MMA에 **걸지 않음**(순수 FOM 상승; gray 필드의 Zhou penalty는 의미가 약하고 FOM만 저해).
  g_solid/g_void는 매 eval 계산·기록되고 best-feasible 갱신은 여전히 실제 g<=0을 요구 →
  DRC/finalize 불변식 무변. warm-up 스테이지는 **절대 abort하지 않음**
  (`warmup_converged`/`maxeval_warmup`로 진급; `adaptive_stage/v2-constraint-warmup`).
- **binarization polish**: 사다리 종료 후 nominal 밀도의 gray fraction(노드가 (0.02,0.98)에
  있는 비율)이 `VC_GRAY_TOL`(기본 0.05) 초과이면 β를 2배씩 올리는 짧은 제약-활성 스테이지를
  `VC_BETA_CAP`(기본 256)까지 자동 추가(`VC_POLISH_MAXEVAL`=8). 매 eval `gray_fraction`을
  history에 기록하고 stop.json에 `final_gray_fraction`/`binarization_converged` 기록.
  (exact-binary projection + DRC + exact FOM 이 최종 판정이라는 원칙은 그대로.)
- 스케줄/warm-up/polish 파라미터 전부 contract에 들어가 config_hash에 반영 → **이전 attempt로의
  resume은 거부됨**(의도된 동작; 새 attempt로 시작).

### 같은 라운드 추가 (runtime-opt 워크스트림에서 이식 + 시각화)
- **`opt.set_xtol_rel(0.0)`**: nlopt 자체 xtol이 한 번의 미소 스텝 후 스테이지를 조기 종료시켜
  min_evals/feasibility 게이팅을 우회하던 것 차단 — 스테이지 종료 권한은 adaptive controller 단독.
- **`--rho-init`(`VC_RHO_INIT`, 기본 1e-2)**: CCSA inner penalty warm start. nlopt 기본
  rho_init=1.0은 **매 스테이지 첫 ~5 eval을 step RMS 4.45e-6→6.5e-4로 낭비**(실측, eval당
  ~15분 FDTD) → 1e-2면 eval 2-3부터 실질 이동. set 후 get_param으로 수락 검증, contract에
  `nlopt_rho_init` 기록.
- **iteration별 시각화** (`iteration_plots.py`, runtime-opt에서 이식): 매 objective eval마다
  `<run_root>/plots/design_it####_beta<b>.png`(nominal 밀도 + exact-binary 프리뷰 + rho 히스토그램)
  와 rolling `plots/progress.png`(Fx/Fy/F_sum, g_solid/void+feasible 음영, binarization/rails,
  latent step RMS; β-스테이지 경계 표시) 갱신. history.jsonl에도 `binarization`, `frac_rails`,
  `latent_step_rms`, `gray_fraction` 등 기록. 플롯 실패는 run을 죽이지 않음(try/except).
  테스트: `test_iteration_plots.py` 4개 + 소스 가드 2개 (`pytest` = **100 passed**).

## Lumerical full-chain AD/FD 실측 인증 완료 (2026-07-25, GPU) — P0-8 CLOSED
새 매핑 + solver-safe affine 층 + mapping VJP + FieldRegion adjoint(+ Yee 정합·주기소스 right-inverse·
27-color rho→eps 측정 Jacobian) **전 경로**를 실제 FDTD로 검증. broadband 소스(3–6µm), 4µm 분석.
`tests/test_full_chain_filter_project_adfd.py`, 각 케이스 ~10 solve.

| mesh 계약 | safe β=4 (probe_safe) | safe β=32 (고-beta) | exact β=8 (delta=0) |
|---|---|---|---|
| **CV1 / accuracy 5 / auto non-uniform (프로덕션, P0-8)** | **1.80%** | **2.33%** | **1.20%** |
| uniform / accuracy 2 / precise-volume-avg (대조) | PASS | 2.33% | 1.20% |

전 케이스 **AD/FD < 5% → PASS**. CV1 적용은 fsp로 직접 확인(`mesh type=auto non-uniform`,
`mesh accuracy=5`, `mesh refinement=conformal variant 1`, `source 3000–6000nm`).
CV1과 uniform 수치가 표시정밀도까지 거의 동일한 이유 = **FOM 영역(flake)이 5nm mesh override로 지배돼
bulk mesh 정밀도/refinement에 무감**(강건성; 버그 아님).

**결론**:
- 프로덕션 경로(probe_safe) + broadband 소스 + CV1 프로덕션 mesh 에서 adjoint gradient가 FD와 <2.5%로 정합
  → "broadband으로 이상하게 나온다"는 우려는 **실측으로 완전 해소**.
- exact-gradient 경로(검증전용, 프로덕션 미사용)도 구조 latent에서 1.2%로 정합. (앞선 21.5%는 near-uniform
  latent에서 FOM이 노이즈바닥 1e-16이라 나온 **테스트 결함**이었고, 구조 latent로 교정하여 해소.)
- Yee 정합/주기소스/rho→eps 코어는 안 바뀐 채 **실측 delta·index로 자동 적응 + 런타임 자기검증**(pairing<1e-13,
  leakage)하며, 위 full-chain에 포함되어 함께 인증됨.

## 테스트 개수 (source vs GitHub bundle 구분 — 리뷰 #4)
- **source tree** (`.../VOLUME_CURRENT_INVERSE_DESIGN_BASE_...`): `pytest inverse_design/tests/` = **69 passed**
  (source에는 구 Adam 파이프라인용 `test_adam/convergence/objectives`가 남아있음).
- **GitHub bundle** (`volume_current_inverse_design/`, 최소 실행 세트): `pytest inverse_design/tests/` = **60 passed**
  (구 Adam 모듈 3개와 그 테스트는 번들에서 제외). `pytest tests/` = **3개** — full-chain AD/FD로 **Lumerical 없으면 skip**.
- 아래 각 라운드의 "N passed"는 **source tree** 기준. 번들은 위 60이 기준.

## 코드리뷰 4차(2026-07-25, 7f8d726) 대응 — SUCCESS 완전성 + provenance 정책 + calibration 진단
- **#1 SUCCESS capped 플래그**: `minimum_solid/void/gap_width_capped`를 SUCCESS에도 기록(하한값 의미 보존).
- **#2 positive-path 강화**: manifest/SUCCESS의 `design_config_hash=="cfg-test"`, `design_attempt==1`,
  `mapping_identity==ident` + capped 플래그까지 assert.
- **#3 provenance 정책**: strict 모드에서 `config_hash`·`attempt`도 **필수**(없으면 missing_provenance);
  `--allow-unsafe-provenance`는 없어도 허용하되 `None` 기록 + **design_mapping_identity / current_mapping_identity 분리 기록**.
  누락 테스트 2개(config_hash/attempt) 추가.
- **#5 lint**: `test_finalize_provenance.py`의 미사용 `import pytest` 제거.
- **#6 450nm constraint (임의 튜닝 금지)**: 425/475/525nm 경계를 **진단 테스트로 pin**
  (`test_constraint_calibration.py`). 재현 결과 아래. tol/p는 실런 calibration 대상 — env 노브 기록.

### 450nm constraint calibration (재현, tol=1e-5·p=8 기본)
| 폭 | cells | g_solid | constraint | DRC solid(µm) | DRC |
|---:|---:|---:|---|---:|---|
| 425nm | 17 | +1.45e-3 | FAIL | 0.45 | FAIL |
| 475nm | 19 | -1.0e-5 (floor) | PASS | 0.50 | FAIL |
| 525nm | 21 | -1.0e-5 (floor) | PASS | 0.55 | PASS |

**진단**: 475·525 둘 다 penalty가 floor(≈0)라 `tol`/`p` 조정으로 분리 불가(필터반경↑ 또는 formulation 변경 필요) →
**임의 튜닝하지 않음**. 최종 DRC가 475nm를 차단하므로 **잘못된 SUCCESS는 불가능**. optimizer 효율(475nm를 best로 좇을 위험)은
실런에서 아래 노브로 보정:
- `VC_TOL_SOLID`, `VC_TOL_VOID` (기본 1e-5): 낮추면 near-floor 설계를 infeasible로.
- `VC_CONSTRAINT_PNORM` (기본 8): 높이면 worst-case에 더 민감.
- `VC_FILTER_RADIUS_UM` (기본 MFS): 키우면 length-scale 강제가 강해짐(가장 근본적).
판정 기준: 실런에서 425/475/525nm 재현값이 (425 infeasible, 475 infeasible, 525 feasible)이 되도록 위 노브를 조정 후 고정.

## 코드리뷰 3차(2026-07-25, b3948ba) 대응 — 성공 경로 NameError 수정
`pytest inverse_design/tests/` = **65 passed → (4차 후) 69 passed** (positive-path test 추가).

- **P0(성공경로) `had_feasible` NameError**: finalizer 성공 경로에서 `had_feasible`가 정의 안 된 채 manifest에
  쓰여 FDTD까지 끝내고 `SUCCESS.json`을 못 만들던 버그 → provenance 분기 앞에서 **항상 정의**하도록 수정.
- **positive-path integration test 추가**: FDTD evaluator를 mock해서 valid provenance + DRC PASS →
  `status=completed` + `SUCCESS.json`(Fx/Fy/F_sum) + artifact hash 까지 **실제 도달**을 검증(이 테스트가 없어 NameError를 놓쳤음).
- **P1-4 넓은 phase None→하한**: 규칙보다 충분히 넓어 측정이 cap된 경우 `minimum_*_width_um`을 **수치 하한값 + `*_capped: true`**로 기록(“측정 실패”처럼 보이던 None 제거). gap도 동일(넓은 moat는 PASS).
- **P1-5 provenance 완전화**: SUCCESS/manifest에 `design_config_hash`·`design_attempt`·`mapping_identity` 추가.
- 잔여: **P1-1**(450nm constraint) 및 **Lumerical full-chain/1-stage smoke**는 실행 환경 필요 — 코드로는 미해결(정직).

## 코드리뷰 2차(2026-07-25, d90fe21) 대응 — 실행 차단 버그 6개 P0 수정
`pytest inverse_design/tests/` = **64 passed** (integration guard 추가).

- **P0-1 경로 불일치**: runner/finalizer가 `--output`을 **CWD 기준**(`Path(args.output).resolve()`)으로 통일,
  쉘은 OUT을 절대경로화 → shell OUT == runner run_root. (source+동작 테스트)
- **P0-2 체크포인트 파일명**: `best_feasible.npz.tmp`(numpy가 `.npz` 재추가→replace 실패) →
  **`best_feasible.tmp.npz`**. (원자저장 테스트)
- **P0-3 launcher exit**: `exit "${frc:-5}"`(frc=0이면 0) 제거 → **rc≠0이면 그 rc, SUCCESS 없으면 exit 5**로 분리. (두 런처, bash 테스트)
- **P0-4 stage 실패 전파**: 분류된 stage 실패 시 runner가 **`SystemExit(6)`** → 런처가 abort(자동 finalize 안 함).
- **P0-5 provenance 필수화**: finalizer가 `had_feasible`/`code_hash`가 **없으면 거부**(옛 NPZ 통과 불가); `--allow-unsafe-provenance`로만 우회. (테스트)
- **P0-6 config/identity 검증**: `final_design.npz`에 **mapping_identity**(config+isolation+MFS+mode) 저장,
  finalizer가 현재와 불일치 시 **config_mismatch 거부**. `isolation_gap_um`을 MappingConfig에 포함. (테스트)
- **P1-2** 고립 단일-island의 gap을 void-width(moat)로 측정(더 이상 None인데 PASS 아님). **P1-4** DRC CLI 자동 임계 제거(비이진 입력 거부).
  **P1-5** `--no-fdtd`는 mapping을 env로 직접 빌드 → **Lumerical 없이 실행**(테스트가 그 경로로 통과). **P1-3** imageruler는 기록만(문구 정정). **P2-7** mapping 입력검증(radius>0, gap<period).
- 잔여(정직): **P1-1** 제약이 450nm를 아직 feasible로 볼 수 있음(150–400nm는 잡음) — 최종 DRC가 거르므로 잘못된 SUCCESS는 없으나 optimizer 효율 이슈; tol/p 보정은 실런 대상. **P0-8**(broadband/CV1 EM 인증)·full-chain은 여전히 Lumerical 필요.

## 코드리뷰(2026-07-25) 대응 — 반영된 수정
리뷰가 지적한 P0/P1을 수정하고 비-FDTD로 재검증했습니다 (`pytest inverse_design/tests/` = **55 passed**).

- **P0-1 DRC 재작성**: 측정 방식을 inscribed-disk → **opposing-boundary(local linewidth)** 로 교체.
  이제 all-solid/all-void는 **trivial gate로 FAIL**, 2µm 사각형/큰 island는 **PASS**(corner 오판 제거),
  얇은 bar/finger/ring·대각(수직폭)·seam·island-gap 은 정확히 FAIL. adversarial fixture 전부 통과.
- **P0-2 constraint**: 전역 mean → **power-mean(worst-case 보존)**. 국소 위반 희석 제거(단일노드 pmean/mean≈1.5e4),
  얇은 설계가 두꺼운 설계보다 penalty 큼 확인. gradient FD 일치 유지. (DRC가 최종 gate.)
- **P0-3 launcher**: `set -euo pipefail` + optimizer/DRC/FDTD 실패 시 nonzero exit, stale `final_design.npz` 선삭제,
  `SUCCESS.json` 있을 때만 exit 0.
- **P0-4 finalization transaction**: 진입 시 stale `SUCCESS.json` 삭제 + `status=pending`,
  실패 시 `status=failed`(category), `SUCCESS.json`은 **맨 마지막에만** 기록. (stale 제거 실측 확인.)
- **P0-5 provenance**: `final_design.npz`에 code/config hash·had_feasible 저장; finalizer가 **code_hash 불일치/미-feasible 거부**(실측 exit 2).
- **P0-6 AD/FD**: full-chain 테스트를 **safe-vs-safe / exact-vs-exact**(동일 함수) 로 분리. skip을 fixture로 견고화.
- **P0-7 240격자 깨짐**: `eqc_lib.physical_seed`를 240-latent→mapping 경로로 수정. (구 standalone adfd 스크립트는 번들에서 제외.)
- **P1-1** `--robust`는 미구현이라 즉시 `SystemExit`. **P1-2/3** resume가 best-feasible latent·objective·**beta**를 복원/기록.
  **P1-4** stage 예외를 분류(license/solver/numerical)하고 중단, 완주 못하면 `completed` 금지.
  **P1-5** code_hash에 model/eqc_lib/msopt/DRC/finalizer 등 전부 포함 + 누락 시 실패, config_hash에 maxeval/제약/nlopt 포함.
  **P1-6** attempt id = max+1, contract exclusive create. **P1-7** history `feasible`→`constraint_feasible`.
- **P2-3** `SUCCESS.json` 은 pass(boolean)와 measurement(float) 분리. **P2-4** artifact sha256 기록.
  **P2-1** 번들에서 무효 MANIFEST·캐시·legacy 런처 제외.

미해결/주의: **P0-8**(broadband+CV1 mesh 계약의 EM AD/FD 인증)과 full-chain AD/FD는 **Lumerical+GPU 필요** → Go/No-Go로 남김(문서화).
`--robust`는 의도적으로 미구현(raise). report generator(make_run_report)는 새 history.jsonl에 부분 대응.

---


## 범례
- ✅ **구현 + Lumerical 없이 실측 검증됨** (아래 pytest 실제 통과)
- 🔶 **코드 구현 완료, 최종 검증은 Lumerical+GPU+nlopt 필요** (Go/No-Go 게이트)
- 🤝 **동시 진행된 broadband-source 워크스트림에서 배선됨** (model/eqc_lib/msopt)
- ⏳ 부분/후속

## 검증 실측 결과 (재현: `python -m pytest inverse_design/tests/`)
```
48 passed   (periodic filter, 3-field mapping, 제약 gradient, DRC fixtures,
             solver-safe affine, resume hash, diagnostics)
```
개별 수치(직접 실행 확인):
- periodic conic filter: 상수보존 0.0, roll equivariance 2e-16, impulse=kernel 5e-19, VJP 3.3e-7
- PeriodicConstrainedMapping: fencepost 0.0, z-extrude 0.0, eroded≤nominal≤dilated, uniform 0.5→0.5, VJP 1e-6
- length-scale 제약 gradient(autograd vs FD): solid 9.5e-8, void 4.0e-8
- **DRC: solid/void bar 19px→FAIL, 21px→PASS (20px 보수적 FAIL); diagonal은 수직폭 인식(24px band FAIL, 32px PASS); seam-crossing/island-gap 정상**
- solver-safe affine: 모든 rail(0,1e-8,0.001,0.5,0.999,1)에서 probe∈[0,1], chain factor 일치
- model 로드: Nux,Nuy=240; x0=57600; analysis 4µm; source 3–6µm; 불변식 0.0

---

## 섹션별

### 0. 결론 (네 축)
1. 241→240 unique latent ✅ (§2)  2. 매핑 폐기+제약 ✅ (§1,4)  3. Adam→MMA/CCSA 🔶 (§5)
4. rail 문제 매핑 밖 solver-safe 층 ✅ (§6). 독립 DRC 게이트 ✅ (§8).
"성공=beta 24" 폐기, "성공=exact binary+DRC+exact FOM" ✅ (final_projection, §7,16).

### 1. 동결/폐기 ✅
- `symmetric_mapping.py`(FilterProject/SymmetricSeam) + 구 runner 4종 + Run E/F launcher:
  파일 상단 **[LEGACY 2026-07-24] 배너**. model이 legacy 매핑 선택 시 `DeprecationWarning`.
- `queue_F_after_E`의 `rm -rf` 재시작 **무력화**(실패 이력 보존, 새 attempt만).

### 2. 좌표/배열 ✅
- 2.1 physical 241×241×13 유지, **unique latent 240×240**(`Nux,Nuy,latent_shape,physical_shape,dx_um,dy_um`),
  불변식 `period==Nux*dx==6.0`, fencepost/z-extrude exact — 실측 0.0.
- 2.2 **endpoint-free periodic conic filter**(`bundle/periodic_filter.py`): 240 torus FFT 순환합성곱,
  impulse/roll/constant/ VJP 테스트 통과.
- 2.3 **진짜 periodic seed**(`make_periodic_latent_seed`): 저주파 Fourier 모드 합, seed 결정적, 평균 0.5.

### 3. 새 mapping API ✅
- `bundle/periodic_constrained_mapping.py` `PeriodicConstrainedMapping`:
  `filter_unique/project_unique/three_fields_unique/append_fencepost/extrude/physical`.
  eta_dilate<0.5<eta_erode, **eroded≤nominal≤dilated 보장**(테스트). `MAPPING_VERSION` 기록.
- air border: 전역 floor 제거. `PERIODIC_ISOLATION_GAP_UM`(기본 0)로 이름변경, latent 단계 마스크(하드컷 없음).
  기본 강제 border 없음 — DRC가 seam 넘어 검사.

### 4. 500 nm를 실제 제약으로 ✅
- `inverse_design/geometric_constraints.py`: Zhou2015 `constraint_solid/void`를 **내 periodic 필터+tanh(0.5)**로 래핑,
  `periodic_axes=[0,1]`, unique 240에서 계산, autograd gradient(FD 대조 ~1e-7).
- 4.1 residual `g = C - tol (<=0)`, contract에 식·scale·tol 기록.
- 4.2 "MGS" 3의미 분리: DRC가 solid width / void width / (옵션)disconnected-boundary gap 각각 측정·기록.
  기본 `min_gap=None`(not_applicable) — 실제 요구가 solid+void뿐이면 그대로.
- 4.3 discrete calibration: 19/20/21px fixture로 **보수적 규칙 고정**(21px=525nm 통과, 20px=500nm 보수적 FAIL).

### 5. optimizer 교체 🔶 (nlopt 설치 + Lumerical 필요)
- `inverse_design/run_constrained_inverse_design.py`: **NLopt LD_MMA**, bounds[0,1],
  목적 -(Fx+Fy), 부등식 g_solid,g_void, mapping VJP로 latent gradient.
- 5.1 기본식(nominal, 6 solve/eval) 구현. 5.2 robust(epigraph, 18 solve) 옵션 스텁·문서.
- 5.3 beta continuation(성공조건 아님). 5.4 sum 목적 유지.
- **미검증**: nlopt 미설치(=`requirements.txt`에 `nlopt>=2.7,<3` 추가, launcher preflight가 확인). 실행엔 Lumerical.

### 6. rail deadlock 제거 ✅ (arithmetic) / 🔶 (near-rail full-chain)
- `volume_current_evaluator.py`: **solver-safe affine 층** `rho_solver=delta+(1-2delta)*rho_geom`,
  gradient `*(1-2delta)`. `density_mode="probe_safe"|"exact"`. rail이어도 probe∈[0,1] — 실측.
- 6.2 `assert_mapping_contract`의 rail-fail 제거(정보성만). border floor·beta동결·step반감 등 회피책 제거.
- 6.3 one-sided Jacobian은 후속(문서화). **near-rail AD/FD full-chain은 §12.3/§18, Lumerical 필요**.

### 7. final_projection = 제조 승인기 ✅ (DRC/exact-binary) / 🔶 (FDTD)
- `inverse_design/final_projection.py`: best-feasible latent → **exact binary `(filter>=0.5)`**(beta 무관)
  → **독립 DRC** → 통과시만 241×241×13 exact mask → **exact-binary FDTD(Fx/Fy/sum)**.
  DRC 실패 → nonzero exit, `final_candidate_failed_drc.npz`만, SUCCESS 없음.
  산출물: `final_mask_unique/physical.npz, geometry_drc.json, exact_binary_fom.json, SUCCESS.json` 등.
  `np.unique(mask)⊂{0,1}` 확인. FDTD 부분만 Lumerical 필요(`--no-fdtd`로 DRC만도 가능).

### 8. 독립 geometry DRC ✅
- `inverse_design/geometry_drc.py`: **periodic disk-covering local-thickness**(EDT 기반, 최적화 제약과 다른 알고리즘),
  torus 3×3 타일, r-cap로 빠름(~1s), diagonal·seam 인식, disconnected-component gap(periodic union-find).
  JSON 출력(pass, min widths, violations, convention, version). **fixture 실측 통과.**

### 9. diagnostics/보고 ✅(core) / ⏳(report)
- `mapping_diagnostics.py`: rail-fail 제거, `unique_shape/fraction_exact_zero/one/between_rails/is_exact_binary` 추가.
- `verify_symmetric_mapping.py`(가짜 PASS) → **pytest로 대체**(위 실측). 실패시 nonzero.
- ⏳ `make_run_report.py`는 새 history.jsonl/DRC/stop-category 반영 후속(launcher는 실패해도 무해하게 `|| true`).

### 10. checkpoint/resume/provenance 🔶
- runner: `attempts/attempt_XXXX/{contract.json(immutable),stop.json,solver_*}`, append-only `history.jsonl`,
  `checkpoints/best_feasible.npz`, **code_hash+config_hash 불일치 시 resume 거부**(테스트), best **feasible** 별도.
  stop category(completed/geometry_infeasible/...). 실행 검증은 Lumerical 필요.

### 11. launcher 교체 🔶
- `inverse_design/launch_G_constrained_500nm_20260724.sh`: **preflight**(nlopt+deps import, 240/241 불변식,
  mapping 검사) → optimize → **DRC+exact-FDTD 게이트로만 finalize**. deterministic 실패 자동 retry 금지.

### 12. 테스트 ✅ (비-FDTD) / 🔶 (full-chain)
- `inverse_design/tests/`: periodic_unique_mapping, three_field_ordering, geometry_constraint_gradients,
  geometry_drc, boundary_safe_density, resume_contract, (갱신)mapping_diagnostics — **48 pass**.
- `tests/test_full_chain_filter_project_adfd.py`: 새 매핑+safe층 Fx/Fy/(Fx+Fy) AD/FD, 저/고 beta, near-rail —
  **Lumerical 없으면 자동 skip**(§18 Go/No-Go).

### 13. 파일별 작업 — 대부분 완료(§1~12 참조). ⏳ make_run_report/live_plots/plot_results는 후속.
### 14. 구현 순서 — 1~8 실측 완료, 9~10(짧은 solver smoke / 본 run)은 Lumerical 필요.
### 15. Go/No-Go — 비-FDTD 항목 전부 GREEN. FDTD 항목(near-rail chain, Fx/Fy/sum full-chain, solver preflight)은 **본 실행 전 필수**.
### 16. 최종 성공 판정 — `SUCCESS.json`은 exact_binary+DRC(solid/void/gap)+exact FOM 모두 true일 때만. beta·평균 binarization·radius로 대체 불가. ✅ 구현.
### 17. 하지 말 것 — rho_step 축소/ beta폭 축소/ border floor/ final beta 상향/ p05-chord PASS/ float를 binary로 저장 등 **전부 회피**(회피책 코드 제거·DRC 게이트로 대체).
### 18. 재검증 — full-chain 테스트 파일 제공, Lumerical에서 실행 필수.

---

## ⚠️ 동시 진행 broadband-source 워크스트림 (🤝, 이 리팩터와 별개·상보적)
사용자 병렬 세션이 **broadband source(3–6µm) + 4µm 분석 + material fit 2.7–13.2µm + mesh auto**를
`bundle/tairte4_volume_model.py`, `eqc_lib.py`, `bundle/msopt/Lumerical_utill.py`(add_source `single=`)에
배선함. 본 500nm 리팩터와 **함께 로드·컴파일·불변식 통과 확인**(analysis wl=[4.0] 유지 → evaluator 계약 무충돌).
FOM/분석은 4µm 그대로. 이 경로의 최종 물리 검증(광대역 펄스에서 4µm 추출)은 Lumerical AD/FD로 확인 필요.

## 실행 요약
```
# 1) 비-FDTD 검증 (지금 바로)
python -m pytest inverse_design/tests/ -q          # 48 passed
python inverse_design/geometry_drc.py <mask.npz> --min-solid-um 0.5 --min-void-um 0.5

# 2) 본 실행 (Lumerical v261 + license + GPU + nlopt 설치 후)
pip install 'nlopt>=2.7,<3'
bash inverse_design/launch_G_constrained_500nm_20260724.sh   # preflight→MMA→DRC+exact-FDTD 게이트
```
```
