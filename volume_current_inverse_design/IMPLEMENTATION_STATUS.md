# 500 nm 이진 설계 수정 — 구현 상태 (섹션별 대응표)

작성: 2026-07-24, 코드리뷰 반영 갱신 2026-07-25. 명세 "TaIrTe4 500 nm 이진 설계: 전체 수정 명세"(0~18)에 대응.

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
