# 500 nm 이진 설계 수정 — 구현 상태 (섹션별 대응표)

작성: 2026-07-24. 명세 "TaIrTe4 500 nm 이진 설계: 전체 수정 명세"(0~18)에 대응.

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
