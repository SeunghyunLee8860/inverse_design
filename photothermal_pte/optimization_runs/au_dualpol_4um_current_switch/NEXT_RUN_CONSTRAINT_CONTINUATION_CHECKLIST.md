# 다음 Au dual-polarization inverse-design 필수 체크리스트

작성일: 2026-08-31 KST

이 문서는 현재 실행을 변경하기 위한 문서가 아니다. 다음 inverse-design을 새로 시작하기 전에 반드시 검토하고, 각 항목을 fail-closed preflight로 구현한다.

## 1. 이번 beta=32 FOM 붕괴의 직접 증거

- beta=32 density-preserving 진입 FOM: 19.201017674 nA
- 진입 전류: Ia=+19.549101209 nA, Ib=-19.201017674 nA
- 진입 grayness: 0.254695090
- beta=32에서 한 번에 적용된 cap:
  - solid smooth DFM cap: 0.501934011
  - void smooth DFM cap: 0.172626635
  - grayness cap: 0.203756072
- 진입 구조의 normalized violation:
  - solid: +0.333333333
  - void: +0.176470588
  - grayness: +0.250000000
- 최초 저장 feasible 후보:
  - FOM: 12.575725240 nA
  - Ia=+17.434753617 nA, Ib=-12.575725240 nA
  - grayness: 0.194535612
  - normalized constraints: solid=-0.064913017, void=-0.064937867, grayness=-0.045252441
- 즉 제조 제약을 동시에 만족시키는 동안 FOM이 34.5% 감소했고, 손실은 주로 Eb에서 발생했다.

결론: beta sharpening 자체가 직접 원인이 아니다. beta=32 진입 시 DFM cap을 baseline의 0.75/0.85, grayness cap을 0.8로 동시에 줄여 처음부터 크게 infeasible한 MMA subproblem을 만든 것이 직접 원인이다.

## 2. 현재 구현에서 확인된 구조적 문제

### 2.1 Cap discontinuity

stage_design_caps()가 beta=32 진입과 동시에 solid/void/grayness cap을 크게 줄인다. density-preserving beta remap으로 전류는 보존했지만 feasible set은 보존하지 못했다. MMA가 objective 최적화보다 feasibility restoration에 대부분의 이동을 사용했다.

### 2.2 Stage 내부 FOM 보호 부재

beta transition 직전에는 FOM retention gate가 있지만, fixed-beta MMA 내부에서 FOM이 19.20 nA에서 12.58 nA로 떨어지는 것을 막는 gate가 없다. 다음 beta promotion도 absolute FOM retention을 요구하지 않는다.

### 2.3 Plateau가 동일한 물리 구조를 중복 집계할 가능성

stage_objective_progress()는 feasible callback 수와 FOM 값만 본다. latent가 미세하게 달라도 projection saturation 후 동일한 projected-density hash와 동일한 physics result가 나오면 같은 구조가 여러 feasible point로 집계될 수 있다. beta=32에서 동일한 12.575725240 nA 결과가 반복 관찰됐다.

다음 구현은 density_state_sha256 기준으로 unique physical states만 plateau window에 포함해야 한다. cache hit와 동일 projected density는 새 physics point로 세면 안 된다.

### 2.4 선택 후보와 최신 raw trial 보고 혼동

manifest latest는 selected best feasible 후보를 기록하지만 iteration은 전체 callback 수를 사용한다. 따라서 “eval 10 FOM 12.58 nA”처럼 실제 callback index와 후보가 섞여 보인다.

다음 manifest는 다음을 분리해야 한다.

- latest_raw_trial
- best_feasible_candidate
- optimizer_terminal_candidate
- restart_selected_candidate
- unique_physics_evaluation_count

### 2.5 250 nm와 500 nm 정의가 두 군데 존재

중요 정정:

- 현재 Lumerical production optimizer의 실제 상수는 solid=250 nm, void=250 nm, filter radius=250 nm이다.
- exact binary candidate audit도 opening radius 125 nm를 사용한 250 nm audit이다.
- contract.py와 dfm.py의 일부 legacy 기본값/함수명에는 500 nm가 남아 있다.
- smooth_500nm_physical_constraints라는 함수도 production 경로에서는 250 nm 인자를 받아 실행된다.

따라서 현재 run의 FOM 붕괴는 500 nm 제약 때문이 아니다. 다음 run 전에는 최소 feature의 single source of truth를 하나로 합치고, 함수/manifest 이름도 실제 단위와 일치시켜야 한다.

필수 preflight assertion:

- runtime minimum solid nm
- runtime minimum void nm
- filter radius nm
- exact-audit opening radius nm
- smooth-cap calibration nm
- manifest constraint names

위 여섯 값이 의도한 공정 규칙과 일치하지 않으면 Maxwell solve를 시작하지 않는다.

## 3. 다음 run의 권장 continuation 알고리즘

### 3.1 Beta와 constraint continuation을 분리

각 beta 진입 시에는 density-preserving remap 후 이전 stage에서 feasible했던 cap을 그대로 유지한다. 새 beta의 target cap을 즉시 적용하지 않는다.

각 beta 안에서 별도의 constraint-homotopy substage를 둔다.

1. 현재 구조가 feasible한 cap으로 시작
2. fixed cap에서 FOM을 수렴
3. cap을 작은 폭으로 축소
4. 동일 physical density에서 새 constrained subproblem 시작
5. 다시 feasibility와 FOM을 함께 수렴
6. target cap까지 반복

cap이 바뀌면 feasible set 자체가 바뀌므로 수학적으로 새 MMA subproblem이다. 단, latent/physical density는 그대로 넘기고 rho=0.5로 돌아가지 않는다.

### 3.2 Adaptive cap step

0.75, 0.85 같은 고정 큰 감소율을 사용하지 않는다. 다음 target까지의 cap 감소량은 직전 substage의 constraint margin, FOM sensitivity, feasibility restoration cost로 결정한다.

권장 acceptance:

- 새 cap 진입 normalized violation이 작도록 제한
- 한 cap step에서 FOM 손실이 허용 범위를 넘으면 step을 절반으로 줄여 재시도
- 최소 cap step에서도 FOM floor와 제조 target이 양립하지 않으면 자동 promotion하지 말고 Pareto conflict로 중단

### 3.3 FOM retention/Pareto gate

각 beta와 cap substage에 다음을 기록한다.

- stage-entry FOM
- best feasible FOM
- FOM retention ratio
- constraint improvement
- grayness improvement

다음 beta promotion에는 current sign뿐 아니라 최소 FOM retention도 요구한다. 예를 들어 retention floor는 처음부터 임의로 고정하지 말고, T-array 동일-pipeline benchmark와 feasible Pareto sweep으로 정한다.

제조 target을 만족시키기 위해 retention floor를 깨야 한다면 이를 숨기지 말고 다음 두 후보를 모두 보존한다.

- highest-FOM candidate
- most-manufacturable candidate

### 3.4 Robust manufacturability formulation 검토

단일 projected density에 morphology penalty를 갑자기 강화하는 방식 대신 eroded/intermediate/dilated three-field robust topology optimization을 우선 검토한다.

- intermediate field: nominal optical/thermal/electrical objective
- eroded field: 최소 Au feature robustness
- dilated field: 최소 void/spacing robustness
- 세 field 모두에서 current signs와 필요한 성능을 audit

관련 근거:

- Wang, Lazarov, Sigmund, On projection methods, convergence and robust formulations in topology optimization, DOI 10.1007/s00158-010-0602-y
- Guest, Prevost, Belytschko, Achieving minimum length scale in topology optimization using nodal design variables and projection functions, DOI 10.1002/nme.1064
- Christiansen and Sigmund, Inverse design in photonics by topology optimization: tutorial, arXiv:2008.11816

### 3.5 Unique-state plateau gate

plateau 판정은 callback 수가 아니라 unique density_state_sha256 수를 사용한다.

필수 조건:

- 최소 unique feasible states
- recent unique-state window
- no repeated cache-only points
- FOM improvement와 latent/projected-density movement을 둘 다 검사
- best feasible candidate가 entry FOM retention gate도 통과

### 3.6 AD-FD cadence

- beta entry에서 한 번
- beta 변경 시 한 번
- 최종 differentiable precursor에서 한 번
- exact binary는 별도 물리 재평가
- per-eval AD-FD 금지

late beta의 centered FD step은 commit 77e44b42 정책을 사용한다.

- beta<=16: h=0.0025
- beta=32: h=0.00125
- beta=64: h=0.000625
- beta=128: h=0.0003125
- 1% acceptance tolerance는 완화하지 않는다.

## 4. T-array baseline을 다음 run 전에 반드시 같은 pipeline으로 평가

친구의 T-array 결과가 Ea=+29 nA, Eb=-29 nA라면 현재 maximin objective와 직접 비교 가능한 강한 baseline이다. 단, 다음을 완전히 같게 맞춘다.

- incident/sample-plane power 정의
- Gaussian waist 정의와 source calibration
- flake, substrate, Au thickness
- crystal-axis mapping
- left/right terminal 및 current sign
- Au-Ta electrical/thermal contact
- same Lumerical R1.2 Maxwell
- same custom CUDA thermal/electrical PDE
- same dual-polarization FOM=min(Ia,-Ib)

T mask에 exact 250 nm solid/void audit도 적용한다. 동일 pipeline에서도 T FOM이 우수하면 다음 run은 uniform rho=0.5 하나만 쓰지 않는다.

- T-seeded start
- uniform start
- fixed-seed random/multistart
- 필요하면 8x8과 16x16 design-domain 비교

## 5. 다음 run 시작 전 fail-closed 체크

- [ ] 실제 공정 최소 feature/spacing을 250 nm 또는 500 nm 중 하나로 사용자 확인
- [ ] single DFM contract로 통합
- [ ] T-array same-pipeline dual-pol benchmark 완료
- [ ] cap substage가 stage entry를 크게 infeasible하게 만들지 않는 테스트
- [ ] in-stage FOM retention/Pareto gate
- [ ] unique projected-density hash plateau gate
- [ ] raw/best/terminal/restart 후보 분리 기록
- [ ] beta-scaled AD-FD step과 unchanged 1% gate
- [ ] exact-binary ordinary dispersive Au 재평가
- [ ] final 100-to-50 nm optical/PDE convergence
- [ ] GPU/license watchdog와 latest-successful checkpoint
- [ ] rho=0.5 초기화 외 T seed와 multistart 포함

## 6. 현재 run 처리 원칙

현재 beta=32 run은 비교 자료로 계속 보존한다. 다음 run에서는 위 문제를 수정한 별도 output root를 사용한다. 현재 run의 manifest, checkpoint, AD-FD certificate를 덮어쓰지 않는다.
