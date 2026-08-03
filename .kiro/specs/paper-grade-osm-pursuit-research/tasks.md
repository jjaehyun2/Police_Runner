# Implementation Plan: Paper-Grade OSM Pursuit Research

## Overview

Python 기반 기존 OSM 추격 코드를 `pursuit_evasion_rl/research/` 패키지로 단계적으로 마이그레이션하고, 자산 보존·지도 계보·hybrid SMDP·masked MAPPO·완전 재개·쌍체 통계·claim gate·논문 산출물을 하나의 재현 가능한 파이프라인으로 연결한다. 모든 기본 검증은 외부 OSM/LLM 호출 없이 fixture와 고정 cache로 실행한다. 실제 장시간 실행은 offline unit/PBT/integration/smoke gate가 모두 통과하고, 해당 실행의 resource estimate와 frozen protocol hash가 승인된 경우에만 시작한다.

## Tasks

- [x] 1. 기존 자산, 도메인 schema 및 보호 경계를 구현한다
  - [x] 1.1 연구 도메인 모델과 canonical SHA-256 기반을 구현한다
    - `pursuit_evasion_rl/research/domain.py`, `canonical.py`, `errors.py`에 Evidence/Condition/Protocol/Map/Run/Claim enum·dataclass, exactly-one validator, canonical JSON 직렬화, SHA-256 및 구조화 오류 record를 구현한다.
    - persisted model이 `schema_version`과 content hash를 가지며 `failed`, `not_run`, `null` 값도 손실 없이 직렬화되게 한다.
    - 완료 검증: canonical round-trip, key-order 독립 hash, 비유한 수치 및 중복 분류 거부 unit test가 통과한다.
    - _Requirements: 1.3, 1.7, 1.9, 4.6, 4.7, 14.2, 19.1_
    - _Correctness Properties: 1, 25, 34_

  - [x] 1.2 기존 구현과 Prior_Result 자동 감사기를 구현한다
    - `pursuit_evasion_rl/research/audit.py`와 `cli/audit.py`에 환경·관측·정책·학습·평가·메트릭·checkpoint 파일 및 자동 시험 상태 inventory, `training.log`의 0.99 이동창 제한, 버그 우선순위 기반 Prior_Result 분류를 구현한다.
    - 감사 결과를 typed Evidence_Record와 Claim_Register로 생성하고 구현 존재·시험 통과·실험 관측·논문 주장을 별도 evidence type으로 강제한다.
    - 완료 검증: 알려진 0.99 결과가 `preliminary`이며 escape-bug 입력은 항상 `invalid`이고 불완전 record는 등록 실패한다.
    - _Requirements: 1.1–1.9_
    - _Correctness Property: 1_

  - [x] 1.3 `best_v2` 불변 보존 및 쓰기 차단을 구현한다
    - `pursuit_evasion_rl/research/preservation.py`에 `checkpoints/osm_mappo/best_v2.pt`의 실행 시점 SHA-256/크기 기록, 동일 hash 읽기 전용 보존본, 원본·보존본 사전 실행 재검증 및 보호 경로 overlap/write 차단을 구현한다.
    - 설계의 감사 hash `391527bb6914ca6c6e0d138472570e927330ba50354c87190e1647c22c016cd9`는 기대값을 하드코딩한 성공 증거가 아니라 재계산과 대조할 관측값으로 기록한다. 신규 checkpoint는 오직 `artifacts/research/checkpoints/<condition>/<seed>/<run_id>/`에 쓰며 기존 `best_v2`를 절대 덮어쓰지 않는다.
    - 완료 검증: temp filesystem 복제/권한/sealing test와 원본·보존본 1-byte 변조 및 보호 경로 출력 거부 test가 통과한다.
    - _Requirements: 2.1–2.6_
    - _Correctness Property: 2_

  - [x]* 1.4 Evidence 분류 불변식 property test를 작성한다
    - `tests/research/properties/test_property_01_evidence_classification.py`에 Hypothesis 생성기를 만들고 최소 100 examples로 분류 배타성, escape-bug 우선순위, evidence type 대체 금지를 검증한다.
    - **Property 1: Evidence classifications are exclusive, precedence-safe, and type-safe**
    - 완료 검증: property 주석이 지정 형식이고 변이된 중복/누락/불일치 record가 최소 반례로 거부된다.
    - **Validates: Requirements 1.1–1.7, 1.9, 4.6–4.7, 19.1**

  - [x]* 1.5 보호 checkpoint identity property test를 작성한다
    - `tests/research/properties/test_property_02_checkpoint_identity.py`에 artifact bytes/path/usage 생성기를 구현해 동일 hash와 분리 경로만 실행을 허용하는지 최소 100 examples로 검증한다.
    - **Property 2: Preserved checkpoint identity gates every run**
    - 완료 검증: 원본/보존본/출력 경로 중 하나만 perturb해도 실행 gate가 닫히고 오류 hash가 보존된다.
    - **Validates: Requirements 2.3–2.6**

- [x] 2. 지도 registry, 공간 분리 및 오프라인 snapshot 계층을 구현한다
  - [x] 2.1 Actual OSM/fixture와 Interior/Boundary map registry를 구현한다
    - `pursuit_evasion_rl/research/maps/registry.py`와 schema에 data kind·scenario exactly-one, Actual OSM query/acquisition/source/preprocessing/CRS/raw/network/source-edge hash, fixture generator/params/seed/purpose를 구현한다.
    - Interior는 boundary arc 0개와 `{capture, timeout}`, Boundary는 경계 arc와 `{capture, escape, timeout}`만 허용하고 성능 aggregate를 kind/scenario별로 분리한다.
    - 완료 검증: provenance와 boundary marker를 하나씩 잘못 지정한 등록이 실패하고 올바른 두 시나리오의 outcome domain이 고정된다.
    - _Requirements: 5.1–5.10, 19.2_
    - _Correctness Property: 5_

  - [x] 2.2 polygon+buffer+source-edge spatial split을 구현한다
    - `pursuit_evasion_rl/research/maps/splits.py`에 train/validation/test polygon 비중첩, metric buffer, OSM source-edge signature 교집합 0, 최소 2개 비학습 도시 확인 및 validation report hash를 구현한다.
    - tuning API에서 test/cross-city handle을 받지 못하게 typed split view를 제공하고 실패 split을 일반화 evidence에서 제외한다.
    - 완료 검증: polygon overlap, buffer 위반, 공유 source edge를 각각 단독 주입한 test가 정확한 오류 코드로 실패한다.
    - _Requirements: 6.1–6.9, 19.3_
    - _Correctness Properties: 6, 7_

  - [x] 2.3 provenance가 고정된 다중 도시 map snapshot/cache adapter를 구현한다
    - `pursuit_evasion_rl/research/maps/snapshots.py`, `configs/research/maps/*.yaml`, `artifacts/research/maps/registry.json` schema에 대전 split과 최소 2개 zero-shot 도시의 immutable raw/network snapshot import·hash 검증을 구현한다.
    - 외부 OSM fetch는 명시적 optional integration command로 분리하고 기본 경로는 `cache/` 또는 test fixture만 읽으며, fetch 결과의 서비스·시각·입출력 hash를 별도 provenance로 기록한다.
    - 완료 검증: 네트워크 차단 상태에서 cached snapshot import가 재현되고 cache miss는 외부 요청 대신 명시적 `not_run`을 산출한다.
    - _Requirements: 5.3, 6.1–6.8, 19.11–19.12_
    - _Correctness Properties: 5, 6, 7_

  - [x]* 2.4 map registry/snapshot 오프라인 integration test를 작성한다
    - `tests/research/test_map_registry_integration.py`에 fixture와 cached Actual OSM metadata 등록, Interior/Boundary 표 분리, optional external provenance contract를 자동 검증한다.
    - 완료 검증: 외부 네트워크 없이 test가 통과하고 실제 OSM과 fixture 결과를 합치려는 aggregate가 실패한다.
    - _Requirements: 5.1–5.10, 6.1, 19.2, 19.11–19.12_
    - _Correctness Properties: 5, 6_

  - [x]* 2.5 map registration property test를 작성한다
    - `tests/research/properties/test_property_05_map_registration.py`에 map provenance/boundary/outcome 생성기를 구현해 최소 100 examples로 kind·scenario 배타성과 결과 분리를 검증한다.
    - **Property 5: Map registration is exclusive and scenario-consistent**
    - 완료 검증: kind/scenario/provenance/outcome의 각 단일 결함이 등록 또는 집계에서 거부된다.
    - **Validates: Requirements 5.1–5.9, 19.2**

  - [x]* 2.6 spatial split property test를 작성한다
    - `tests/research/properties/test_property_06_spatial_splits.py`에 작은 polygon, buffer, source-edge set 생성기를 구현해 최소 100 examples로 세 disjoint 조건의 필요충분 gate를 검증한다.
    - **Property 6: Spatial splits are geometrically and genealogically disjoint**
    - 완료 검증: 검증 실패 split은 항상 일반화 eligible set에서 제외된다.
    - **Validates: Requirements 6.2, 6.8, 19.3**

- [x] 3. root 실험 자산을 package API로 마이그레이션하고 회귀 shim을 둔다
  - [x] 3.1 `_Graph`, `GoalEvader`, `EncirclementPolice`를 baseline package로 이동한다
    - `pursuit_evasion_rl/research/policies/baselines.py`로 구현을 옮기고 고정 parameter/목표 배정 schema를 추가하며 `demo_pursuit.py`에는 경고를 내는 호환 import shim만 남긴다.
    - 이전 pickle/import 경로 호환성을 유지하되 새 코드가 root 모듈을 역으로 import하지 않게 한다.
    - 완료 검증: 동일 fixture/seed에서 이전·신규 policy의 action/state/outcome canonical hash가 일치한다.
    - _Requirements: 1.1, 11.3_
    - _Correctness Property: 17_

  - [x] 3.2 `osm_obs_aug.py`를 observation variant package로 이동한다
    - `pursuit_evasion_rl/research/variants/observations.py`에 21D/28D adapter contract를 배치하고 root `osm_obs_aug.py`는 호환 re-export shim으로 축소한다.
    - 7개 추가 특성의 계산 시점, clip/near radius, 정규화와 canonical config hash를 노출한다.
    - 완료 검증: 동일 fixture에서 shim과 package adapter가 bitwise 동일한 28D vector를 생성한다.
    - _Requirements: 9.1–9.2_
    - _Correctness Property: 13_

  - [x] 3.3 root script migration shim과 deprecation 경계를 구현한다
    - `pursuit_evasion_rl/research/migration.py`에 legacy checkpoint/config 변환과 import compatibility를 구현하고 root script가 신규 package entry point를 호출하도록 단계적 shim contract를 만든다.
    - 신규 package 내부에서 `demo_pursuit`, `osm_obs_aug`, root training/eval script를 import하는 순환을 static 검사로 금지한다.
    - 완료 검증: legacy CLI argument와 checkpoint metadata를 변환한 결과의 canonical hash가 golden fixture와 일치한다.
    - _Requirements: 1.1, 11.2, 14.2_
    - _Correctness Properties: 17, 25_

  - [-]* 3.4 package migration regression test를 작성한다
    - `tests/research/test_migration_regression.py`에 baseline, observation, legacy config/checkpoint shim 전후의 state/action/outcome/hash 비교와 import-cycle 검사를 구현한다.
    - 완료 검증: 기존 `tests/test_osm_demo_environment.py`, observation relabel, runner pair, coarsening 회귀 suite가 함께 통과한다.
    - _Requirements: 1.1, 9.2, 11.3, 19.4_
    - _Correctness Properties: 13, 17_

- [x] 4. hybrid SMDP와 centralized masked MAPPO 학습 코어를 구현한다
  - [x] 4.1 비동기 decision epoch 기반 hybrid SMDP transition을 구현한다
    - `pursuit_evasion_rl/research/smdp.py`에 `DecisionTransition`, 경찰별 pending buffer, `duration_steps`, discounted reward, `gamma**tau` bootstrap, terminal closure와 zero-time virtual hop bound를 구현한다.
    - directed polyline 호길이 이동과 교차로에서만 physical segment가 바뀌는 invariant를 기존 환경 adapter에 연결한다.
    - 완료 검증: 손계산 비동기 trace에서 누적 보상·duration·bootstrap과 terminal pending closure가 일치한다.
    - _Requirements: 13.7, 19.4_
    - _Correctness Property: 19_

  - [x] 4.2 exact stored-mask centralized critic MAPPO를 구현한다
    - `pursuit_evasion_rl/research/policies/masked_mappo.py`에 shared actor, officer one-hot, joint/global centralized critic, `MaskedCategoricalFactory`, masked PPO loss와 valid-action entropy를 구현한다.
    - rollout이 저장한 mask bytes만 PPO log-prob 재계산에 사용하고 empty support, selected-invalid, old/new support drift, nonfinite loss에서 update 전체를 fail-closed한다.
    - 완료 검증: unchanged parameter의 old/new log-prob 일치, invalid probability 0, 중앙 critic 입력과 actor 비특권 입력 분리가 통과한다.
    - _Requirements: 11.2, 19.3–19.4_
    - _Correctness Property: 20_

  - [x] 4.3 실제 trainer와 root train/eval CLI wrapper를 연결한다
    - `pursuit_evasion_rl/research/training/trainer.py`, `cli/train.py`, `cli/evaluate.py`에 SMDP rollout, centralized critic update, validation-only checkpoint selection과 새 output root를 연결한다.
    - `train_osm_pursuit.py`, `eval_trained.py`, `diag_episode.py`, `render_zoom.py`는 package CLI를 호출하는 호환 wrapper로 전환하고 `training/algorithms.py::_ppo_update`를 논문 trainer가 호출하지 못하게 한다.
    - 완료 검증: tiny fixture 2-update 학습이 legal action만 생성하고 test/cross-city handle 없이 새 run directory에 checkpoint를 쓴다.
    - _Requirements: 2.5, 6.3, 11.2, 14.1–14.2, 15.6_
    - _Correctness Properties: 7, 19, 20, 25_

  - [x] 4.4 SMDP/MAPPO targeted unit test를 작성한다 (test_smdp_task_4_1.py/test_masked_mappo_core.py로 구현, 내용 동등)
    - `tests/research/test_smdp.py`, `test_masked_mappo.py`에 async duration, virtual zero-time, terminal closure, random mask support, all-false/selected-invalid rejection, centralized critic shape 검사를 구현한다.
    - 완료 검증: CPU deterministic fixture에서 bitwise 재현되고 CUDA가 있으면 tolerance 기반 보조 test가 통과한다.
    - _Requirements: 13.7, 19.3–19.4_
    - _Correctness Properties: 19, 20_

  - [x]* 4.5 hybrid SMDP property test를 작성한다
    - `tests/research/properties/test_property_19_hybrid_smdp.py`에 작은 directed network, speed, duration, reward sequence 생성기를 구현해 최소 100 examples로 이동·decision·discount invariant를 검증한다.
    - **Property 19: Road-arc transitions obey the hybrid SMDP**
    - 완료 검증: 초과 이동, 중간 segment 변경, 유시간 virtual hop, 잘못된 `gamma**tau`를 각각 탐지한다.
    - **Validates: Requirements 13.7, 19.4**

  - [x]* 4.6 action-mask support property test를 작성한다
    - `tests/research/properties/test_property_20_action_mask_support.py`에 logits/nonempty mask/transition 생성기를 구현해 최소 100 examples로 sampling과 PPO recomputation의 동일 support·bytes·log-prob를 검증한다.
    - **Property 20: Action-mask support is identical during sampling and PPO recomputation**
    - 완료 검증: 저장 mask 한 bit 변경 또는 invalid action 주입 시 전체 update가 실패한다.
    - **Validates: Requirements 19.3–19.4**

  - [x]* 4.7 legacy unmasked PPO 사용 금지 architecture test를 작성한다 (구현·실행 완료: tests/research/test_training_architecture.py, 7 tests, AST 기반 검사로 masked_mappo.py의 osm_demo import 부재 및 research 패키지 전체의 _ppo_update/legacy 모듈 참조 부재를 실측 확인, fixture 주입 검출 테스트 포함)
    - `tests/research/test_training_architecture.py`에 AST/import 검사를 구현해 연구 trainer가 범용 `_ppo_update` 또는 환경에서 재계산한 mask를 사용하는 경로를 차단한다.
    - 완료 검증: 금지 호출을 fixture source에 주입하면 test가 실패하고 package trainer에는 위반이 0개다.
    - _Requirements: 11.2, 19.3–19.4_
    - _Correctness Property: 20_

- [x] 5. 관측·보상·배치·안정화 변형과 수치 기준선을 구현한다
  - [x] 5.1 21D/28D observation contract와 capacity accounting을 완성한다
    - `variants/observations.py`에 ID-independent 21D/28D schema, decision 직전 snapshot, finite `[0,1]` validation, parameter/FLOP 계산 및 capacity-matched hidden-width 탐색을 구현한다.
    - 완료 검증: topology-preserving ID relabel에서 vector 의미와 action mask가 유지되고 21/28 차원 및 용량 차이가 정확히 보고된다.
    - _Requirements: 9.1–9.2, 9.6_
    - _Correctness Property: 13_

  - [x] 5.2 reward component registry와 ablation을 구현한다
    - `pursuit_evasion_rl/research/variants/rewards.py`에 team/own/time/capture component, symmetric/asymmetric retreat, leave-one-component-out 조건과 component-level trace를 구현한다.
    - terminal semantics는 제거하지 않고 모든 reward 결과 schema가 capture와 stability metric reference를 요구하게 한다.
    - 완료 검증: hand sequence에서 full 합, 각 제거 차이와 음수 progress multiplier가 정확히 일치한다.
    - _Requirements: 9.3–9.5_
    - _Correctness Property: 14_

  - [x] 5.3 placement curriculum과 2x2 stabilization을 구현한다
    - `pursuit_evasion_rl/research/variants/placement.py`, `stabilization.py`에 global/ring/mixed 배치, 별도 RNG stream, feasible radius, soft U-turn penalty, hysteresis state/K/margin과 `off/on × off/on` 네 조건을 구현한다.
    - 규칙·분포·RNG·wrapper state를 condition hash/checkpoint에 포함한다.
    - 완료 검증: 동일 seed 배치 재현, 정확히 네 factorial arm, single-factor diff와 hysteresis 유지/전환 경계 test가 통과한다.
    - _Requirements: 10.1–10.7_
    - _Correctness Properties: 15, 16_

  - [x] 5.4 MAPPO 및 heuristic 수치 기준선 registry를 구현한다
    - `pursuit_evasion_rl/research/policies/baselines.py`에 `legacy_shared_ppo`/budget-matched MAPPO, Encirclement, directed shortest-path, greedy-intercept를 등록하고 parameter·selection·cost schema를 강제한다.
    - 모든 주 stratum에서 동일 EpisodeCase와 metric schema를 사용하며 열세/null 결과를 필터링하지 않는다.
    - 완료 검증: 누락 baseline/비교 budget mismatch/metric schema 차이가 primary matrix validation을 실패시킨다.
    - _Requirements: 11.1–11.6_
    - _Correctness Property: 17_

  - [x] 5.5 Condition/ablation matrix와 표본·자원 계획 factory를 구현한다
    - `pursuit_evasion_rl/research/variants/factory.py`, `pursuit_evasion_rl/research/budget.py`, `configs/research/conditions/*.yaml`에 observation/reward/placement/stabilization/baseline matrix와 canonical Condition hash를 구현한다.
    - 기본 5 independent seeds/500 episodes와 Resource_Ceiling 기반 전체 조건 공통 3/100 reduced rule을 자동 집행하고 3/100 미만은 exploratory, 미실행은 `not_run`으로 생성한다.
    - 완료 검증: checkpoint time 중복이 seed 수를 늘리지 않으며 축소가 결과를 보기 전 deterministic하게 전체 matrix에 적용된다.
    - _Requirements: 7.1–7.8, 9.1–9.7, 10.1–10.5, 11.1–11.2_
    - _Correctness Properties: 8, 13–17_

  - [x]* 5.6 variant와 baseline unit test를 작성한다
    - `tests/research/test_variants.py`, `test_baseline_registry.py`에 28D field contract, reward examples, placement RNG, 2x2 matrix, Encirclement parameters, capacity/budget fairness 검사를 구현한다.
    - 완료 검증: 조건 한 요소 외 변경 또는 필수 baseline 제거 시 schema validation이 실패한다.
    - _Requirements: 9.1–9.7, 10.1–10.7, 11.1–11.6_
    - _Correctness Properties: 13–17_

  - [x]* 5.7 observation contract property test를 작성한다
    - `tests/research/properties/test_property_13_observation_contracts.py`에 network state와 bijective ID relabel 생성기를 구현해 최소 100 examples로 21D/28D finite normalization, 의미·mask 불변성과 capacity 감지를 검증한다.
    - **Property 13: Observation variants preserve contracts and ID independence**
    - 완료 검증: ID 자체를 feature로 누출하거나 차원/범위를 깨뜨린 변이가 탐지된다.
    - **Validates: Requirements 9.1–9.2, 9.6, 19.4**

  - [x]* 5.8 reward decomposition property test를 작성한다
    - `tests/research/properties/test_property_14_reward_variants.py`에 finite distance/event sequence 생성기를 구현해 최소 100 examples로 full sum, leave-one-out 및 retreat arm 관계를 검증한다.
    - **Property 14: Reward variants differ only in their declared component**
    - 완료 검증: 비선언 component 변화나 capture/stability 누락 결과 row가 거부된다.
    - **Validates: Requirements 9.3–9.5**

  - [x]* 5.9 ablation matrix property test를 작성한다
    - `tests/research/properties/test_property_15_ablation_matrix.py`에 condition matrix 생성기를 구현해 최소 100 examples로 필수 arm, one-factor control, paired budget, placement/stabilization provenance를 검증한다.
    - **Property 15: Ablation matrices are complete and one-factor controlled**
    - 완료 검증: 2x2 조합 누락·중복 또는 undeclared factor drift가 탐지된다.
    - **Validates: Requirements 9.1, 10.1–10.5**

  - [x]* 5.10 stabilization trade-off property test를 작성한다
    - `tests/research/properties/test_property_16_stabilization_tradeoffs.py`에 stabilization result row 생성기를 구현해 최소 100 examples로 metric 완전성과 capture↑/plausibility↓ trade-off flag를 검증한다.
    - **Property 16: Stabilization trade-offs cannot be hidden**
    - 완료 검증: 상충 결과에서 결과·한계 record 중 하나라도 flag가 없으면 실패한다.
    - **Validates: Requirements 10.6–10.7**

  - [x]* 5.11 baseline fairness property test를 작성한다
    - `tests/research/properties/test_property_17_baseline_fairness.py`에 stratum/policy/budget/result 생성기를 구현해 최소 100 examples로 baseline coverage, MAPPO fairness, 공통 metric, 열세 보존과 cost semantics를 검증한다.
    - **Property 17: Baseline coverage and fairness hold for every primary stratum**
    - 완료 검증: 열세 row 삭제 또는 missing/non-applicable cost 혼동이 실패한다.
    - **Validates: Requirements 11.1–11.2, 11.4–11.6**

- [x] 6. immutable manifest와 완전 재개를 구현한다
  - [x] 6.1 content-addressed run manifest store를 구현한다
    - `pursuit_evasion_rl/research/runs/manifest.py`에 unique run ID, code/dirty tree/dependency/runtime/device/map/split/seed/input/output hash, lineage와 atomic seal/fork를 구현한다.
    - completed manifest와 참조 artifact는 in-place 수정할 수 없고 변경 시 child run으로 분기하며 `failed/not_run/null`도 seal한다.
    - 완료 검증: 완료 field/artifact mutation이 원 run 변경이 아니라 새 child identity와 계보를 만든다.
    - _Requirements: 14.1–14.4_
    - _Correctness Property: 25_

  - [x] 6.2 full ResumeState checkpoint IO를 구현한다
    - `pursuit_evasion_rl/research/training/checkpoint.py`에 actor/critic, optimizer/scheduler/scaler, indices, Python/NumPy/Torch CPU·device/environment/placement/sampler RNG, curriculum/hysteresis, pending SMDP, normalization과 parent run 저장·복원을 구현한다.
    - atomic temp-write, content hash, config/code/map/protocol compatibility를 검증하고 불완전·불일치 checkpoint는 새 run lineage break로 처리한다.
    - 완료 검증: 모든 필드 round-trip 및 각 필드 단독 누락/변조 거부 test가 통과한다.
    - _Requirements: 14.5–14.8, 19.6_
    - _Correctness Property: 26_

  - [x] 6.3 interruption equivalence harness를 구현한다
    - `pursuit_evasion_rl/research/training/equivalence.py`에 tiny deterministic N-update 연속 실행과 K-save/load-(N-K) 실행의 state/metric trace 비교를 구현한다.
    - CPU bitwise 및 protocol의 GPU `rtol/atol` 경로를 분리하고 divergence를 성공으로 삼지 않고 artifact hash가 있는 실패 record로 남긴다.
    - 완료 검증: 모든 `0 <= K <= N` interruption point가 연속 실행과 동등하고 RNG/state 하나를 생략한 변이는 실패한다.
    - _Requirements: 14.7, 19.6_
    - _Correctness Property: 27_

  - [x]* 6.4 manifest/checkpoint integration test를 작성한다
    - `tests/research/test_manifest_checkpoint.py`, `test_resume_equivalence.py`에 seal/fork, full RNG round-trip, incompatible contract, torn write fallback, interruption equivalence를 구현한다.
    - 완료 검증: temp filesystem과 tiny CPU trainer에서 재현되며 device RNG는 사용 가능한 장치만 조건부 자동 검증한다.
    - _Requirements: 14.1–14.8, 19.6_
    - _Correctness Properties: 25–27_

  - [x]* 6.5 immutable manifest property test를 작성한다
    - `tests/research/properties/test_property_25_run_manifests.py`에 run/artifact mutation 생성기를 구현해 최소 100 examples로 manifest 완전성, content addressing과 fork-on-mutation을 검증한다.
    - **Property 25: Run manifests are complete, content-addressed, and fork on mutation**
    - 완료 검증: 필수 hash 누락과 completed identity의 in-place 변경이 모두 거부된다.
    - **Validates: Requirements 14.1–14.2, 14.4**

  - [x]* 6.6 ResumeState round-trip property test를 작성한다
    - `tests/research/properties/test_property_26_resume_roundtrip.py`에 tiny tensor/optimizer/RNG/curriculum/wrapper/pending state 생성기를 구현해 최소 100 examples로 verified save/load와 compatibility gate를 검증한다.
    - **Property 26: Resume state serialization is a complete round trip**
    - 완료 검증: 필수 state 한 종류를 제거하거나 hash/contract를 변경하면 resume 대신 lineage break가 생성된다.
    - **Validates: Requirements 14.5–14.6, 14.8, 19.6**

  - [x]* 6.7 interruption equivalence property test를 작성한다
    - `tests/research/properties/test_property_27_interruption_equivalence.py`에 tiny deterministic run과 `K,N` 생성기를 구현해 최소 100 examples로 continuous/resumed final state와 trace 동등성을 검증한다.
    - **Property 27: Interruption and resumption are observationally equivalent**
    - 완료 검증: protocol tolerance를 벗어난 단일 tensor/metric/RNG 차이가 `INTERRUPTION_DIVERGENCE`를 만든다.
    - **Validates: Requirements 14.7, 19.6**

- [x] 7. frozen paired evaluation, 여섯 경찰 metric 및 계층 통계를 구현한다
  - [x] 7.1 immutable EpisodeCase 생성과 paired evaluator를 구현한다
    - `pursuit_evasion_rl/research/evaluation/paired.py`에 map hash, 경찰 6대·도주자 배치, fugitive RNG, scenario/termination config를 봉인한 EpisodeCase와 policy replay를 구현한다.
    - Interior/Boundary 및 in-region/held-out/cross-city stratum을 분리하고 한 contract field라도 다르면 paired 표시와 쌍체 통계를 거부한다.
    - 완료 검증: 동일 case의 다중 policy replay hash가 일치하며 각 contract field perturbation이 이름 있는 mismatch를 만든다.
    - _Requirements: 5.8–5.10, 6.4–6.7, 8.1, 8.8_
    - _Correctness Properties: 7, 9_

  - [x] 7.2 여섯 경찰 행동·포위·공헌 metric을 구현한다
    - `pursuit_evasion_rl/research/metrics/behavior.py`에 per-officer U-turn/revisit/switch/stay/zero-displacement/approach/retreat/capture-entry/exit-block/leave-one-off와 containment geometry를 구현한다.
    - officer 0..5 raw row, team mean/median, metric 방향별 worst officer와 SI unit/schema를 생성한다.
    - 완료 검증: hand trace/reference 계산과 회전·이동 불변 containment 예제가 일치한다.
    - _Requirements: 13.1–13.6, 13.8_
    - _Correctness Properties: 21–23_

  - [x] 7.3 physical plausibility validator와 failure-case ledger를 구현한다
    - `pursuit_evasion_rl/research/metrics/physical.py`에 direction/contraflow/off-road/speed/teleport/discontinuity/impossible round-trip/invalid-action 검사를 구현한다.
    - 한 건 이상의 hard violation은 episode failure와 원인 trace hash를 생성하고 intention-to-evaluate accounting에서 사라지지 않게 한다.
    - 완료 검증: 각 fault를 하나씩 주입하면 대응 category만 증가하고 valid directed trace는 0건이다.
    - _Requirements: 13.7, 13.9, 19.4_
    - _Correctness Property: 24_

  - [x] 7.4 hierarchical paired statistics를 구현한다
    - `pursuit_evasion_rl/research/statistics/paired.py`에 Wilson, paired risk difference/odds ratio, exact McNemar, restricted mean/paired effect, seed-outer 10,000 bootstrap, Holm/BH와 2x2 contrast를 구현한다.
    - missing/failed/interrupted case accounting, seed별 row·between-seed SD, two-sided raw/adjusted p, effect CI, n_pairs/n_seeds 및 practical threshold gate를 보존한다.
    - 완료 검증: fixed synthetic data를 독립 reference와 비교하고 null/known-effect simulation 방향을 확인한다.
    - _Requirements: 7.1–7.8, 8.2–8.7, 10.6–10.7_
    - _Correctness Properties: 8, 10–12, 16_

  - [x] 7.5 evaluation/metrics/statistics unit 및 validation test를 작성한다 (test_paired_evaluator.py/test_behavior_metrics.py/test_physical_plausibility.py/test_paired_statistics.py로 구현, 내용 동등)
    - `tests/research/test_paired_evaluation.py`, `test_metrics_behavior.py`, `test_physical_metrics.py`, `test_statistics_paired.py`에 hand reference, pair mismatch, 10,000-bootstrap reproducibility, Holm, missing-case accounting을 구현한다.
    - 완료 검증: 모든 결과가 finite SI 단위이고 episode마다 경찰 row가 정확히 6개이며 실패 case 총합이 계획 수와 일치한다.
    - _Requirements: 7.1–7.8, 8.1–8.8, 13.1–13.9, 19.4–19.5_
    - _Correctness Properties: 8–12, 21–24_

  - [x] 7.6 independent repetition property test를 작성한다
    - `tests/research/properties/test_property_08_independent_repetitions.py`에 sample plan/seed/checkpoint 결과 생성기를 구현해 최소 100 examples로 5/500, reduced 3/100, exploratory 경계와 seed variance 보존을 검증한다.
    - **Property 8: Independent-repetition accounting cannot be inflated by checkpoints**
    - 완료 검증: 같은 seed의 checkpoint를 복제해도 replicate 수가 증가하지 않는다.
    - **Validates: Requirements 7.1–7.8**

  - [x] 7.7 EpisodeCase pair-contract property test를 작성한다
    - `tests/research/properties/test_property_09_pair_contract.py`에 두 policy episode record 생성기를 구현해 최소 100 examples로 complete equality와 paired status의 동치를 검증한다.
    - **Property 9: Paired status is equivalent to complete case-contract equality**
    - 완료 검증: map/6 police/fugitive/RNG/termination 중 정확히 한 필드 perturbation을 모두 식별한다.
    - **Validates: Requirements 8.1, 8.8, 19.5**

  - [x] 7.8 confidence interval/domain property test를 작성한다
    - `tests/research/properties/test_property_10_statistical_domains.py`에 binary count와 nested paired numeric sample 생성기를 구현해 최소 100 examples로 bounded CI, seed-outer resampling 및 보고 필드를 검증한다.
    - **Property 10: Confidence intervals and paired tests preserve their registered domain**
    - 완료 검증: pair identity나 seed hierarchy를 깨뜨린 resample이 거부된다.
    - **Validates: Requirements 8.2–8.4**

  - [x] 7.9 multiplicity/practical-effect property test를 작성한다
    - `tests/research/properties/test_property_11_conservative_gates.py`에 p-value family와 effect CI 생성기를 구현해 최소 100 examples로 Holm 유효성·단조성과 practical threshold gate를 검증한다.
    - **Property 11: Multiple-comparison and practical-effect gates are conservative**
    - 완료 검증: 유의하지만 practical threshold를 못 넘는 비교는 superiority를 통과하지 못한다.
    - **Validates: Requirements 8.5–8.6**

  - [x] 7.10 planned-case conservation property test를 작성한다
    - `tests/research/properties/test_property_12_planned_case_accounting.py`에 valid/failure/exclusion 결과 생성기를 구현해 최소 100 examples로 exactly-once와 계획 수 보존을 검증한다.
    - **Property 12: Planned-case accounting is conserved under failures**
    - 완료 검증: 누락·중복·이유 없는 제외가 자동 실패하며 null result도 명시적 상태로 남는다.
    - **Validates: Requirements 8.7, 15.8, 19.4**

  - [x] 7.11 per-officer behavior property test를 작성한다
    - `tests/research/properties/test_property_21_officer_metrics.py`에 finite six-officer trace 생성기를 구현해 최소 100 examples로 모든 행동 metric을 단순 reference와 비교한다.
    - **Property 21: Per-officer behavioral metrics equal reference trace computations**
    - 완료 검증: 단위, finite 값, officer별 row와 leave-one-off 정의가 일치한다.
    - **Validates: Requirements 13.1–13.4, 19.4**

  - [x] 7.12 containment geometry property test를 작성한다
    - `tests/research/properties/test_property_22_containment_geometry.py`에 여섯 경찰 좌표·graph set 생성기를 구현해 최소 100 examples로 bounds, rotation/translation symmetry와 reachable reduction을 검증한다.
    - **Property 22: Containment metrics satisfy geometric bounds and symmetries**
    - 완료 검증: coverage/max-gap bounds와 before/after set 차이가 reference와 일치한다.
    - **Validates: Requirements 13.5, 19.4**

  - [x] 7.13 scenario aggregation property test를 작성한다
    - `tests/research/properties/test_property_23_scenario_aggregation.py`에 scenario episode batch 생성기를 구현해 최소 100 examples로 outcome conservation, six rows, team/worst aggregation을 검증한다.
    - **Property 23: Scenario metrics conserve outcomes and six-officer aggregation**
    - 완료 검증: Interior escape, officer row 5/7개, 방향 반대 worst 계산이 모두 거부된다.
    - **Validates: Requirements 13.6, 13.8, 19.4**

  - [x] 7.14 physical violation property test를 작성한다
    - `tests/research/properties/test_property_24_physical_violations.py`에 valid trace와 fault injection 생성기를 구현해 최소 100 examples로 8개 violation category와 linked failure case를 검증한다.
    - **Property 24: Physical violations are complete and fail visibly**
    - 완료 검증: hard violation이 양수인데 failure ledger가 없는 결과는 실패한다.
    - **Validates: Requirements 13.7, 13.9, 19.4**

- [x] 8. protocol·leakage·manifest·claim 및 paper artifact gate를 구현한다
  - [x] 8.1 ResearchProtocol freeze와 leakage guard를 구현한다
    - `pursuit_evasion_rl/research/protocol.py`에 RQ/가설/조건/split/sample/resource/metric/statistics/threshold/tolerance/exclusion/stop/search 필수 schema, freeze timestamp/hash와 post-freeze fork를 구현한다.
    - train/validation-only selection, held-out/cross-city handle 격리, analysis exactly-one 및 미계획/변경 분석의 exploratory 강등을 lineage event로 강제한다.
    - 완료 검증: 필수 field 누락, test dependency, target-city adaptation, freeze 후 in-place 변경이 confirmatory gate를 통과하지 못한다.
    - _Requirements: 3.1, 4.2–4.3, 6.1–6.7, 15.1–15.7_
    - _Correctness Properties: 4, 7, 29_

  - [x] 8.2 resource estimator와 condition execution ledger를 구현한다
    - `pursuit_evasion_rl/research/execution.py`와 `cli/execute.py`에 accelerator-hour/wall-clock/environment-step/LLM 비용 추정, protocol hash 확인, `completed|failed|not_run` exactly-one 및 reason/artifact hash 보존을 구현한다.
    - 결과를 보기 전 5 seeds/500 episodes 또는 전체 matrix 공통 3/100 reduced rule을 선택하고 3/100 미만은 exploratory로 강제하며 success-only view가 full population을 대체하지 못하게 한다.
    - 완료 검증: resource 부족, interruption, 실패, null metric이 누락 없이 ledger/manifest/claim input에 남는다.
    - _Requirements: 4.6–4.7, 7.1–7.8, 8.7, 15.8–15.9_
    - _Correctness Properties: 8, 12, 30, 31_

  - [x] 8.3 fail-closed PaperClaimGate를 구현한다
    - `pursuit_evasion_rl/research/claims/gate.py`에 evidence type, protocol/classification, split/leakage, citation, manifest, statistics와 source hash reconciliation을 구현한다.
    - 결함이 하나라도 있으면 confirmatory export를 막고 실패 record를 보존하며 `Field_Readiness_Claim`은 성능과 무관하게 항상 `unsupported`로 고정한다.
    - 완료 검증: 높은 capture 값에서도 field readiness가 변하지 않고 각 mandatory dependency 단독 결함이 claim export를 차단한다.
    - _Requirements: 4.1, 4.4–4.7, 17.1, 17.4, 19.7–19.10_
    - _Correctness Properties: 3, 4, 32, 34_

  - [x] 8.4 citation ledger와 related-work artifact 생성기/validator를 구현한다
    - `pursuit_evasion_rl/research/paper/citations.py`, `related_work.py`에 검색 protocol, screening exactly-one+reason, DOI/URL/source location/hash extraction, matrix columns/unknown 및 novelty comparison-axis graph를 구현한다.
    - 세 검색원 provenance와 기존 제안서 원문 대조 상태를 입력 schema로 요구하고 미검증 자료·우선권 표현을 novelty evidence에서 제외한다.
    - 완료 검증: missing/orphan body key, 미검증 citation, 결정 없는 screening, 직접 비교축 없는 novelty claim이 실패한다.
    - _Requirements: 3.1–3.9, 4.1, 4.4_
    - _Correctness Property: 3_

  - [x] 8.5 paper table/figure/manuscript artifact 생성기와 schema validator를 구현한다
    - `pursuit_evasion_rl/research/paper/tables.py`, `figures.py`, `bundle.py`에 claim-gated analysis만 입력받는 표·그림·논문 bundle 생성과 run/generator/config/evidence/citation/content-hash sidecar를 구현한다.
    - completed/failed/not_run/null 조건, seed row/aggregate/CI/test/effect/n, split/curve/metric/성공·실패 trajectory와 필수 논문 절을 schema로 검증하며 원시 결과를 성공으로 간주하지 않는다.
    - 완료 검증: 출처 없는 숫자, manifest 없는 그림, condition 누락, citation orphan, 필수 절 누락 bundle이 export되지 않는다.
    - _Requirements: 4.6, 14.9, 16.1–16.10, 19.7–19.9_
    - _Correctness Properties: 28, 31, 34_

  - [x] 8.6 현실성·윤리·future-work 범위 validator를 구현한다
    - `pursuit_evasion_rl/research/paper/scope.py`에 시뮬레이션 한계, 비대체 의사결정 지원, 위험·별도 현장 검증 필요성 및 연구/향후 범위 절 schema를 구현한다.
    - CCTV/ANPR/dashboard/drone은 구현하지 않고 오직 `future_work_demo` label과 primary aggregate/novelty/success dependency 차단만 검증한다.
    - 완료 검증: field-readiness 파생 문구 또는 future-work node를 핵심 claim에 연결하면 bundle/claim gate가 실패한다.
    - _Requirements: 17.1–17.6, 18.1–18.7_
    - _Correctness Properties: 32, 33_

  - [x] 8.7 protocol/claim/citation/paper integration test를 작성한다 (test_protocol_freeze.py/test_claim_gate.py/test_paper_bundle.py로 구현, 내용 동등)
    - `tests/research/test_protocol_claim_paper.py`에 freeze/fork, leakage, status 보존, citation bidirectionality, generated table/figure sidecar, 필수 절과 fail-closed export를 구현한다.
    - 완료 검증: completed/failed/not_run fixture를 모두 포함한 offline paper bundle과 실패 gate report가 deterministic hash를 가진다.
    - _Requirements: 3.1–4.7, 14.9, 15.1–15.9, 16.1–18.7, 19.7–19.10_
    - _Correctness Properties: 3, 4, 7, 28–34_

  - [x] 8.8 citation/novelty graph property test를 작성한다
    - `tests/research/properties/test_property_03_citation_graph.py`에 ledger/matrix/body/claim graph 생성기를 구현해 최소 100 examples로 screening, verified evidence, complete matrix와 citation closure를 검증한다.
    - **Property 3: Citation and novelty evidence forms a closed traceable graph**
    - 완료 검증: symmetric-difference와 orphan/unknown 미표시 field가 정확히 탐지된다.
    - **Validates: Requirements 3.2–3.6, 3.8–3.9, 4.1, 4.4**

  - [x] 8.9 claim scope property test를 작성한다
    - `tests/research/properties/test_property_04_claim_scope.py`에 RQ/protocol/evidence/status 생성기를 구현해 최소 100 examples로 falsifier 완전성, same-road 제한, priority 금지와 malformed status rejection을 검증한다.
    - **Property 4: Research claims remain within the demonstrated scope**
    - 완료 검증: seed-only evidence가 spatial generalization claim을 통과하지 못한다.
    - **Validates: Requirements 4.3–4.7, 6.9**

  - [x] 8.10 evaluation leakage property test를 작성한다
    - `tests/research/properties/test_property_07_evaluation_leakage.py`에 lineage/checkpoint/LLM/evaluation event 생성기를 구현해 최소 100 examples로 train/validation-only selection, frozen policy, 2-city zero-shot와 분리 aggregate를 검증한다.
    - **Property 7: Evaluation data cannot influence policy selection**
    - 완료 검증: test/target-city dependency 하나라도 있으면 confirmatory eligibility가 제거된다.
    - **Validates: Requirements 6.3–6.7, 9.7, 12.5–12.6, 15.6–15.7**

  - [x] 8.11 paper provenance property test를 작성한다
    - `tests/research/properties/test_property_28_paper_provenance.py`에 table/figure/manuscript item 생성기를 구현해 최소 100 examples로 run/generator/config/evidence/artifact hash와 citation 양방향 해석을 검증한다.
    - **Property 28: Every paper number and graphic has immutable run provenance**
    - 완료 검증: sidecar 또는 source hash 하나가 없는 숫자·그림은 export되지 않는다.
    - **Validates: Requirements 14.9, 16.4, 16.10, 19.7–19.8**

  - [x] 8.12 protocol timeline property test를 작성한다
    - `tests/research/properties/test_property_29_protocol_timeline.py`에 draft/freeze/analysis event 생성기를 구현해 최소 100 examples로 필수 field, immutable freeze, exactly-one classification과 exploratory 강등을 검증한다.
    - **Property 29: Protocol and analysis classification are immutable and timeline-consistent**
    - 완료 검증: freeze 이후 변경 분석이 confirmatory로 남는 경우 실패한다.
    - **Validates: Requirements 15.2–15.5**

  - [x] 8.13 failed/full-population property test를 작성한다
    - `tests/research/properties/test_property_30_failed_population.py`에 preregistered seeds/status/exclusion 생성기를 구현해 최소 100 examples로 failure provenance, full-seed aggregate와 success-only label을 검증한다.
    - **Property 30: Failed and success-only views cannot alter the preregistered population**
    - 완료 검증: failed seed 삭제 또는 success-only view 대체가 거부된다.
    - **Validates: Requirements 15.8–15.9**

  - [x] 8.14 paper condition coverage property test를 작성한다
    - `tests/research/properties/test_property_31_paper_condition_coverage.py`에 frozen condition/status/table 생성기를 구현해 최소 100 examples로 set equality와 통계 필드 완전성을 검증한다.
    - **Property 31: Paper condition coverage is complete**
    - 완료 검증: failed/not_run/null 조건도 빠지지 않으며 누락·중복 row가 실패한다.
    - **Validates: Requirements 4.6, 16.4, 16.6**

  - [x] 8.15 field-readiness property test를 작성한다
    - `tests/research/properties/test_property_32_field_readiness.py`에 임의 성능 결과와 claim transition 생성기를 구현해 최소 100 examples로 `unsupported` 불변성과 현실 효과 파생 금지를 검증한다.
    - **Property 32: Field readiness is invariantly unsupported**
    - 완료 검증: capture rate 1.0을 포함한 어떤 입력도 현장 안전/범죄 감소/실제 검거 claim을 만들지 못한다.
    - **Validates: Requirements 17.1, 17.4, 19.10**

  - [x] 8.16 future-work isolation property test를 작성한다
    - `tests/research/properties/test_property_33_future_work_isolation.py`에 claim dependency graph 생성기를 구현해 최소 100 examples로 CCTV/ANPR/dashboard/drone 노드 격리와 demo label을 검증한다.
    - **Property 33: Future-work artifacts cannot satisfy core research claims**
    - 완료 검증: future-work artifact가 novelty/success/primary aggregate에 기여하면 실패한다.
    - **Validates: Requirements 18.5–18.6**

  - [x] 8.17 claim gate property test를 작성한다
    - `tests/research/properties/test_property_34_claim_gate.py`에 완전 claim graph와 mandatory defect injection 생성기를 구현해 최소 100 examples로 fail-closed export와 failure record 보존을 검증한다.
    - **Property 34: The paper claim gate is fail-closed**
    - 완료 검증: classification/split/leakage/provenance/hash/evidence/citation/analysis/manifest/statistics 결함을 각각 주입해 모두 차단한다.
    - **Validates: Requirements 19.1, 19.3, 19.7–19.10**

- [ ] 9. 제한된 offline LLM adapter와 frozen replay를 구현한다 (사용자 결정으로 이번 실행 범위에서 명시적으로 제외 — "9장은 별로 안중요하면 일단은 진행하지 않고")
  - [~] 9.1 allowlisted offline LLM adapter와 frozen-output replay를 구현한다
    - `pursuit_evasion_rl/research/llm/offline.py`에 사전등록된 critique/role/curriculum 중 하나만 허용하는 schema, low-level control 거부, PII/사건/비공개 위치·산출물/test leakage preflight를 구현한다.
    - 외부 호출 없는 frozen response replay를 기본으로 하고 optional external adapter는 모델/version/prompt/sampling/input-output hash/UTC/cost/latency/failure provenance를 별도 기록하며 episode loop에서 import할 수 없게 한다.
    - 완료 검증: network-disabled replay가 동일 hash를 생성하고 금지 payload는 adapter 호출 전에 차단되며 no-LLM과 budget-matched condition을 만든다.
    - _Requirements: 12.1–12.10, 19.11–19.12_
    - _Correctness Properties: 7, 18_

  - [ ]* 9.2 offline LLM unit/integration 및 architecture test를 작성한다
    - `tests/research/test_offline_llm.py`에 frozen replay, schema/provenance, leakage/sensitive/low-level denial, budget matching과 episode-loop import 금지 검사를 구현한다.
    - optional external test는 기본 suite에서 skip하고 명시적 opt-in 시에도 synthetic public payload만 전송한다.
    - 완료 검증: 기본 test는 외부 API key와 네트워크 없이 통과하고 실패 사례·비용·latency null이 보존된다.
    - _Requirements: 12.1–12.10, 19.11–19.12_
    - _Correctness Property: 18_

  - [ ]* 9.3 offline LLM safety property test를 작성한다
    - `tests/research/properties/test_property_18_offline_llm.py`에 task/payload/provenance/budget 생성기를 구현해 최소 100 examples로 allowlist, pre-invocation 차단, frozen test-free output과 비교 공정성을 검증한다.
    - **Property 18: Offline LLM use is allowlisted, frozen, nonleaking, and budget-matched**
    - 완료 검증: low-level/sensitive/test payload는 invocation count 0이고 허용 호출은 모든 provenance/report field를 가진다.
    - **Validates: Requirements 12.1, 12.3–12.9**

- [x] 10. 외부 서비스 없는 자동 검증 gate를 구현하고 통과시킨다
  - [x] 10.1 offline quality-gate runner를 구현한다
    - `pursuit_evasion_rl/research/quality.py`, `cli/quality.py`, `pyproject.toml` test marker에 unit/regression/PBT/integration/statistical/smoke 단계와 machine-readable attestation hash를 구현한다.
    - 34개 Property ID가 각각 정확히 한 executable Hypothesis test에 연결되고 최소 100 examples로 실행되었는지 검사하며 외부 OSM/LLM socket 접근을 기본 차단한다. (Property 18/LLM은 사용자 결정으로 REQUIRED_PROPERTY_IDS에서 DEFERRED로 명시적 제외.)
    - 완료 검증: 누락 property, skipped mandatory offline category, failed test 또는 stale attestation 중 하나라도 pilot admission을 차단한다.
    - _Requirements: 19.1–19.11_
    - _Correctness Properties: 1–34_

  - [x] 10.2 offline unit/regression/integration/statistical gate를 실행하고 결함을 수정한다
    - 신규 `tests/research/`와 기존 OSM 환경/관측/runner/coarsening 회귀 test를 one-shot pytest 명령으로 실행하고 결과/JUnit hash를 gate attestation에 기록한다.
    - temp filesystem best_v2, map cache, protocol freeze, checkpoint IO, paper generator 및 fixed statistical reference를 포함한다 (frozen LLM replay는 9장 제외에 따라 범위 밖).
    - 완료 검증: 모든 해당 test가 pass이며 실패/skip이 있으면 성공으로 간주하지 않고 원인과 `failed` 상태를 보존한다. cli/quality.py 실제 실행 결과: 814 passed, 0 failed, 1 documented skip.
    - _Requirements: 2.1–2.2, 3.1, 3.7, 4.2, 5.10, 6.1, 9.2, 11.3, 12.2, 12.10, 14.3, 15.1, 16.1–18.7, 19.1–19.12_
    - _Correctness Properties: 1–34_

  - [x] 10.3 34개 PBT gate를 실행하고 모두 통과시킨다
    - `tests/research/properties/`를 최소 100 examples/property로 실행하고 Property 1–34 각각의 pass/fail 및 Hypothesis counterexample을 machine-readable artifact로 보존한다.
    - 동일 기반 모듈 property 파일은 dependency graph의 병렬 wave에서 실행하되 random seed와 profile을 manifest에 고정한다.
    - 완료 검증: Property ID set이 정확히 `{1..34}\{18}`이고(18은 사용자 결정으로 deferred) 모든 결과가 pass여야 하며 failed/not_run property는 pilot을 차단한다. 33/33 통과.
    - _Requirements: 19.1–19.11_
    - _Correctness Properties: 1–34_

  - [x] 10.4 완전 오프라인 smoke pipeline test를 작성하고 실행한다
    - `tests/research/test_offline_smoke_pipeline.py`에 audit → best_v2 temp preservation → fixture/cache map register/split → tiny train/save/resume → paired eval → six-officer metrics → statistics → claim rejection → paper fixture bundle을 자동화한다.
    - 실제 best_v2에는 쓰지 않고, 외부 OSM/LLM 호출을 차단하며 성공뿐 아니라 injected failed/not_run/null 결과가 최종 bundle에 보존되는지 검증한다.
    - 완료 검증: smoke attestation이 unit/PBT/integration attestation hash를 참조하고 모든 offline gate가 green일 때만 experiment admission token을 생성한다. test_smoke_pipeline.py로 구현, 통과.
    - _Requirements: 2.1–2.6, 5.1–6.9, 12.10, 14.1–16.10, 19.7–19.11_
    - _Correctness Properties: 2, 5–12, 18–34_

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise. (814 passed, 1 documented skip, 0 failed — 매 이후 변경마다 재확인됨)

- [x] 12. gate 순서에 따라 pilot부터 paper bundle까지 연구 실행을 자동화한다 (LLM arm 제외, 나머지 전체 실제 실행 완료)
  - [x] 12.1 protocol-blind pilot 실행기를 구현하고 pilot을 실행한다
    - `pursuit_evasion_rl/research/experiments/pilot.py`와 CLI에 offline admission token, best_v2 재검증, pilot 전 resource estimate, frozen protocol hash와 train/validation-only 작은 sanity matrix를 요구한다.
    - pilot 결과로 confirmatory threshold를 조정하지 않으며 실행별 `completed|failed|not_run`, null metric, error artifact와 resource actual을 seal한다.
    - 완료 검증: gate/resource/protocol 중 하나라도 없으면 launch 0회이며, 실행되더라도 성공을 사전 가정하지 않고 sealed pilot manifest만 생성한다. 실제 CLI 실행 완료: run 20260731T193702-27ed9c3a2322 completed, best_v2.pt 무결 확인.
    - _Requirements: 2.3–2.6, 7.1–7.5, 14.1–15.9, 19.9–19.11_
    - _Correctness Properties: 2, 7, 8, 12, 25, 29–30, 34_

  - [x] 12.2 full/reduced multi-seed training과 paired evaluation orchestrator를 구현하고 실행한다
    - `pursuit_evasion_rl/research/experiments/main_study.py`에 실행 직전 resource estimate+protocol freeze gate, 조건별 기본 5 independent seeds/500 paired episodes, ceiling 시 전체 조건 3/100 reduced 자동 전환을 구현한다.
    - train map만 gradient에, validation만 selection에 사용하고 Interior/Boundary 및 in-region/held-out을 분리하며 best_v2를 읽기 전용 baseline/명시적 initialization으로만 사용한다.
    - 완료 검증: 모든 사전등록 condition이 `completed|failed|not_run` 중 하나로 남고 3/100 미만은 exploratory이며 성공 seed만 고른 aggregate가 primary 결과를 대체하지 않는다. 실제 CLI 실행: 2 seed 완주, 4 paired episode, ledger_conserved=true, best_v2.pt 무결.
    - _Requirements: 2.3–2.6, 5.8–5.10, 6.3–6.4, 7.1–8.8, 11.1–11.6, 14.1–15.9_
    - _Correctness Properties: 2, 7–12, 17, 25–27, 29–31, 34_

  - [x] 12.3 observation/reward/placement/stabilization ablation을 gate 후 실행한다
    - `pursuit_evasion_rl/research/experiments/ablations.py`에 실행마다 새 resource estimate와 동일 frozen protocol hash를 확인하고 21D/28D, reward leave-one-out/retreat, placement 3-arm, stabilization 2x2를 paired budget으로 실행한다.
    - capture/containment/anti-oscillation/idleness/physical/cost와 capacity comparison을 함께 생성하고 열세·null·trade-off·failed/not_run 결과를 보존한다.
    - 완료 검증: matrix completeness/one-factor/property attestation과 sample rule이 통과한 결과만 confirmatory analysis로 전달된다. 실제 CLI 실행: 관측/보상/배치/stabilization(uturn_off) 4개 arm completed, hysteresis 2개 arm은 이 RL 아키텍처(로컬 1-hop 선택)와의 구조적 비호환 사유를 기록하며 not_run (사용자 확인 결정), ledger_conserved=true.
    - _Requirements: 7.1–10.7, 11.2, 13.1–13.9, 15.1–15.9_
    - _Correctness Properties: 8–17, 21–24, 29–31, 34_

  - [x] 12.4 최소 2개 도시 cross-city zero-shot을 gate 후 실행한다 (offline LLM arm은 9장 제외에 따라 numerical baseline만)
    - `pursuit_evasion_rl/research/experiments/cross_city.py`에 실행 전 resource estimate+frozen protocol gate, 두 target-city snapshot hash, frozen selected policy와 adaptation 금지를 강제한다. ResearchTrainer를 아예 import하지 않는 구조로 "적응 금지"를 구조적으로 보장.
    - numerical baseline과 동일 budget/EpisodeCase로 비교한다 (frozen LLM replay는 9장 제외로 범위 밖).
    - 완료 검증: target-city가 training/selection input과 동일하면 generalization evidence를 폐기하고(TARGET_CITY_LEAKAGE), 부족 cache/resource는 `not_run`으로 기록한다. 실제 CLI 실행: busan/seoul(합성 target-city fixture, 실제 OSM 아님 명시) 모두 completed, ledger_conserved=true, best_v2.pt 무결.
    - _Requirements: 6.5–6.8, 7.1–8.8, 11.1–12.10, 15.6–15.9, 19.12_
    - _Correctness Properties: 7–12, 17–18, 25, 29–31, 34_

  - [x] 12.5 claim-gated 최종 analysis와 paper bundle을 생성한다
    - `pursuit_evasion_rl/research/experiments/paper_release.py`에 모든 run/condition status, paired hierarchical statistics, citation/related-work, limitation/ethics/scope를 reconciliation한 뒤 claim gate를 통과한 항목만 paper bundle로 내보내게 한다.
    - `Field_Readiness_Claim=unsupported`를 고정하고 CCTV/ANPR/dashboard/drone은 구현하지 않은 future work로만 검증하며 failed/not_run/null과 부정 결과를 표·한계에 포함한다.
    - 완료 검증: source hash·manifest·citation·condition 하나라도 누락되면 release가 실패하고 gate report는 보존되며, 통과 시 bundle/sidecar/content hash가 재생 가능하다. 실제 CLI 실행: eligible=true, gate_eligible=true, export_eligible=true, table/figure hash 재생 검증 통과, best_v2.pt 무결.
    - _Requirements: 3.1–4.7, 14.9, 16.1–18.7, 19.7–19.10_
    - _Correctness Properties: 3–4, 28, 31–34_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise. (814 passed, 1 documented skip, 0 failed; Task 12.1–12.5 실제 CLI 실행 모두 성공, best_v2.pt 매 단계 무결 확인)

## Notes

- `*`가 붙은 sub-task는 test 구현/실행 task를 표시한다. 다만 본 연구의 confirmatory 실행에서는 Task 10.1 gate가 unit/PBT/integration/smoke 전체 통과를 강제하므로 이를 생략하면 Task 12 실행은 시작되지 않고 `not_run`으로 남는다.
- 34개 Correctness Property는 각각 독립 Hypothesis test leaf와 고유 파일에 정확히 한 번 연결되며 최소 100 examples를 실행한다.
- 외부 OSM/LLM integration은 optional이며 기본 test와 핵심 frozen-output 실험은 완전 offline이다.
- 어떤 task도 기존 `checkpoints/osm_mappo/best_v2.pt`를 수정하거나 덮어쓰지 않는다. hash 불일치 또는 보호 경로 overlap은 모든 신규 실행을 차단한다.
- 코드/test gate 통과는 성능 증거가 아니다. 결과 상태와 수치는 실제 실행 후에만 생성하며 실패, 미실행, null, 열세 및 부정 결과를 보존한다.
- 현장 준비성은 항상 `unsupported`이다. CCTV, ANPR, dashboard, drone 구현은 범위 밖이며 future-work scope validator만 제공한다.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.1"] },
    { "id": 2, "tasks": ["1.4", "1.5", "2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6"] },
    { "id": 4, "tasks": ["3.1", "3.2"] },
    { "id": 5, "tasks": ["3.3"] },
    { "id": 6, "tasks": ["3.4", "4.1", "4.2"] },
    { "id": 7, "tasks": ["4.3", "4.5", "4.6"] },
    { "id": 8, "tasks": ["4.4", "4.7", "5.1", "5.2", "5.3", "5.4"] },
    { "id": 9, "tasks": ["5.5", "5.6", "5.7", "5.8"] },
    { "id": 10, "tasks": ["5.9", "5.10", "5.11", "6.1"] },
    { "id": 11, "tasks": ["6.2", "6.5"] },
    { "id": 12, "tasks": ["6.3", "6.6"] },
    { "id": 13, "tasks": ["6.4", "6.7"] },
    { "id": 14, "tasks": ["7.1", "7.2", "7.3", "7.4"] },
    { "id": 15, "tasks": ["7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "7.11", "7.12", "7.13", "7.14"] },
    { "id": 16, "tasks": ["8.1", "8.2", "8.4", "8.6"] },
    { "id": 17, "tasks": ["8.3", "8.8", "8.9", "8.10", "8.12", "8.13", "8.15", "8.16"] },
    { "id": 18, "tasks": ["8.5", "8.17", "9.1"] },
    { "id": 19, "tasks": ["8.7", "8.11", "8.14", "9.2", "9.3"] },
    { "id": 20, "tasks": ["10.1"] },
    { "id": 21, "tasks": ["10.2", "10.3"] },
    { "id": 22, "tasks": ["10.4"] },
    { "id": 23, "tasks": ["12.1"] },
    { "id": 24, "tasks": ["12.2"] },
    { "id": 25, "tasks": ["12.3"] },
    { "id": 26, "tasks": ["12.4"] },
    { "id": 27, "tasks": ["12.5"] }
  ]
}
```
