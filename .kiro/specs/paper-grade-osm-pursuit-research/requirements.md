# Requirements Document

## Introduction

`paper-grade-osm-pursuit-research`는 기존 OSM 도로 추격 구현을 논문 심사에 견딜 수 있는 재현 가능한 연구 체계로 확장한다. 연구 대상은 실제 OSM provenance를 가진 도로 그래프에서 여섯 경찰 에이전트가 제한 시야 도주자를 협력 추격하는 정책의 공간·도시 일반화, 포위 행동, 안정성 및 물리적 타당성이다. LLM은 실시간 제어기가 아니라 사전등록된 오프라인 연구 보조 조건으로만 평가한다.

저장소에는 21차원 `osm_topology_v1`, 코드상 7개 값을 추가하는 28차원 관측 경로, 여섯 경찰의 파라미터 공유 학습 경로, 합법 행동 마스킹, 교차로 의사결정 중심 전이, 배치 로직, 휴리스틱 정책 및 체크포인트 재개 코드가 존재한다. 이 문장은 구현 존재를 기술할 뿐 시험 통과나 성능을 주장하지 않는다. `training.log`에 보고된 최근 100개 학습 에피소드 이동 창 최고 검거율 0.99는 동일 학습 분포의 선택 편향 가능성이 있는 예비 결과이며 verified 결과가 아니다. `checkpoints/osm_mappo/best_v2.pt`의 존재도 확인 대상일 뿐 성능·무결성·계보가 verified라는 뜻이 아니다.

Research_System은 구현 사실, 시험 상태, Prior_Result, 신규 실험 관측 및 Paper_Claim을 서로 대체할 수 없는 증거로 관리한다. Synthetic_Fixture 결과를 Actual_OSM_Map 결과로 표현하지 않으며, 본 요구사항을 충족하더라도 Field_Readiness는 지원하지 않는다.

## Glossary

- **Research_System**: 본 사양에 따른 감사, 문헌 조사, 프로토콜, 실험, 통계, 재현성 및 논문 산출물 전체.
- **Existing_Implementation**: 현재 저장소의 환경, 관측, 정책, 학습·평가 코드, 메트릭, 체크포인트 및 로그.
- **Implementation_Inventory**: 자산별 경로, Code_Revision 및 Test_Status를 기록하는 목록.
- **Test_Status**: `passed`, `failed`, `not_tested` 중 정확히 하나인 자동 시험 상태.
- **Prior_Result**: Research_Protocol 고정 전에 생성된 로그, 평가, 그림 또는 보고서 결과.
- **Prior_Result_Status**: `verified`, `preliminary`, `invalid` 중 정확히 하나인 과거 결과 분류.
- **Evidence_Record**: Evidence_Type, 생성 주체·UTC 시각·방법, 원천 위치, 추출값, 검증 상태, 제한 및 Content_Hash를 포함하는 provenance 기록.
- **Evidence_Type**: `implementation_existence`, `test_result`, `experiment_observation`, `paper_claim` 중 정확히 하나인 증거 유형.
- **Claim_Register**: 구현 사실, 예비·검증·부정·실패·미실행 결과와 미지원 주장을 근거에 연결하는 장부.
- **Research_Protocol**: 결과 접근 전에 연구 질문, 분석 등급, 결과 변수, 분할, 조건, 표본, 통계, 임계값, 제외·실패 규칙 및 허용오차를 고정한 문서.
- **Analysis_Classification**: `confirmatory` 또는 `exploratory` 중 정확히 하나인 분석 분류.
- **Execution_Status**: `completed`, `failed`, `not_run` 중 정확히 하나인 Condition 실행 상태.
- **Related_Work_Matrix**: 관련 연구별 문제, 지도, 에이전트, 정보, 알고리즘, 기준선, 분할, 표본, 통계, 한계 및 차이를 정리한 표.
- **Citation_Ledger**: 검색 provenance, 중복 연결, screening, 서지 대조, 원문 위치·hash 및 사용 주장을 연결하는 장부.
- **Novelty_Claim**: Related_Work_Matrix와 검증된 Citation_Ledger가 뒷받침하는 제한된 차별성 진술.
- **Model_Network**: 방향 도로, 교차점, 경계 표식, 좌표계 및 생성 계보를 포함하는 추격 네트워크.
- **Actual_OSM_Map**: OSM 출처, 조회 영역, 취득 UTC, 전처리 버전, 좌표계 및 Content_Hash가 있는 실제 도로 자료 기반 Model_Network.
- **Synthetic_Fixture**: 시험 또는 통제 실험을 위해 인공 생성한 그래프이며 Actual_OSM_Map이 아닌 자료.
- **Interior_Contained_Map**: 경계 탈출 표식이 없고 허용 outcome이 `capture` 또는 `timeout`인 Model_Network.
- **Boundary_Escape_Map**: 경계 통과 세그먼트를 유지하고 허용 outcome이 `capture`, `escape`, `timeout`인 Model_Network.
- **Spatially_Disjoint_Split**: train, validation, test 다각형·metric buffer·원천 OSM edge가 분리된 분할.
- **Held_Out_OSM_Evaluation**: 선택과 조정에 사용되지 않은 Actual_OSM_Map test 영역 평가.
- **Cross_City_Zero_Shot_Evaluation**: 대상 도시 자료를 학습·조정·선택에 사용하지 않고 고정 정책을 적용하는 평가.
- **Training_Seed**: 가중치 초기화, 샘플링, 환경 및 학습 난수열을 독립적으로 정하는 정수.
- **Checkpoint_Time**: 동일 Training_Seed 실행 안의 저장 시점이며 독립 반복이 아니다.
- **Evaluation_Episode**: 고정 Condition과 Episode_Case로 실행하는 한 번의 추격.
- **Episode_Case**: 지도, 시나리오, 여섯 경찰·도주자 초기 상태, 도주자 RNG 및 종료 계약을 봉인한 쌍체 평가 단위.
- **Condition**: 정책, 관측, 보상, 배치, 안정화, 지도 시나리오 및 도주자 설정의 고정 조합.
- **Paired_Evaluation**: 동일 Episode_Case를 비교 정책에 재생하는 평가.
- **Practical_Threshold**: 통계적 유의성과 별도로 효과의 실질적 크기를 판정하도록 Research_Protocol에 고정한 임계값.
- **Reduced_Power_Protocol**: Resource_Ceiling으로 기본 표본을 완료할 수 없을 때 결과 접근 전에 적용하는 축소 규칙.
- **Resource_Ceiling**: 환경 스텝, 장치 시간, 벽시계 시간 및 LLM 비용의 사전등록 상한.
- **Content_Hash**: 파일 바이트 또는 정규화 자료의 SHA-256 값.
- **Code_Revision**: 커밋 식별자와 작업 트리 상태를 결합한 코드 식별값.
- **Best_V2_Artifact**: `checkpoints/osm_mappo/best_v2.pt` 원본과 실행 전 검증된 읽기 전용 보존 사본.
- **Observation_21D**: 기존 `osm_topology_v1` 21차원 관측 계약.
- **Observation_28D**: Observation_21D 뒤에 현재 코드가 계산하는 7개 값을 추가한 관측 계약.
- **Reward_Component_Set**: 코드 감사로 식별하고 Research_Protocol에서 계수와 수식을 고정하는 전체 보상 구성요소 집합.
- **Placement_Curriculum**: global, ring 또는 두 분포를 혼합한 배치 조건.
- **U_Turn_Suppression**: 직전 물리 세그먼트의 역방향 선택에 적용하는 조건.
- **Hysteresis**: 목표·역할·행동 전환에 유지 상태를 적용하는 조건.
- **Numerical_Baseline**: LLM 출력을 사용하지 않는 학습 또는 휴리스틱 비교 정책.
- **LLM_Feedback_Arm**: 저장된 train·validation 산출물에만 사전등록된 오프라인 작업 하나를 수행하는 조건.
- **Anti_Oscillation_Metric**: U-turn, 재방문 및 짧은 주기 전환을 측정하는 지표 묶음.
- **Containment_Metric**: 각도 커버리지, 최대 각도 간극, 출구 차단 및 도달 영역 감소를 측정하는 지표 묶음.
- **Capture_Metric**: 시나리오별 검거, 탈출, timeout, 검거 시간 및 censoring을 측정하는 지표 묶음.
- **Physical_Plausibility_Metric**: 여덟 개의 사전 정의된 물리 위반 범주를 측정하는 지표 묶음.
- **Immutable_Run_Manifest**: draft에서 sealed로 전환되고 sealed 뒤에는 fork로만 변경되는 실행 기록.
- **Resume_State**: 모델·학습기·전체 RNG·환경·pending SMDP·buffer 상태를 포함하는 재개 자료.
- **Paper_Artifact_Set**: 논문 본문, 표, 그림, 프로토콜, 인용, 한계, 윤리 및 재현성 자료 묶음.
- **Paper_Claim_Gate**: 주장별 필수 근거와 품질 게이트를 실패 폐쇄 방식으로 판정하는 출판 전 게이트.
- **Field_Readiness_Claim**: 시뮬레이션 결과를 실제 현장 안전성·효과성·운용 가능성으로 확대하는 주장.
- **Future_Work_System**: CCTV, ANPR, 대시보드, 드론 또는 이기종 에이전트 연동처럼 핵심 실험 밖의 기능.

## 현재 증거 경계

- Existing_Implementation의 구체 기능은 Implementation_Inventory와 자동 시험 결과가 생성되기 전까지 `implementation_existence` 후보이며 `test_result` 또는 `experiment_observation`이 아니다.
- 코드상 Observation_28D의 추가 값 순서와 정규화는 감사 가능한 구현 사실이지만 성능 기여는 검증 전 가설이다.
- 최근 100개 학습 에피소드 이동 창 최고 검거율 0.99는 독립 Training_Seed, 공간 분리, Confidence_Interval 및 도시 간 평가가 없는 `preliminary` 강제 대상이다.
- Best_V2_Artifact는 사전 실행 무결성 검사를 통과하기 전까지 성능 근거 또는 verified 자산으로 승격하지 않는다.
- Actual_OSM_Map이라는 표시는 OSM provenance 검사가 통과한 자료에만 사용하며 Synthetic_Fixture와 캐시 시험 자료에는 사용하지 않는다.

## Requirements

### Requirement 1: 기존 자산과 Prior_Result 감사

**User Story:** 연구 책임자로서, 기존 구현과 과거 결과의 유효 범위를 알고 싶다. 그래야 검증되지 않은 결과를 논문 근거로 상속하지 않는다.

#### Acceptance Criteria

1. WHEN 연구 감사를 시작하면, THE Research_System SHALL 환경, 관측, 정책, 학습, 평가, 메트릭 및 체크포인트 자산마다 저장소 상대 경로, Code_Revision 및 Test_Status를 Implementation_Inventory에 기록한다.
2. IF 자산의 시험을 실행하지 않았거나 시험 결과를 찾을 수 없으면, THEN THE Research_System SHALL Test_Status를 `not_tested`로 기록하고 원인을 기록한다.
3. WHEN Prior_Result를 등록하면, THE Research_System SHALL 생성 코드·Code_Revision, data kind, 지도 Content_Hash, Condition, Training_Seed, Checkpoint_Time, 에피소드 수, 생성 UTC·방법 및 알려진 결함을 Evidence_Record에 기록한다.
4. IF Prior_Result의 필수 provenance를 확인할 수 없으면, THEN THE Research_System SHALL 확인 불가 필드를 `unknown`으로 기록하고 필드별 unknown 사유를 기록한다.
5. WHEN Prior_Result를 분류하면, THE Claim_Register SHALL Prior_Result_Status 세 값 중 정확히 하나를 기록한다.
6. IF Prior_Result가 수정 전 탈출 판정 결함 또는 결과를 바꾸는 알려진 구현 결함의 영향을 받으면, THEN THE Claim_Register SHALL 다른 조건보다 우선하여 Prior_Result를 `invalid`로 분류한다.
7. IF Prior_Result가 invalid가 아니면서 독립 Training_Seed가 3개 미만이거나 Condition당 Evaluation_Episode가 100개 미만이거나 95% Confidence_Interval이 없거나 Spatially_Disjoint_Split 평가가 없으면, THEN THE Claim_Register SHALL Prior_Result를 `preliminary`로 분류한다.
8. WHEN 최근 100개 학습 에피소드 이동 창 최고 검거율 0.99를 등록하면, THE Claim_Register SHALL 해당 값을 `preliminary`로 고정하고 동일 학습 분포 이동 창이라는 제한을 연결한다.
9. IF Prior_Result가 invalid·preliminary 조건에 해당하지 않고 모든 필수 Evidence_Record가 완전성 검사를 통과하면, THEN THE Claim_Register SHALL Prior_Result를 `verified`로 분류한다.
10. WHEN Evidence_Record를 등록하면, THE Research_System SHALL Evidence_Type 네 값 중 정확히 하나를 기록한다.
11. IF Evidence_Type 또는 Prior_Result_Status가 누락·중복·provenance 불일치이면, THEN THE Research_System SHALL 감사 완전성 검사를 실패 처리하고 해당 기록을 Paper_Claim의 근거에서 제외한다.
12. THE Research_System SHALL implementation_existence, test_result, experiment_observation 및 paper_claim Evidence_Record가 서로를 대체하지 못하게 한다.

### Requirement 2: Best_V2_Artifact 불변 보존

**User Story:** 실험 담당자로서, 신규 실행이 기존 체크포인트를 변경하지 않게 하고 싶다. 그래야 기준 자산과 결과 계보를 복원할 수 있다.

#### Acceptance Criteria

1. WHEN 첫 신규 학습 또는 평가를 승인하기 전에, THE Research_System SHALL Best_V2_Artifact 원본의 실제 경로, SHA-256 Content_Hash, 바이트 크기 및 측정 UTC를 기록한다.
2. WHEN Best_V2_Artifact를 보존하면, THE Research_System SHALL 원본 경로와 분리된 경로에 바이트 동일 사본을 생성하고 사본을 읽기 전용으로 설정한다.
3. WHEN 신규 실행의 경로를 검증하면, THE Research_System SHALL 심볼릭 링크와 정규화된 실제 경로를 해석한 뒤 입력·출력·임시 경로와 원본·보존본의 중첩이 0개인지 검사한다.
4. WHILE 신규 실행이 진행될 때, THE Research_System SHALL Best_V2_Artifact 원본과 보존본에 대한 write, move 및 delete 작업을 차단한다.
5. WHEN 신규 실행을 시작하기 직전에, THE Research_System SHALL 원본과 보존본 각각의 Content_Hash와 바이트 크기를 등록값과 대조한다.
6. IF Content_Hash, 바이트 크기, 읽기 전용 상태 또는 실제 경로 분리 검사 중 하나가 실패하면, THEN THE Research_System SHALL 체크포인트·로그·결과 산출물을 생성하기 전에 실행을 차단하고 무결성 오류를 기록한다.
7. WHEN Best_V2_Artifact를 기준선 또는 초기 가중치로 사용하면, THE Immutable_Run_Manifest SHALL 원본 Content_Hash, 보존본 Content_Hash 및 사용 방식을 기록한다.

### Requirement 3: 체계적 관련 연구와 인용 추적성

**User Story:** 논문 저자로서, 관련 연구와 차별성의 근거를 원문까지 추적하고 싶다. 그래야 인용과 Novelty_Claim을 재검증할 수 있다.

#### Acceptance Criteria

1. WHEN 첫 confirmatory 결과에 접근하기 전에, THE Research_System SHALL 검색일 UTC, 검색식, 검색원, 기간, 언어, 포함 기준 및 제외 기준을 Research_Protocol에 고정한다.
2. WHEN 관련 연구 검색을 수행하면, THE Research_System SHALL 상호 독립적인 검색원 3개 이상을 사용하고 그중 학술 색인을 2개 이상 사용한다.
3. WHEN 검색 결과를 수집하면, THE Citation_Ledger SHALL 검색원, 검색식, 실행 UTC, 결과 순위, 원 레코드 식별자 및 검색 결과 Content_Hash를 모든 후보에 기록한다.
4. WHEN 동일 문헌 후보를 둘 이상 발견하면, THE Citation_Ledger SHALL 후보 레코드를 하나의 canonical 문헌 식별자에 연결하고 중복 관계를 보존한다.
5. WHEN 후보 문헌을 screening하면, THE Citation_Ledger SHALL `included` 또는 `excluded` 중 정확히 하나와 screening reason을 기록한다.
6. IF screening 분류나 reason이 누락·중복·불일치이면, THEN THE Research_System SHALL 해당 후보를 Novelty_Claim 근거에서 제외한다.
7. WHEN 문헌 사실을 검증하면, THE Research_System SHALL DOI 또는 출판사 영구 식별자의 공식 metadata를 저자·연도·제목과 대조하고 원문 페이지·절·표 위치 및 원문 Content_Hash와 연결한다.
8. IF Related_Work_Matrix의 필드를 원문에서 보고하지 않으면, THEN THE Research_System SHALL 해당 필드를 `not_reported`로 기록한다.
9. IF 공식 metadata 대조, 원문 위치 또는 원문 Content_Hash 검증이 완료되지 않으면, THEN THE Citation_Ledger SHALL 해당 항목을 `unverified`로 표시하고 Novelty_Claim 근거에서 제외한다.
10. WHEN 기존 제안서 참고문헌을 사용하면, THE Research_System SHALL 공식 metadata와 원문 대조를 통과한 항목만 verified 인용으로 승격한다.
11. WHEN Paper_Artifact_Set을 검증하면, THE Research_System SHALL 본문 인용, 참고문헌 및 Citation_Ledger 사이의 누락·고아 항목 수가 각각 0인지 검사한다.
12. IF 본문·참고문헌·Citation_Ledger 고아 검사가 0이 아니면, THEN THE Paper_Claim_Gate SHALL 관련 Paper_Artifact_Set 내보내기를 차단한다.

### Requirement 4: 연구 질문과 방어 가능한 차별성

**User Story:** 연구 책임자로서, 연구 기여를 반증 가능하고 과장 없는 질문으로 정의하고 싶다. 그래야 결과와 주장 범위를 일관되게 판정할 수 있다.

#### Acceptance Criteria

1. THE Research_Protocol SHALL RQ1 OSM 공간 일반화, RQ2 협력 포위 안정성, RQ3 관측·보상·배치 기여 및 RQ4 오프라인 LLM 가치의 네 연구 질문을 각각 식별한다.
2. WHEN Research_Protocol을 고정하면, THE Research_Protocol SHALL 각 연구 질문에 Analysis_Classification 두 값 중 정확히 하나를 배정한다.
3. WHEN Research_Protocol을 고정하면, THE Research_Protocol SHALL 각 연구 질문에 정확히 한 개의 primary outcome을 배정한다.
4. WHEN Research_Protocol을 고정하면, THE Research_Protocol SHALL 각 연구 질문에 비교 대상을 포함한 방향성 부등식과 단위가 있는 Practical_Threshold를 기록한다.
5. WHEN 연구 질문 결과를 판정하면, THE Research_System SHALL 사전등록 부등식, Practical_Threshold, 보정된 통계 결과 및 95% Confidence_Interval로 `supported`, `falsified`, `inconclusive` 중 정확히 하나를 배정한다.
6. WHEN Novelty_Claim을 등록하면, THE Research_System SHALL Related_Work_Matrix의 직접 비교 축 하나 이상과 verified Evidence_Record 하나 이상을 연결한다.
7. THE Research_System SHALL `최초`, `유일` 및 동등한 우선권 표현을 Novelty_Claim에서 사용하지 않는다.
8. WHEN Condition 실행을 종료하면, THE Research_System SHALL Execution_Status 세 값 중 정확히 하나와 상태 사유를 기록한다.
9. WHEN completed Condition이 불리하거나 영 효과를 산출하면, THE Research_System SHALL Execution_Status를 `completed`로 유지하고 부정 결과를 별도 결과 필드에 보존한다.
10. IF 연구 질문 분류, primary outcome, 판정 또는 Condition 상태가 누락·중복·기록 불일치이면, THEN THE Paper_Claim_Gate SHALL 의존 주장을 실패 처리한다.

### Requirement 5: 지도 출처와 시나리오의 배타적 분리

**User Story:** 실험 검토자로서, 실제 OSM과 fixture 및 서로 다른 종료 시나리오를 분리하고 싶다. 그래야 외적 타당성을 오해하지 않는다.

#### Acceptance Criteria

1. WHEN Model_Network를 등록하면, THE Research_System SHALL data kind를 Actual_OSM_Map 또는 Synthetic_Fixture 중 정확히 하나로 분류한다.
2. WHEN Model_Network를 등록하면, THE Research_System SHALL scenario를 Interior_Contained_Map 또는 Boundary_Escape_Map 중 정확히 하나로 분류한다.
3. WHEN Actual_OSM_Map을 등록하면, THE Research_System SHALL OSM 조회 geometry, 원본 출처, 취득 UTC, raw Content_Hash, 전처리 버전, metric 좌표계, network Content_Hash 및 원천 edge 식별자를 기록한다.
4. WHEN Synthetic_Fixture를 등록하면, THE Research_System SHALL 생성기 경로·버전, 생성 파라미터, 생성 seed, Content_Hash 및 시험 목적을 기록한다.
5. IF data kind 또는 scenario가 누락·중복·provenance 불일치이면, THEN THE Research_System SHALL Model_Network 등록을 실패 처리하고 성능 집계에서 제외한다.
6. WHEN 한 physical step에서 capture 조건과 boundary escape 조건이 함께 충족되면, THE Research_System SHALL capture를 우선 outcome으로 기록한다.
7. WHEN Interior_Contained_Map을 평가하면, THE Research_System SHALL outcome domain을 정확히 `{capture, timeout}`으로 제한한다.
8. WHEN Boundary_Escape_Map을 평가하면, THE Research_System SHALL outcome domain을 정확히 `{capture, escape, timeout}`으로 제한한다.
9. WHEN Evaluation_Episode를 등록하면, THE Research_System SHALL 해당 에피소드를 data kind와 scenario의 단일 교차 층에 정확히 한 번 배정한다.
10. IF Evaluation_Episode outcome이 scenario domain 밖이거나 둘 이상의 data-kind×scenario 층에 배정되면, THEN THE Research_System SHALL 해당 에피소드를 실패 처리하고 성능 집계에서 제외한다.
11. THE Research_System SHALL Actual_OSM_Map과 Synthetic_Fixture 및 두 scenario의 결과를 각각 별도 표와 별도 해석으로 보고한다.

### Requirement 6: 공간 분리 OSM 일반화

**User Story:** 연구자로서, 학습 도로 암기와 보지 않은 도로·도시 일반화를 구분하고 싶다. 그래야 OSM 일반화 주장을 방어할 수 있다.

#### Acceptance Criteria

1. WHEN Spatially_Disjoint_Split을 고정하면, THE Research_System SHALL train, validation 및 test polygon 각각의 geometry Content_Hash와 network Content_Hash를 기록한다.
2. WHEN Research_Protocol을 고정하면, THE Research_Protocol SHALL split 검사용 metric 좌표계와 0보다 큰 buffer 거리를 protocol 값으로 기록한다.
3. WHEN Spatially_Disjoint_Split을 검증하면, THE Research_System SHALL train·validation·test polygon 교차 면적이 0이고 protocol buffer 이내에 걸친 map element가 0인지 검사한다.
4. WHEN Spatially_Disjoint_Split을 구성하면, THE Research_System SHALL buffer와 교차하는 node 및 edge를 모든 split에서 제외한다.
5. WHEN Spatially_Disjoint_Split을 검증하면, THE Research_System SHALL split 쌍마다 공유 OSM 원천 edge 수가 0인지 검사한다.
6. THE Research_System SHALL test polygon, test map, test episode 및 test metric을 학습, 보상 조정, 조기 종료, 하이퍼파라미터 선택 또는 체크포인트 선택에 사용하지 않는다.
7. WHEN Held_Out_OSM_Evaluation을 실행하면, THE Research_System SHALL validation 기반 모델 선택 뒤 변경되지 않은 정책을 test split에 적용한다.
8. WHEN Cross_City_Zero_Shot_Evaluation을 실행하면, THE Research_System SHALL 학습 도시에 속하지 않은 최소 2개 도시의 Actual_OSM_Map에 동일한 고정 정책을 적용한다.
9. WHILE Cross_City_Zero_Shot_Evaluation을 실행할 때, THE Research_System SHALL 대상 도시 자료를 학습, 미세조정, 보상 조정, LLM 입력 또는 모델 선택에 사용하지 않는다.
10. WHEN 일반화 결과를 보고하면, THE Research_System SHALL train-distribution, spatial holdout 및 cross-city zero-shot 층을 분리한다.
11. IF polygon, buffer, shared-source-edge, provenance 또는 leakage 검사 중 하나가 실패하면, THEN THE Claim_Register SHALL 해당 실행과 파생 산출물을 일반화 증거에서 제외한다.
12. IF 평가가 동일 network에서 Evaluation_Episode seed만 분리하면, THEN THE Claim_Register SHALL 해당 평가를 in-distribution으로 분류한다.

### Requirement 7: 사전등록 표본 규모와 독립 반복

**User Story:** 통계 분석자로서, 우연한 seed와 불균형 표본에 의존하지 않는 결과를 얻고 싶다. 그래야 추정 안정성을 평가할 수 있다.

#### Acceptance Criteria

1. WHEN 기본 표본 계획을 고정하면, THE Research_Protocol SHALL Condition당 서로 다른 Training_Seed 5개를 배정한다.
2. WHEN 기본 표본 계획을 고정하면, THE Research_Protocol SHALL Condition당 Evaluation_Episode 500개를 배정한다.
3. WHEN Condition matrix에 표본을 배정하면, THE Research_System SHALL confirmatory Condition마다 동일한 Training_Seed 수와 Evaluation_Episode 수를 배정한다.
4. IF Resource_Ceiling이 기본 계획을 허용하지 않으면, THEN THE Research_System SHALL 결과 접근 전에 모든 confirmatory Condition에 공통으로 Training_Seed 3개 이상 5개 이하와 Evaluation_Episode 100개 이상 500개 이하를 고정한다.
5. WHEN Reduced_Power_Protocol을 적용하면, THE Research_System SHALL 축소 결정 입력, Resource_Ceiling 측정값, 실제 seed·episode 수 및 검정력 제한을 기록한다.
6. IF Condition이 Training_Seed 3개 미만이거나 Evaluation_Episode 100개 미만이면, THEN THE Research_System SHALL 해당 분석을 exploratory로 분류한다.
7. THE Research_System SHALL Checkpoint_Time을 Training_Seed로 계산하지 않는다.
8. WHEN 동일 Training_Seed의 여러 Checkpoint_Time을 분석하면, THE Research_System SHALL 해당 값들을 하나의 training replicate에 속한 반복 측정으로 기록한다.
9. WHEN Condition 결과를 집계하면, THE Research_System SHALL Training_Seed별 추정치와 Training_Seed 간 분산을 함께 보고한다.
10. IF Condition 간 seed 또는 episode 배정이 사전등록 계획과 다르면, THEN THE Research_System SHALL confirmatory 비교를 차단하고 불균형 사유를 기록한다.

### Requirement 8: 쌍체 비교와 통계 추론

**User Story:** 심사자로서, 정책 차이가 완전한 쌍체 계약과 불확실성으로 보고되기를 원한다. 그래야 통계적·실질적 효과를 함께 판단할 수 있다.

#### Acceptance Criteria

1. WHEN Episode_Case를 생성하면, THE Research_System SHALL map Content_Hash, data kind, scenario, 여섯 경찰 초기 상태, 도주자 초기 상태, 도주자 RNG state 또는 stream hash, 환경 설정 및 종료 설정을 봉인한다.
2. WHEN 두 정책을 비교하면, THE Research_System SHALL 동일 Episode_Case와 Training_Seed 대응 관계를 사용하는 Paired_Evaluation을 수행한다.
3. WHEN primary effect의 95% Confidence_Interval을 계산하면, THE Research_System SHALL Training_Seed를 outer cluster로 재표집하고 각 seed 안의 Episode_Case pair를 inner unit으로 재표집하는 hierarchical paired bootstrap을 사용한다.
4. WHEN bootstrap 횟수를 고정하면, THE Research_Protocol SHALL 기본 10,000회 또는 10,000회와 다른 양의 정수 횟수 및 선택 근거를 기록한다.
5. WHEN 정책 차이를 보고하면, THE Research_System SHALL two-sided test, 방향과 단위가 있는 Effect_Size, 95% Confidence_Interval, pair 수 및 Training_Seed 수를 보고한다.
6. WHEN 한 연구 질문 family에서 둘 이상의 confirmatory 가설을 검정하면, THE Research_System SHALL Holm 방법으로 보정한 p값을 보고한다.
7. WHEN 우월성 또는 비열등성을 판정하면, THE Research_System SHALL 통계 기준과 사전등록 Practical_Threshold를 모두 적용한다.
8. WHEN failed, interrupted 또는 missing Episode_Case가 발생하면, THE Research_System SHALL 정책별 건수·사유와 사전등록 주 분석 처리를 적용하고 complete-case 보조 민감도 분석을 함께 보고한다.
9. IF Episode_Case의 봉인 필드 또는 seed 대응 관계 중 하나가 비교 정책 사이에서 다르면, THEN THE Research_System SHALL paired 표시, paired test 및 paired Confidence_Interval 생성을 거부하고 mismatch 필드를 기록한다.
10. IF pair 보존, bootstrap 계층, 필수 통계 필드 또는 Holm family 검사가 실패하면, THEN THE Paper_Claim_Gate SHALL 해당 비교의 confirmatory 주장을 차단한다.

### Requirement 9: 관측과 보상 one-factor Ablation

**User Story:** 모델 개발자로서, 성능 변화가 관측·보상의 어떤 요소에서 발생하는지 알고 싶다. 그래야 핵심 기여와 복잡성을 분리할 수 있다.

#### Acceptance Criteria

1. WHEN 관측 Ablation을 고정하면, THE Research_System SHALL Observation_21D와 Observation_28D만 다른 one-factor Condition pair를 생성한다.
2. WHEN Observation_28D 계약을 감사하면, THE Research_System SHALL 추가 7개 값을 코드 순서대로 자기-도주자 거리, 자기-도주자 방위 sine, shifted cosine, 팀 최소 거리, 팀 평균 거리, angular coverage, near-officer fraction으로 기록한다.
3. WHEN Observation_28D 정규화를 감사하면, THE Research_System SHALL 현재 코드의 `min(distance/clip_distance_m,1)`, sine 뒤 최종 `[0,1]` vector clip, `(cosine+1)/2`, team minimum·mean distance clip, `1-max_angular_gap/(2π)`, 200 m 이내 경찰 비율 및 실행별 `clip_distance_m`을 코드 경로·Code_Revision과 함께 기록한다.
4. WHEN Observation_28D 계산 시점을 고정하면, THE Research_Protocol SHALL 경찰 행동 선택 직전의 동일 state snapshot을 decision timing으로 기록한다.
5. IF 추가 7개 값의 순서·정규화·decision timing이 감사된 코드 계약과 다르면, THEN THE Research_System SHALL Observation_28D 비교를 실패 처리한다.
6. WHEN 보상 Ablation을 고정하면, THE Research_System SHALL 감사된 full Reward_Component_Set과 각 구성요소를 하나씩 제거한 leave-one-component-out Condition을 비교한다.
7. WHEN 후퇴 보상 Ablation을 고정하면, THE Research_System SHALL 다른 요소가 같은 symmetric distance condition과 감사된 asymmetric regress condition을 비교한다.
8. WHEN Reward_Component_Set을 고정하면, THE Research_Protocol SHALL 실제 코드에서 감사한 구성요소, 계수, 부호, 단위 및 적용 시점을 기록하고 미감사 계수를 기존 사실로 표현하지 않는다.
9. WHEN 관측 또는 보상 Ablation을 보고하면, THE Research_System SHALL Capture_Metric과 Containment_Metric, Anti_Oscillation_Metric 및 Physical_Plausibility_Metric을 함께 보고한다.
10. WHEN Observation_21D와 Observation_28D를 비교하면, THE Research_System SHALL 각 조건의 parameter count와 FLOP estimate를 동일 계산 규칙으로 보고한다.
11. IF 관측 차원이 모델 용량을 바꾸면, THEN THE Research_System SHALL parameter count 차이가 1% 이하인 capacity-matched auxiliary comparison을 추가한다.
12. IF one-factor 이외의 Condition 필드, 학습 예산 또는 Paired_Evaluation 계약이 다르면, THEN THE Research_System SHALL 해당 Ablation의 인과적 기여 주장을 차단한다.

### Requirement 10: 배치 및 안정화 factorial Ablation

**User Story:** 연구자로서, 배치와 진동 억제 장치의 기여를 분리하고 싶다. 그래야 안정성 변화의 원인을 설명할 수 있다.

#### Acceptance Criteria

1. WHEN 배치 Ablation을 고정하면, THE Research_System SHALL global-only, ring-only 및 mixed Placement_Curriculum Condition을 포함한다.
2. WHEN 배치 Condition을 고정하면, THE Research_Protocol SHALL 배치 확률분포, support, feasibility rule 및 배치 RNG stream을 기록한다.
3. WHEN mixed Placement_Curriculum을 실행하면, THE Research_System SHALL protocol-fixed mixture probability와 component distribution으로만 배치를 생성한다.
4. WHEN 안정화 Ablation을 고정하면, THE Research_System SHALL U_Turn_Suppression off/on과 Hysteresis off/on의 정확히 4개 조합을 생성한다.
5. WHEN U_Turn_Suppression을 적용하면, THE Research_Protocol SHALL 역방향 판정식과 penalty 또는 mask 방식을 고정한다.
6. WHEN Hysteresis를 적용하면, THE Research_Protocol SHALL 유지 대상, 상태 전이, 최소 유지 조건 및 전환 조건을 고정한다.
7. WHEN 안정화 Condition을 실행하면, THE Immutable_Run_Manifest SHALL wrapper state, U-turn state, Hysteresis state 및 관련 RNG provenance를 기록한다.
8. WHEN 2×2 결과를 분석하면, THE Research_System SHALL U_Turn_Suppression 주효과, Hysteresis 주효과 및 두 요인의 interaction Effect_Size와 95% Confidence_Interval을 보고한다.
9. WHEN 안정화 결과를 보고하면, THE Research_System SHALL Capture_Metric, Containment_Metric, Anti_Oscillation_Metric 및 Physical_Plausibility_Metric의 trade-off를 같은 표에 보고한다.
10. IF capture가 개선되면서 행동 안정성 또는 물리적 타당성이 악화되면, THEN THE Research_System SHALL 개선 단독 주장을 차단하고 trade-off를 결과와 한계에 기록한다.
11. IF factorial arm이 누락·중복되거나 비대상 Condition 필드가 다르면, THEN THE Research_System SHALL 안정화 기여 분석을 실패 처리한다.

### Requirement 11: 수치 기준선과 선택 공정성

**User Story:** 심사자로서, 제안 방식이 강하고 공정한 수치 기준선과 비교되기를 원한다. 그래야 복잡한 방식의 필요성을 판단할 수 있다.

#### Acceptance Criteria

1. THE Research_System SHALL 모든 primary evaluation stratum에 proposed policy, budget-matched MAPPO 및 EncirclementPolice를 최소 기준선 집합으로 포함한다.
2. WHEN Research_Protocol을 고정하면, THE Research_System SHALL shortest-path와 greedy-intercept 기준선 각각의 공통 policy interface 적합성 검사를 실행하고 적합 여부를 기록한다.
3. WHERE shortest-path 또는 greedy-intercept 기준선이 공통 interface 적합성 검사를 통과하면, THE Research_System SHALL 통과한 기준선을 모든 primary evaluation stratum에 포함한다.
4. IF shortest-path 또는 greedy-intercept 기준선이 적합성 검사를 실패하면, THEN THE Research_System SHALL 해당 기준선을 `not_run`으로 보존하고 실패 사유를 보고한다.
5. WHEN 학습 정책을 비교하면, THE Research_System SHALL 환경 스텝, optimizer update, Resource_Ceiling, 모델 선택 자료 및 체크포인트 선택 규칙을 동일하게 적용한다.
6. WHEN 기준선 hyperparameter를 선택하면, THE Research_System SHALL train·validation 자료만 사용하고 정책별 검색 공간과 선택 규칙을 기록한다.
7. WHEN 기준선 결과를 보고하면, THE Research_System SHALL 동일 Episode_Case와 Capture, Containment, Anti_Oscillation, idleness, individual contribution 및 Physical_Plausibility metric 계약을 적용한다.
8. WHEN proposed policy가 기준선보다 열세이거나 동률이면, THE Research_System SHALL 해당 결과를 Condition row와 Claim_Register에 보존한다.
9. WHEN 정책 비용을 보고하면, THE Research_System SHALL 학습 환경 스텝, 학습 벽시계 시간, accelerator time, parameter count, FLOP estimate 및 episode당 추론 latency의 측정 방법과 값을 기록한다.
10. IF 최소 기준선이 누락되거나 selection·budget·metric·pairing 계약이 불공정하면, THEN THE Paper_Claim_Gate SHALL proposed policy 우월성 주장을 차단한다.

### Requirement 12: offline-only LLM 피드백

**User Story:** 연구자로서, LLM의 제한된 오프라인 가치만 수치 기준선과 비교하고 싶다. 그래야 언어 모델을 실시간 차량 제어와 혼동하지 않는다.

#### Acceptance Criteria

1. WHEN LLM_Feedback_Arm을 고정하면, THE Research_Protocol SHALL `trajectory_critique`, `role_assignment`, `curriculum_proposal` 중 정확히 한 작업을 allowlist task로 선택한다.
2. WHILE Evaluation_Episode가 진행될 때, THE Research_System SHALL 외부 또는 로컬 LLM 호출 수를 0으로 유지한다.
3. IF LLM에 교차로 행동, 조향, 가감속 또는 episode별 저수준 행동 선택이 요청되면, THEN THE Research_System SHALL 요청을 실행 전에 차단하고 위반을 기록한다.
4. WHEN LLM 입력을 구성하면, THE Research_System SHALL train·validation의 저장된 궤적·보상만 포함하고 입력 Content_Hash를 기록한다.
5. IF test 또는 cross-city 자료·label·metric이 LLM 입력이나 출력 선택에 포함되면, THEN THE Research_System SHALL 해당 arm과 모든 파생 Condition을 일반화·confirmatory 증거에서 제외한다.
6. WHEN LLM 출력을 후속 학습에 사용하면, THE Research_System SHALL schema 검사를 통과한 출력을 Content_Hash로 고정하고 외부 호출 없는 replay로 사용한다.
7. WHEN LLM_Feedback_Arm을 비교하면, THE Research_System SHALL Numerical_Baseline과 동일한 학습 환경 스텝, optimizer update, 평가 Episode_Case 및 Resource_Ceiling을 적용한다.
8. WHEN LLM 호출을 기록하면, THE Research_System SHALL provider, model identifier, model version, prompt, sampling setting, 입력·출력 Content_Hash, 실행 UTC, 비용 및 latency를 기록한다.
9. IF 외부 전송 후보에 PII, 실제 사건 자료, 비공개 위치 자료 또는 비공개 연구 산출물이 포함되면, THEN THE Research_System SHALL 외부 전송 바이트를 0으로 유지하고 차단 사유를 기록한다.
10. WHERE 외부 LLM 호출을 사용하지 않는 실행이면, THE Research_System SHALL 고정 출력 replay와 Numerical_Baseline의 핵심 비교를 완료할 수 있게 한다.
11. IF allowlist task, zero-call, leakage, replay, equal-budget 또는 provenance 검사 중 하나가 실패하면, THEN THE Paper_Claim_Gate SHALL LLM 가치 주장을 차단한다.

### Requirement 13: 행동 품질과 물리적 타당성 메트릭

**User Story:** 연구 검토자로서, 검거율 외에 여섯 경찰의 포위·행동·물리 품질을 확인하고 싶다. 그래야 보상 해킹과 비현실적 움직임을 탐지할 수 있다.

#### Acceptance Criteria

1. WHEN 첫 평가 결과에 접근하기 전에, THE Research_Protocol SHALL 모든 metric의 기호, 수식, 단위, 방향, 시간창, 집계 수준 및 Practical_Threshold를 고정한다.
2. WHEN angular coverage를 고정하면, THE Research_Protocol SHALL bearing 기준, 정렬, wrap-around, 최대 각도 간극 및 coverage 변환을 포함하는 단일 수식을 기록한다.
3. WHEN Anti_Oscillation_Metric을 계산하면, THE Research_System SHALL 경찰 6명 각각의 physical U-turn rate, protocol-window revisit rate 및 action·role switch rate를 기록한다.
4. WHEN idleness와 이동 품질을 계산하면, THE Research_System SHALL 경찰 6명 각각의 legal-move-available idle rate, approach distance, retreat distance 및 zero-displacement time을 기록한다.
5. WHEN individual contribution을 계산하면, THE Research_System SHALL 경찰 6명 각각의 capture-radius entry, exit-block duration 및 사전등록 neutral replacement를 사용한 leave-one-off difference를 기록한다.
6. WHEN Containment_Metric을 계산하면, THE Research_System SHALL angular coverage, maximum angular gap, blocked-exit fraction 및 reachable-region reduction을 기록한다.
7. WHEN Capture_Metric을 계산하면, THE Research_System SHALL scenario별 capture·escape·timeout indicator, minimum separation 및 사전등록 censoring rule을 적용한 capture-time estimand를 기록한다.
8. WHEN Physical_Plausibility_Metric을 계산하면, THE Research_System SHALL direction violation, contraflow, off-road, speed-limit violation, teleport, discontinuous segment transition, impossible immediate round-trip 및 invalid action의 정확히 8개 category를 기록한다.
9. WHEN officer metric을 집계하면, THE Research_System SHALL 경찰별 6개 row, team aggregate 및 metric 방향에 따른 worst-officer 값을 보고한다.
10. IF 한 Evaluation_Episode에서 hard physical violation 8개 category 중 하나의 count가 1 이상이면, THEN THE Research_System SHALL 해당 에피소드를 failure ledger에 원인 trace Content_Hash와 함께 기록한다.
11. IF metric 수식이 고정본과 다르거나 값이 비유한 수이거나 단위·경찰 row·worst 방향 검사가 실패하면, THEN THE Research_System SHALL 관련 집계와 Paper_Claim을 차단한다.

### Requirement 14: 불변 manifest와 완전 재개

**User Story:** 재현성 검토자로서, 실행 중단과 논문 산출물까지 정확히 재생하고 싶다. 그래야 결과 계보가 변경 없이 유지된다.

#### Acceptance Criteria

1. WHEN 실행을 생성하면, THE Research_System SHALL 고유 run identifier와 `draft` Immutable_Run_Manifest를 생성한다.
2. WHEN 실행이 terminal Execution_Status에 도달하면, THE Research_System SHALL manifest 필수 필드와 참조 Content_Hash를 검증한 뒤 manifest를 `sealed`로 전환한다.
3. WHILE manifest가 sealed 상태일 때, THE Research_System SHALL manifest와 참조 산출물의 in-place 변경을 거부한다.
4. IF sealed manifest 또는 참조 산출물의 변경이 필요하면, THEN THE Research_System SHALL 새 run identifier와 parent identifier를 가진 child manifest로 fork한다.
5. WHEN Resume_State를 저장하면, THE Resume_State SHALL actor·critic, optimizer, scheduler, scaler, episode·update·environment-step index, Python·NumPy·PyTorch CPU·device RNG, 환경·도주자·배치·sampler RNG, curriculum·Hysteresis state, running normalization, pending SMDP transition 및 rollout buffer를 포함한다.
6. WHEN 중단 실행의 재개를 요청하면, THE Research_System SHALL code, dependency, protocol, Condition, map, split, tensor shape 및 Resume_State Content_Hash compatibility gate를 실행한다.
7. IF compatibility gate가 실패하거나 필수 Resume_State가 누락되면, THEN THE Research_System SHALL 원 run 재개를 거부하고 새 lineage 또는 failed 상태를 기록한다.
8. WHEN 재개 동등성을 검증하면, THE Research_System SHALL 동일 seed의 continuous run과 interrupted-resumed run을 protocol-fixed checkpoint에서 비교한다.
9. WHEN 재개 동등성 허용오차를 고정하면, THE Research_Protocol SHALL 비교 필드별 절대·상대 허용오차와 결정론 backend의 exact-match 여부를 기록한다.
10. IF continuous run과 resumed run의 필수 state 또는 metric trace가 protocol tolerance를 초과하면, THEN THE Research_System SHALL resume gate를 실패 처리한다.
11. WHEN 표 cell 또는 그림 panel을 생성하면, THE Paper_Artifact_Set SHALL cell·panel identifier를 run identifier, 생성 script·Code_Revision, config Content_Hash, 입력 Content_Hash 및 출력 Content_Hash에 연결한다.
12. WHEN 표 또는 그림 재생을 검증하면, THE Research_System SHALL 기록된 입력으로 재생한 Content_Hash가 등록 출력 Content_Hash와 일치하는지 검사한다.
13. IF cell·panel provenance 또는 hash replay가 실패하면, THEN THE Paper_Claim_Gate SHALL 해당 표·그림 내보내기를 차단한다.

### Requirement 15: 프로토콜 고정과 누출 방지

**User Story:** 연구 책임자로서, 결과를 본 뒤 규칙이 바뀌거나 test 자료가 선택에 유입되지 않게 하고 싶다. 그래야 선택 편향을 통제할 수 있다.

#### Acceptance Criteria

1. WHEN 첫 confirmatory result artifact를 읽기 전에, THE Research_System SHALL Research_Protocol의 Content_Hash, 고정 UTC 및 signer를 기록하고 sealed 원본을 보존한다.
2. WHEN sealed Research_Protocol을 변경하려면, THE Research_System SHALL 원본을 덮어쓰지 않고 parent hash를 가진 새 protocol branch를 생성한다.
3. WHEN 분석을 등록하면, THE Research_System SHALL Analysis_Classification 두 값 중 정확히 하나를 기록한다.
4. IF 분석이 sealed protocol에 없거나 protocol branch에서 변경되면, THEN THE Research_System SHALL 해당 분석을 exploratory로 분류하고 변경 전후 hash와 사유를 보존한다.
5. THE Research_System SHALL test 및 cross-city map·episode·metric handle을 tuning, early stopping, reward design, LLM 입력, baseline selection 및 checkpoint selection interface에서 제외한다.
6. IF test 또는 cross-city handle이 선택·조정 경로에 유입되면, THEN THE Research_System SHALL 오염 실행과 모든 descendant run·analysis·claim을 confirmatory 증거에서 제외한다.
7. WHEN failure·missing·interruption 처리를 고정하면, THE Research_Protocol SHALL 상태별 outcome mapping, 제외 가능 조건, 민감도 분석 및 planned-case accounting 규칙을 기록한다.
8. WHEN 실패 실행을 처리하면, THE Research_System SHALL 사전등록 규칙, run identifier, 실패 산출물 Content_Hash 및 사유를 보존한다.
9. IF 성공한 Training_Seed만 포함한 aggregate를 생성하면, THEN THE Research_System SHALL 해당 aggregate를 exploratory로 분류하고 전체 사전등록 seed aggregate를 함께 보고한다.
10. IF protocol hash, analysis 분류, leakage lineage 또는 failure-handling 검사가 실패하면, THEN THE Paper_Claim_Gate SHALL 관련 confirmatory 내보내기를 차단한다.

### Requirement 16: 완전한 논문 산출물

**User Story:** 논문 저자로서, 실험과 주장을 완전하고 재현 가능한 논문 자료로 변환하고 싶다. 그래야 독립 검토가 가능하다.

#### Acceptance Criteria

1. THE Paper_Artifact_Set SHALL 제목, 초록, 서론, 관련 연구, 문제 정의, 방법, 실험, 결과, 논의, 타당성 한계, 윤리, 재현성, 결론 및 참고문헌 절을 포함한다.
2. THE Paper_Artifact_Set SHALL 상태·관측·행동 공간, SMDP 전이, 학습 목적함수, 보상, 종료 및 모든 보고 metric의 방정식과 기호 정의를 포함한다.
3. THE Paper_Artifact_Set SHALL Related_Work_Matrix, Citation_Ledger 및 sealed Research_Protocol을 포함한다.
4. WHEN 결과 표를 생성하면, THE Paper_Artifact_Set SHALL Training_Seed별 값, aggregate, 95% Confidence_Interval, two-sided test, Effect_Size, pair 수 및 seed 수를 포함한다.
5. WHEN 결과 그림을 생성하면, THE Paper_Artifact_Set SHALL split, learning curve, seed variation, behavior metric 및 결정론 규칙으로 선택한 대표 success·failure trajectory를 포함한다.
6. WHEN 대표 trajectory 선택 규칙을 고정하면, THE Research_Protocol SHALL 결과 label을 보기 전에 metric과 tie-break ordering을 기록한다.
7. WHEN Condition matrix를 보고하면, THE Paper_Artifact_Set SHALL 모든 Ablation, baseline 및 LLM arm에 Execution_Status와 부정·실패·미실행 사유를 포함한다.
8. THE Paper_Artifact_Set SHALL 내부·외적·구성·통계 결론 타당성, 윤리 위험 및 재현성 제한을 구분한다.
9. WHEN 논문 수치 또는 인용을 생성하면, THE Research_System SHALL 수치를 sealed manifest·analysis Content_Hash에 대조하고 인용을 verified Citation_Ledger에 대조한다.
10. IF source reconciliation에서 출처 없는 수치, unverified 인용, 누락 Condition 또는 hash 불일치가 발견되면, THEN THE Paper_Claim_Gate SHALL Paper_Artifact_Set 내보내기를 차단한다.

### Requirement 17: 현실적 한계와 Field_Readiness 제한

**User Story:** 윤리 검토자로서, 시뮬레이션 결과가 실제 현장 안전성으로 확대되지 않기를 원한다. 그래야 적용 범위가 명확하다.

#### Acceptance Criteria

1. THE Claim_Register SHALL Field_Readiness_Claim을 모든 실험 결과와 관계없이 `unsupported`로 유지한다.
2. WHEN 성능 결과를 서술하면, THE Research_System SHALL 실제 교통·차량 동역학, 센서 오차, 통신 지연·손실, 인간 행동 및 법적·운영 제약이 모델링되지 않았음을 명시한다.
3. THE Paper_Artifact_Set SHALL Research_System이 실제 경찰 지휘·추격 결정을 대체하지 않는 연구용 시뮬레이션이라는 문구를 포함한다.
4. IF 시뮬레이션 정책이 Practical_Threshold를 넘는 결과를 산출하면, THEN THE Research_System SHALL 결과를 현장 안전성, 범죄 감소 또는 실제 검거율 향상으로 변환하지 않는다.
5. WHEN 위험을 보고하면, THE Research_System SHALL 과도한 추격 유도, 지역 편향, 감시 확대, 자동화 편향 및 정책 오용의 정확히 5개 risk category를 각각 기록한다.
6. THE Paper_Artifact_Set SHALL 현장 적용 전에 교통 미시모사, 센서 불확실성 평가, 인간 참여 평가, 안전 검증, 법률 검토 및 통제된 pilot의 정확히 6개 별도 validation category가 필요하다고 명시한다.
7. WHEN pre-field validation 상태를 기록하면, THE Research_System SHALL 6개 category마다 별도 identifier와 `not_performed`, `failed`, `passed` 중 정확히 하나를 기록한다.
8. THE Research_System SHALL 한 pre-field validation category의 결과를 다른 category의 통과 근거로 대체하지 않는다.
9. IF Field_Readiness 상태, 5개 risk 또는 6개 validation category 검사가 실패하면, THEN THE Paper_Claim_Gate SHALL 논문 내보내기를 차단한다.

### Requirement 18: Future_Work_System 범위 분리

**User Story:** 프로젝트 관리자로서, 핵심 연구와 장기 제품 아이디어를 분리하고 싶다. 그래야 연구 기여가 미검증 통합 기능에 의존하지 않는다.

#### Acceptance Criteria

1. THE Paper_Artifact_Set SHALL CCTV를 Future_Work_System으로만 분류한다.
2. THE Paper_Artifact_Set SHALL ANPR을 Future_Work_System으로만 분류한다.
3. THE Paper_Artifact_Set SHALL 실시간 지휘 dashboard를 Future_Work_System으로만 분류한다.
4. THE Paper_Artifact_Set SHALL drone 및 heterogeneous agent 연동을 Future_Work_System으로만 분류한다.
5. THE Research_System SHALL Future_Work_System의 구현·demo·예상 효과를 Novelty_Claim, 실험 성공 기준 또는 Field_Readiness 근거로 사용하지 않는다.
6. WHEN Future_Work_System demo를 생성하면, THE Research_System SHALL demo를 `future_work_demo`로 표시하고 primary result artifact와 분리된 경로에 저장한다.
7. THE Paper_Artifact_Set SHALL 현재 연구 범위와 Future_Work_System을 서로 다른 절에 배치한다.
8. IF Future_Work_System 산출물이 Novelty_Claim, success evidence 또는 primary 결과에 연결되면, THEN THE Paper_Claim_Gate SHALL 해당 연결과 주장을 차단한다.

### Requirement 19: 자동 검증과 실패 폐쇄 품질 게이트

**User Story:** 품질 담당자로서, 연구 불변식과 보고 누락을 자동 검출하고 싶다. 그래야 논문 결과가 구현 오류에 의존하지 않는다.

#### Acceptance Criteria

1. WHEN exactly-one gate를 실행하면, THE Research_System SHALL Prior_Result_Status, Evidence_Type, Test_Status, screening, data kind, scenario, Execution_Status 및 Analysis_Classification의 누락·중복·허용값·provenance 일치를 자동 검증한다.
2. WHEN map-outcome gate를 실행하면, THE Research_System SHALL Actual_OSM_Map과 Synthetic_Fixture provenance, capture 우선순위, scenario outcome domain 및 data-kind×scenario 단일 배정을 자동 검증한다.
3. WHEN split-leakage-hash gate를 실행하면, THE Research_System SHALL polygon hash, 양의 protocol buffer, buffer 교차 element 0, shared source edge 0, test·cross-city leakage 0 및 Content_Hash 일치를 자동 검증한다.
4. WHEN metric-six-officer gate를 실행하면, THE Research_System SHALL protocol formula·unit·direction, finite value, outcome conservation, officer row 6개, team·worst 집계 및 physical violation 8개 category를 자동 검증한다.
5. WHEN paired-statistics gate를 실행하면, THE Research_System SHALL Episode_Case 전체 계약, seed-outer/case-inner bootstrap, two-sided test, Effect_Size, 95% Confidence_Interval, n, Holm 보정, Practical_Threshold 및 missing sensitivity를 자동 검증한다.
6. WHEN resume gate를 실행하면, THE Research_System SHALL Resume_State 필수 필드 save-load round trip, compatibility, pending SMDP·buffer state 및 interrupted-continuous tolerance 동등성을 자동 검증한다.
7. WHEN claim-citation gate를 실행하면, THE Research_System SHALL claim별 Evidence_Type, sealed protocol, Analysis_Classification, Execution_Status, split, statistics, manifest, source Content_Hash, verified citation 및 body-reference-ledger orphan 0을 자동 검증한다.
8. WHEN Paper_Artifact_Set gate를 실행하면, THE Research_System SHALL 필수 절, 모든 Condition 상태, seed-level table 필드, deterministic trajectory provenance, cell·panel hash replay, validity·ethics·reproducibility 및 source reconciliation을 자동 검증한다.
9. IF required gate 하나가 실패하면, THEN THE Research_System SHALL 의존 결과·표·그림·주장을 confirmatory Paper_Artifact_Set으로 내보내지 않고 실패 기록을 보존한다.
10. THE Paper_Claim_Gate SHALL Field_Readiness_Claim을 성능과 관계없이 실패시키고 `unsupported` 상태로 유지한다.
11. WHILE 기본 자동 시험을 실행할 때, THE Research_System SHALL 외부 OSM 호출 수와 외부 LLM 호출 수를 각각 0으로 유지하고 Synthetic_Fixture 또는 고정 cache만 사용한다.
12. WHEN 외부 OSM 또는 LLM integration test를 실행하면, THE Research_System SHALL 기본 시험과 분리된 실행 identifier, 명시적 opt-in, 서비스·버전, 실행 UTC 및 입력·출력 Content_Hash를 기록한다.
13. IF 기본 시험이 외부 호출을 시도하거나 integration test provenance가 불완전하면, THEN THE Research_System SHALL 해당 시험을 실패 처리하고 연구 산출물 생성을 차단한다.
