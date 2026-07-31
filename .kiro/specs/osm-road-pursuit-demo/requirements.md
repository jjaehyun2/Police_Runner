# Requirements Document

## Introduction

`osm-road-pursuit-demo`는 대전의 제한된 실제 OpenStreetMap(OSM) 도로 영역을 모델 호환 방향 그래프로 변환하고, 경찰 6대용 도로 추격 정책을 검증 가능한 방식으로 실행·평가·시각화·API 제공하는 경진대회 데모이다. 이 기능은 기존 코드와 체크포인트가 존재한다는 사실과 실제 OSM 도로에서의 성능이 입증되었다는 주장을 분리한다. OSM 성능, 검거율, 일반화, 실시간성 및 안전 효과는 본 요구사항에 따른 실험 전에는 검증되지 않은 결과로 취급한다.

## Glossary

- **OSM_Road_Pursuit_Demo**: 이 사양이 정의하는 OSM 도로 추격 데모 전체 시스템.
- **OSM**: OpenStreetMap에서 제공하는 도로·교차로·방향·형상 데이터.
- **Bounded_Area**: 북·남·동·서 위경도와 면적 제한으로 지정한 유한한 OSM 조회 영역.
- **Daejeon_Example**: 설정 파일에 좌표와 이름을 명시한 대전 내 재현용 Bounded_Area.
- **Raw_OSM_Graph**: OSM에서 받은 원본 노드 식별자, 방향 간선, 도로 속성 및 형상을 포함한 그래프.
- **Model_Network**: 의사결정 교차로와 방향성 도로 세그먼트로 구성되고 모델 입력 계약을 만족하도록 전처리된 그래프.
- **Intersection**: 차량이 다음 방향을 선택하는 Model_Network의 정점.
- **Segment**: 두 Intersection을 잇는 주행 가능한 방향성 도로 구간.
- **Graph_Coarsener**: Raw_OSM_Graph의 비의사결정 중간 노드를 축약하여 Model_Network를 생성하는 구성요소.
- **Degree_Splitter**: 진출 차수가 5를 넘는 Intersection을 합법 방향 손실 없이 최대 진출 차수 5의 결정론적 보조 라우팅 구조로 변환하는 구성요소.
- **Canonical_ID**: Raw OSM 식별자와 분리된 결정론적 연속 내부 식별자.
- **Mapping_Manifest**: Raw OSM 식별자·Canonical_ID·축약된 형상 간 대응 관계를 기록한 자료.
- **Network_Metadata**: 조회 범위, 원본, 좌표계, 전처리 버전, 그래프 통계, 해시 및 생성 시각을 기록한 자료.
- **Offline_Cache**: 네트워크 연결 없이 재사용할 수 있도록 원본과 전처리 결과를 보관하는 로컬 저장소.
- **Checkpoint**: 학습된 경찰 정책의 가중치 파일.
- **Checkpoint_Manifest**: Checkpoint의 구조, 관측 계약, 학습 네트워크 및 파일 해시를 명시한 메타데이터.
- **Checkpoint_Contract**: 경찰 수 6, `fixed_max_degree` 5, 행동 차원 6, 관측 차원 21, MLP 은닉층 `[128, 128]` 및 명시적 관측 프로필 식별자를 포함한 현재 정책 구조 호환 조건.
- **Legacy_Observation_Profile**: 기존 합성 격자 학습 코드의 `agent_id_onehot(6)`, `current_step(1)`, `my_position(1)`, `my_progress(1)`, `nearest_boundary_dist(1)`, `neighbors(5)`, `other_positions(6)`를 키 이름 오름차순으로 평탄화하는 21차원 입력 규약. 위치와 이웃 값은 학습 그래프의 연속 노드 번호 의미를 가지므로 임의 OSM 네트워크에 대한 의미 호환성을 제공하지 않는다.
- **OSM_Observation_Profile**: `agent_id_onehot(6)`, 정규화 시간(1), 세그먼트 진행률(1), 도주자까지의 정규화 방향 최단거리(1), 경계까지의 정규화 방향 최단거리(1), 결정론적으로 정렬된 최대 5개 진출 방향의 정규화 도주자 거리 점수(5), 다른 경찰 5대와 도주자까지의 정규화 방향 최단거리(6)를 `float32`로 평탄화하는 21차원 입력 규약. Raw OSM 식별자와 Canonical_ID 숫자 자체는 입력하지 않는다.
- **Observation_Contract**: 선택한 관측 프로필의 식별자, 21개 필드 의미와 순서, 자료형, 정규화 기준, 도달 불가 값, 패딩 값 및 행동 슬롯 정렬 규칙을 포함한 버전 규약.
- **Compatibility_Validator**: Checkpoint_Manifest, Checkpoint 가중치 형상, Network_Metadata 및 실행 설정의 호환성을 판정하는 구성요소.
- **Inference_Adapter**: Model_Network와 차량 상태를 Observation_Contract와 행동 마스크로 변환하고 Checkpoint 출력을 Segment 선택으로 해석하는 구성요소.
- **OSM_Training_Path**: OSM Model_Network에서 호환 정책을 미세조정하거나 재학습하고 검증하는 절차.
- **Episode_Runner**: 경찰 6대와 휴리스틱 도주자 1대의 에피소드를 실행하는 구성요소.
- **Baseline_Police**: 각 경찰 위치에서 도주자의 현재 또는 마지막 관측 위치까지의 방향 최단거리 감소를 우선하고 동률을 결정론적으로 해소하는 비학습 기준 경찰 전략.
- **Episode_Outcome**: `capture`, `escape`, `timeout` 중 하나인 에피소드 최종 결과.
- **Deterministic_Seed**: 데이터 전처리, 초기 배치, 도주자 행동 및 정책 행동의 난수 상태를 재현하는 정수.
- **Metrics_Collector**: Episode_Outcome, 에피소드 길이 및 추론 지연시간을 집계하는 구성요소.
- **Renderer**: OSM 도로, 차량, 이동 경로 및 체포 반경을 이미지로 표현하는 구성요소.
- **Competition_Export**: 경진대회 제출용 프레임 이미지, 요약 그림, 표 및 재현 메타데이터 묶음.
- **Demo_API**: 네트워크 로드·검증·추천·에피소드 실행·결과 조회·내보내기를 제공하는 HTTP JSON 인터페이스.
- **Claim_Status**: 기능 또는 결과를 `implemented`, `verified`, `experimental`, `unsupported` 중 하나로 분류한 상태.
- **Property_Based_Test**: 다수의 생성 입력에 대해 불변식·라운드트립·멱등성·변형 관계를 확인하는 자동화 시험.
- **HTTP_JSON**: HTTP 상태 코드와 JSON 본문을 사용하는 Demo_API 교환 형식.
- **MLP**: 입력층, 은닉층 및 출력층으로 구성된 다층 퍼셉트론 신경망.
- **PNG**: Competition_Export 프레임에 사용하는 무손실 래스터 이미지 형식.

## 현재 기준선과 검증 경계

- 코드 검토로 OSM 로더, 체크포인트 추론 모듈, FastAPI 기반 API, 격자형 도로 에피소드 환경 및 matplotlib Renderer의 구현 파일은 확인되었다. 각 구성요소의 실행 가능 상태는 자동 시험 전까지 `implemented`로 확정하지 않는다.
- 현재 OSM 변환은 Raw OSM 노드를 연속 번호로 바꾸고 간선을 Segment로 옮기는 수준이다. 의사결정점 기반 Graph_Coarsener와 도달 가능성 보존 검증은 확인되지 않았다.
- 현재 경찰 6대 학습 스크립트는 `fixed_max_degree=5`, 관측 차원 21, 행동 차원 6, MLP `[128, 128]`를 사용한다. 실제 Checkpoint 파일의 존재·완전성·가중치 형상은 Compatibility_Validator 실행 전까지 검증되지 않은 상태이다.
- 현재 독립 추론 모듈의 기본값은 경찰 4대와 MLP `[64, 64]`이므로 경찰 6대 Checkpoint_Contract와 자동 호환된다고 간주하지 않는다.
- 현재 체크포인트 저장 코드는 가중치를 저장하지만 Checkpoint_Manifest를 포함하지 않는다. 학습 영역과 OSM 호환성은 별도 메타데이터와 실험으로 입증해야 한다.
- 실제 대전 OSM 에피소드의 검거율·탈출률·타임아웃률·지연시간과 일반화 성능을 입증하는 재현 실험은 확인되지 않았다. 해당 결과는 요구사항 14의 검증 전까지 `experimental` 또는 `unsupported`이다.

## Requirements

### Requirement 1: 구현 사실과 실험 결과의 분리

**User Story:** 경진대회 평가자로서, 구현된 기능과 아직 검증되지 않은 성능 주장을 구분하고 싶다. 그래야 데모 증거를 과장 없이 평가할 수 있다.

#### Acceptance Criteria

1. THE OSM_Road_Pursuit_Demo SHALL 각 주요 기능과 성능 주장에 Claim_Status와 근거 파일 또는 실험 식별자를 기록한다.
2. WHEN 기존 OSM 로더, 추론 엔진, API 또는 Renderer가 감지되면, THE OSM_Road_Pursuit_Demo SHALL 실행 가능한 자동 시험을 통과한 범위만 `implemented` 또는 `verified`로 표시한다.
3. IF OSM 에피소드 실험 결과가 존재하지 않으면, THEN THE OSM_Road_Pursuit_Demo SHALL OSM 검거율, 탈출률, 일반화 성능, 지연시간 및 안전 효과를 `experimental`로 표시한다.
4. IF Checkpoint가 합성 격자 도로에서만 학습되었으면, THEN THE OSM_Road_Pursuit_Demo SHALL 실제 OSM 도로에 즉시 일반화된다는 주장을 `unsupported`로 표시한다.
5. WHEN 실험 결과를 표시하면, THE OSM_Road_Pursuit_Demo SHALL Checkpoint 해시, Model_Network 해시, 실행 설정, Deterministic_Seed 목록 및 에피소드 수를 함께 표시한다.

### Requirement 2: 제한된 실제 OSM 영역 선택 및 로드

**User Story:** 데모 운영자로서, 대전 예시를 포함한 제한 영역의 실제 도로를 명시적으로 선택하고 싶다. 그래야 조회 규모와 재현 조건을 통제할 수 있다.

#### Acceptance Criteria

1. THE OSM_Road_Pursuit_Demo SHALL Daejeon_Example의 이름, 북·남·동·서 좌표, 자동차 도로 유형 및 설정 버전을 제공한다.
2. WHEN 유효한 Bounded_Area가 선택되면, THE OSM_Road_Pursuit_Demo SHALL 해당 범위의 자동차 주행 가능 Raw_OSM_Graph를 로드한다.
3. WHEN Raw_OSM_Graph를 로드하면, THE OSM_Road_Pursuit_Demo SHALL 조회 범위, OSM 출처, 조회 시각, 도로 유형 및 원본 해시를 Network_Metadata에 기록한다.
4. IF 지역명 조회 결과가 요청된 Bounded_Area를 초과하면, THEN THE OSM_Road_Pursuit_Demo SHALL 결과를 명시된 좌표 범위로 절단하고 작업 상태를 `TRUNCATED`로 기록한다.
5. WHERE 사용자가 사용자 정의 영역을 선택하면, THE OSM_Road_Pursuit_Demo SHALL Daejeon_Example과 동일한 검증·캐시·메타데이터 절차를 적용한다.

### Requirement 3: 입력 검증과 오류 보고

**User Story:** API 및 명령행 사용자로서, 잘못된 지도·차량·실험 입력을 실행 전에 거부하고 싶다. 그래야 무효 결과를 성능 결과로 오인하지 않는다.

#### Acceptance Criteria

1. WHEN Bounded_Area가 입력되면, THE OSM_Road_Pursuit_Demo SHALL 모든 좌표가 유한수이고 `north > south`, `east > west`, 위도 범위가 `[-90, 90]`, 경도 범위가 `[-180, 180]`인지 검증한다.
2. IF Bounded_Area의 면적이 설정된 최대 조회 면적을 초과하면, THEN THE OSM_Road_Pursuit_Demo SHALL 허용 최대값과 입력값을 포함한 오류를 반환한다.
3. WHEN 차량 위치가 입력되면, THE OSM_Road_Pursuit_Demo SHALL 경찰 위치 6개와 도주자 위치 1개가 Model_Network의 유효한 Canonical_ID 또는 Segment 위 좌표인지 검증한다.
4. IF Model_Network에 Intersection 또는 Segment가 없으면, THEN THE OSM_Road_Pursuit_Demo SHALL 에피소드 및 추론 실행을 오류로 종료한다.
5. IF Model_Network의 주행 가능 부분에 경찰 6대와 도주자 1대를 배치할 수 없으면, THEN THE OSM_Road_Pursuit_Demo SHALL 배치 실패 사유를 포함한 오류를 반환한다.
6. IF 입력 JSON, 네트워크 파일 또는 메타데이터의 스키마 버전이 지원되지 않으면, THEN THE OSM_Road_Pursuit_Demo SHALL 지원 버전 목록을 포함한 오류를 반환한다.

### Requirement 4: 오프라인 캐시와 출처 보존

**User Story:** 데모 운영자로서, 네트워크 장애 중에도 이미 받은 대전 지도를 재현 가능하게 사용하고 싶다. 그래야 발표와 시험이 외부 서비스 상태에 의존하지 않는다.

#### Acceptance Criteria

1. WHEN OSM 조회가 성공하면, THE Offline_Cache SHALL Raw_OSM_Graph, Model_Network, Mapping_Manifest 및 Network_Metadata를 원자적으로 저장한다.
2. THE Offline_Cache SHALL Bounded_Area 좌표, 자동차 도로 유형, 원본 스키마 버전 및 전처리 설정 해시로 캐시 키를 생성한다.
3. WHEN 동일한 캐시 키로 다시 요청하면, THE Offline_Cache SHALL 저장 자료의 해시와 스키마를 검증한 후 저장 자료를 재사용한다.
4. WHILE 네트워크 연결을 사용할 수 없을 때, THE Offline_Cache SHALL 유효한 일치 항목이 있으면 외부 조회 없이 해당 항목을 반환한다.
5. IF 네트워크 연결을 사용할 수 없고 유효한 일치 항목이 없으면, THEN THE Offline_Cache SHALL 필요한 캐시 키와 사전 준비 방법을 포함한 오류를 반환한다.
6. IF 캐시 파일의 해시가 Network_Metadata와 일치하지 않으면, THEN THE Offline_Cache SHALL 손상 항목을 실행에 사용하지 않고 손상 상태를 보고한다.
7. WHEN 캐시 자료를 내보내면, THE Offline_Cache SHALL OSM 출처 표시와 데이터 생성 시각을 함께 보존한다.

### Requirement 5: 방향 도로 그래프 전처리와 코어싱

**User Story:** 모델 개발자로서, OSM 도로를 모델이 실행할 수 있는 교차로·세그먼트 네트워크로 축약하고 싶다. 그래야 실제 도로 방향성과 거리를 보존하면서 입력 규모를 제어할 수 있다.

#### Acceptance Criteria

1. WHEN Raw_OSM_Graph가 제공되면, THE Graph_Coarsener SHALL 교차로, 분기점, 합류점, 막다른 끝, 영역 경계 및 도로 속성 전환점을 의사결정 Intersection으로 식별한다.
2. WHEN 두 의사결정 Intersection 사이에 비분기 노드 연쇄가 존재하면, THE Graph_Coarsener SHALL 연쇄를 하나의 방향성 Segment로 축약한다.
3. WHEN 노드 연쇄를 축약하면, THE Graph_Coarsener SHALL 구성 도로 길이의 합, 진행 방향, 도로 형상 및 원본 도로 참조를 Segment에 보존한다.
4. WHEN 일방통행 도로를 변환하면, THE Graph_Coarsener SHALL OSM이 허용한 방향의 Segment만 생성한다.
5. WHEN 양방향 도로를 변환하면, THE Graph_Coarsener SHALL 각 허용 방향을 별도 Segment로 표현한다.
6. WHEN Model_Network를 생성하면, THE Graph_Coarsener SHALL Raw OSM 식별자와 무관한 결정론적 Canonical_ID를 부여한다.
7. WHEN Model_Network를 생성하면, THE Graph_Coarsener SHALL 모든 축약 관계를 Mapping_Manifest에 기록한다.
8. WHEN 동일한 Raw_OSM_Graph와 전처리 설정을 두 번 처리하면, THE Graph_Coarsener SHALL 동일한 Model_Network 해시와 Mapping_Manifest를 생성한다.
9. IF 코어싱 전후의 경계 간 방향 도달 가능성이 달라지면, THEN THE Graph_Coarsener SHALL 결과를 호환 불가로 판정하고 실행을 중단한다.

### Requirement 6: 모델 네트워크 구조 검증

**User Story:** 모델 개발자로서, 실제 교차로의 높은 차수와 연결성 문제를 명시적으로 처리하고 싶다. 그래야 다섯 방향 제한 때문에 도로를 조용히 누락하지 않는다.

#### Acceptance Criteria

1. THE Compatibility_Validator SHALL Model_Network의 Intersection 수, Segment 수, 최대 진출 차수, 연결 성분, 경계 Intersection 수 및 도달 불가 Segment 수를 계산한다.
2. IF Intersection의 진출 차수가 5를 초과하면, THEN THE Degree_Splitter SHALL 모든 합법 진출 방향을 보존하는 결정론적 보조 라우팅 구조를 생성하거나 Model_Network를 호환 불가로 판정한다.
3. IF Intersection의 진출 차수가 5를 초과하면, THEN THE Degree_Splitter SHALL 여섯 번째 이후 Segment를 잘라낸 결과를 생성하지 않는다.
4. WHEN 진출 차수가 5 이하인 Intersection에 행동 마스크를 생성하면, THE Inference_Adapter SHALL 각 진출 Segment에 하나의 유효 행동과 대기 행동 하나를 대응시킨다.
5. IF Segment가 존재하지 않는 행동이 선택되면, THEN THE Inference_Adapter SHALL 해당 출력을 실행 결과에서 오류로 분류한다.
6. WHEN Model_Network 검증이 성공하면, THE Compatibility_Validator SHALL 검증 통계와 판정 사유를 Network_Metadata에 기록한다.

### Requirement 7: 체크포인트와 관측 계약 검증

**User Story:** 모델 운영자로서, 현재 경찰 6대 체크포인트를 정확한 신경망·관측 계약으로만 로드하고 싶다. 그래야 형상 일치와 의미 호환성을 혼동하지 않는다.

#### Acceptance Criteria

1. THE Compatibility_Validator SHALL 현재 Checkpoint_Contract를 경찰 수 6, `fixed_max_degree` 5, 행동 차원 6, 관측 차원 21 및 MLP 은닉층 `[128, 128]`로 검증한다.
2. THE Compatibility_Validator SHALL 선택한 Observation_Contract의 프로필 식별자, 필드 의미, 필드 순서, 각 필드 크기, 자료형, 패딩 값, 정규화 규칙 및 행동 슬롯 정렬 규칙을 검증한다.
3. WHEN Checkpoint를 로드하면, THE Compatibility_Validator SHALL Checkpoint 해시와 `police_actor` 각 층의 가중치 형상을 Checkpoint_Manifest와 대조한다.
4. IF Checkpoint_Manifest가 없으면, THEN THE Compatibility_Validator SHALL 가중치 키와 형상에서 관측 차원 21, 행동 차원 6 및 은닉층 `[128, 128]` 여부를 자동 추론하고 경찰 수·관측 의미·학습 영역처럼 추론할 수 없는 계약을 `unknown`으로 반환한다.
5. IF Checkpoint_Contract의 항목이 실행 설정과 일치하지 않으면, THEN THE Compatibility_Validator SHALL 추론을 시작하지 않고 불일치 항목별 기대값과 실제값을 반환한다.
6. IF Checkpoint의 Observation_Contract가 Legacy_Observation_Profile이거나 확인되지 않았으면, THEN THE Compatibility_Validator SHALL OSM_Observation_Profile과의 의미 호환성을 통과로 판정하지 않는다.
7. WHEN 호환성 검사가 완료되면, THE Compatibility_Validator SHALL 구조 호환성과 의미 호환성을 분리한 통과·경고·실패·미확인 항목을 기계 판독 가능한 보고서로 출력한다.
8. THE OSM_Observation_Profile SHALL Raw OSM 식별자와 Canonical_ID 숫자 자체 대신 정규화된 진행률과 방향 최단거리 기반 특성을 사용한다.

### Requirement 8: 호환 추론과 OSM 학습 경로

**User Story:** 연구자로서, 기존 정책의 실험적 직접 실행과 OSM용 미세조정 또는 재학습을 분리하고 싶다. 그래야 형상 호환을 임의 OSM 네트워크의 성능 일반화로 오인하지 않는다.

#### Acceptance Criteria

1. WHEN Checkpoint_Manifest의 관측 프로필이 실행 Observation_Contract와 일치하고 Checkpoint_Contract와 Model_Network 검사가 통과하면, THE Inference_Adapter SHALL 해당 Observation_Contract와 행동 마스크를 사용하여 경찰 6대의 행동을 생성한다.
2. WHEN Inference_Adapter가 행동을 생성하면, THE Inference_Adapter SHALL 행동 인덱스, 선택 Segment, 다음 Intersection, 유효성 및 정책 확률을 모두 반환한다.
3. IF Inference_Adapter가 필수 행동 결과 항목을 생성할 수 없으면, THEN THE Inference_Adapter SHALL 부분 결과를 반환하지 않고 전체 요청을 오류로 종료한다.
4. IF Raw OSM 식별자 또는 Canonical_ID 숫자 자체가 OSM_Observation_Profile 관측값에 입력되도록 요청되면, THEN THE Inference_Adapter SHALL 요청을 거부하고 Mapping_Manifest는 표시·추적용으로만 사용하는 방법을 반환한다.
5. WHERE 사용자가 Legacy_Observation_Profile Checkpoint의 직접 OSM 실행을 명시적으로 선택하면, THE OSM_Road_Pursuit_Demo SHALL 구조 호환성 검사를 수행하고 결과를 `experimental`로 격리하며 OSM 성능 증거와 검증 집계에서 제외한다.
6. IF Checkpoint의 의미 호환성이 통과하지 못하면, THEN THE OSM_Road_Pursuit_Demo SHALL OSM_Training_Path를 제시하고 직접 추론 결과를 검증 성능 증거로 집계하지 않는다.
7. THE OSM_Training_Path SHALL 학습·검증·시험 Model_Network 분할, OSM_Observation_Profile, 경찰 6대, 최대 진출 차수 5, MLP `[128, 128]`, Deterministic_Seed 및 Checkpoint 저장 규칙을 명시한다.
8. WHERE 구조 호환 Checkpoint를 초기 가중치로 사용하면, THE OSM_Training_Path SHALL 미세조정 실행과 처음부터 재학습한 기준 실행을 분리해 기록한다.
9. WHEN OSM 학습이 완료되면, THE OSM_Training_Path SHALL 학습 데이터 범위와 겹치지 않는 시험 영역의 결과가 포함된 Checkpoint_Manifest를 생성한다.
10. IF 시험 영역 평가가 완료되지 않으면, THEN THE OSM_Training_Path SHALL 새 Checkpoint를 `experimental`로 표시한다.
11. WHEN OSM_Observation_Profile을 유효한 Model_Network와 차량 상태에 적용하면, THE Inference_Adapter SHALL 식별자 재번호화와 무관한 21개의 유한한 `float32` 값을 생성한다.

### Requirement 9: 재현 가능한 경찰-도주자 에피소드

**User Story:** 실험 담당자로서, 동일한 OSM 지도와 시드로 경찰 대 도주자 에피소드를 반복하고 싶다. 그래야 결과를 비교하고 재현할 수 있다.

#### Acceptance Criteria

1. WHEN 에피소드 묶음이 요청되면, THE Episode_Runner SHALL Model_Network, Checkpoint, Checkpoint_Manifest, 실행 설정, 에피소드 수 및 Deterministic_Seed 목록을 입력으로 사용한다.
2. WHEN 에피소드가 초기화되면, THE Episode_Runner SHALL 경찰 6대와 도주자 1대를 유효한 주행 가능 위치에 결정론적으로 배치한다.
3. WHEN 에피소드가 진행되면, THE Episode_Runner SHALL 차량을 Segment 방향을 따라 이동시키고 Intersection에서만 다음 Segment 행동을 적용한다.
4. WHEN 경찰과 도주자 사이의 미터 단위 거리가 설정된 체포 반경 이하가 되면, THE Episode_Runner SHALL Episode_Outcome을 `capture`로 기록한다.
5. WHEN 도주자가 지정 경계 Segment를 통해 Bounded_Area 밖으로 이동하면, THE Episode_Runner SHALL Episode_Outcome을 `escape`로 기록한다.
6. WHEN 최대 스텝에 도달할 때까지 체포 또는 탈출이 발생하지 않으면, THE Episode_Runner SHALL Episode_Outcome을 `timeout`으로 기록한다.
7. WHEN 체포와 탈출 조건이 같은 스텝에 충족되면, THE Episode_Runner SHALL 체포 판정을 먼저 적용하고 판정 우선순위를 결과에 기록한다.
8. WHEN 동일한 입력 해시와 Deterministic_Seed로 결정론 모드를 두 번 실행하면, THE Episode_Runner SHALL 동일한 초기 배치, 행동 열, 차량 상태 열 및 Episode_Outcome을 생성한다.
9. IF 에피소드 도중 유효 행동이 하나도 없으면, THEN THE Episode_Runner SHALL 대기 행동을 적용하고 해당 사건을 에피소드 로그에 기록한다.

### Requirement 10: 결과 및 지연시간 메트릭

**User Story:** 경진대회 참가자로서, 검거·탈출·타임아웃과 추론 속도를 재현 가능한 표로 얻고 싶다. 그래야 모델 효과와 실시간성 주장을 실험으로 평가할 수 있다.

#### Acceptance Criteria

1. WHEN 에피소드가 종료되면, THE Metrics_Collector SHALL Episode_Outcome, 스텝 수, 모의 경과시간, 최소 경찰-도주자 거리 및 종료 위치를 기록한다.
2. WHEN 에피소드 묶음이 완료되면, THE Metrics_Collector SHALL `capture`, `escape`, `timeout`의 건수와 전체 에피소드 대비 비율을 계산한다.
3. THE Metrics_Collector SHALL `capture`, `escape`, `timeout` 건수의 합이 완료 에피소드 수와 같은지 검증한다.
4. THE Metrics_Collector SHALL 각 에피소드에 정확히 하나의 Episode_Outcome이 존재하는지 검증한다.
5. WHEN 정책 추론이 실행되면, THE Metrics_Collector SHALL 준비 동작을 제외한 경찰 6대 전체 행동 생성의 벽시계 지연시간을 밀리초로 측정한다.
6. WHEN 지연시간 표본을 요약하면, THE Metrics_Collector SHALL 표본 수, 평균, 중앙값, 95백분위수, 최댓값, 실행 장치 및 준비 동작 횟수를 기록한다.
7. WHEN 성능 수치를 보고하면, THE Metrics_Collector SHALL 신뢰구간 계산 방법, 에피소드 수 및 Deterministic_Seed 목록을 함께 기록한다.
8. IF 에피소드 수가 설정된 최소 평가 에피소드 수보다 작으면, THEN THE Metrics_Collector SHALL 비율을 예비 결과로 표시한다.
9. IF 정책 추론이 실행되지 않으면, THEN THE Metrics_Collector SHALL 지연시간을 `-1`로 기록하고 측정 상태를 `not_measured`로 표시한다.

### Requirement 11: OSM 지도 시각화 및 경진대회 내보내기

**User Story:** 발표자로서, 실제 OSM 도로 위에서 차량과 포위 과정을 프레임 및 요약 그림으로 보여주고 싶다. 그래야 제안서의 작동 방식을 검증 가능한 시각 자료로 제시할 수 있다.

#### Acceptance Criteria

1. WHEN 에피소드 프레임을 렌더링하면, THE Renderer SHALL Model_Network의 도로 형상과 진행 방향을 미터 단위 좌표로 표시한다.
2. WHEN 에피소드 프레임을 렌더링하면, THE Renderer SHALL 경찰 6대, 도주자, 현재 Segment 및 누적 이동 경로를 서로 구별되는 표식으로 표시한다.
3. WHEN 에피소드 프레임을 렌더링하면, THE Renderer SHALL 설정된 체포 반경을 지도 축척과 일치하는 원으로 표시한다.
4. WHEN 에피소드가 종료되면, THE Renderer SHALL Episode_Outcome, 에피소드 식별자, Deterministic_Seed 및 현재 스텝을 이미지에 표시한다.
5. WHEN 프레임 내보내기가 요청되면, THE Competition_Export SHALL 시작 상태부터 종료 상태까지 순번이 정렬된 PNG 이미지를 생성한다.
6. WHEN 요약 그림 내보내기가 요청되면, THE Competition_Export SHALL 결과 비율, 에피소드 길이 분포, 추론 지연시간 분포 및 대표 경로 그림을 생성한다.
7. WHEN Competition_Export를 생성하면, THE Competition_Export SHALL 이미지 해상도, 파일 형식, 생성 코드 버전, 입력 해시 및 Claim_Status를 기록한 목록 파일을 포함한다.
8. IF 한글 글꼴을 사용할 수 없으면, THEN THE Renderer SHALL 글자가 누락되지 않는 대체 글꼴 또는 영문 표기를 사용하고 경고를 기록한다.

### Requirement 12: API 제공과 검증

**User Story:** 응용 개발자로서, 지도 로드부터 추천·실험·내보내기까지 API로 호출하고 검증하고 싶다. 그래야 데모를 외부 화면과 재현 스크립트에 연결할 수 있다.

#### Acceptance Criteria

1. THE Demo_API SHALL 상태 확인, 네트워크 로드, 캐시 파일 로드, 호환성 검사, 행동 추천, 에피소드 실행, 메트릭 조회 및 Competition_Export 생성을 위한 버전이 명시된 HTTP JSON 작업을 제공한다.
2. WHEN 행동 추천 요청이 유효하면, THE Demo_API SHALL 경찰 6대 각각의 행동 인덱스, 선택 Segment, 다음 Intersection, 정책 확률, 호환성 판정 및 추론 지연시간을 반환한다.
3. WHEN 에피소드 실행 요청이 유효하면, THE Demo_API SHALL 실행 식별자, 입력 해시, Deterministic_Seed 목록 및 결과 조회 위치를 반환한다.
4. IF 요청 본문이 요구 스키마를 위반하면, THEN THE Demo_API SHALL 필드 경로와 위반 사유를 포함한 HTTP 422 응답을 반환한다.
5. IF 요청한 네트워크 또는 실행 식별자가 없으면, THEN THE Demo_API SHALL 식별자를 포함한 HTTP 404 응답을 반환한다.
6. IF Checkpoint 또는 Model_Network가 호환되지 않으면, THEN THE Demo_API SHALL 불일치 목록과 OSM_Training_Path 안내를 포함한 HTTP 409 응답을 반환한다.
7. IF 내부 실행 오류가 발생하면, THEN THE Demo_API SHALL 비밀값과 로컬 절대 경로를 제외한 오류 식별자를 포함한 HTTP 500 응답을 반환한다.
8. THE Demo_API SHALL 각 작업의 요청·성공 응답·오류 응답 예시와 스키마를 API 문서로 제공한다.
9. WHEN API 검증 모음을 실행하면, THE Demo_API SHALL 유효 요청, 경계값, 잘못된 Canonical_ID, 경찰 수 불일치, 호환성 실패 및 Offline_Cache 사용 사례를 검사한다.

### Requirement 13: 자동 시험과 속성 기반 정확성 검증

**User Story:** 품질 담당자로서, 그래프 변환·관측·종료·집계의 불변식을 다양한 입력에서 자동 검증하고 싶다. 그래야 예시 몇 개로 발견하기 어려운 오류를 찾을 수 있다.

#### Acceptance Criteria

1. WHEN Property_Based_Test가 유효한 방향 그래프를 생성하면, THE Graph_Coarsener SHALL 생성된 모든 Segment의 시작·끝 Canonical_ID가 Model_Network에 존재하도록 보존한다.
2. WHEN Property_Based_Test가 진출 차수 5 이하의 방향 그래프를 생성하면, THE Inference_Adapter SHALL 진출 Segment 수와 유효 이동 행동 수가 같고 대기 행동이 유효하도록 행동 마스크를 생성한다.
3. WHEN Property_Based_Test가 Raw OSM 식별자만 일대일로 변경한 동형 그래프를 생성하면, THE Graph_Coarsener SHALL Canonical_ID 기준으로 동등한 Model_Network를 생성한다.
4. WHEN Property_Based_Test가 동일한 Model_Network에 코어싱을 반복 적용하면, THE Graph_Coarsener SHALL 첫 적용과 동등한 결과를 생성한다.
5. WHEN Property_Based_Test가 유효한 Model_Network를 저장한 후 다시 로드하면, THE Offline_Cache SHALL 저장 전과 동등한 Intersection, Segment, Mapping_Manifest 및 Network_Metadata 핵심 필드를 복원한다.
6. WHEN Property_Based_Test가 유효한 OSM_Observation_Profile 상태를 생성하면, THE Inference_Adapter SHALL Raw OSM 식별자와 Canonical_ID 숫자 자체를 포함하지 않는 정확히 21개의 유한한 `float32` 값으로 평탄화한다.
7. WHEN Property_Based_Test가 잘못된 좌표, 식별자, 경찰 수, 진출 차수 또는 체크포인트 형상을 하나 이상 생성하면, THE Compatibility_Validator SHALL 다른 검증 충돌의 존재와 관계없이 성공 결과 대신 하나 이상의 분류된 오류를 반환한다.
8. WHEN Property_Based_Test가 임의의 Episode_Outcome 열을 생성하면, THE Metrics_Collector SHALL 각 결과 건수의 합을 전체 열 길이와 같게 계산한다.
9. WHEN 단위 시험 모음을 실행하면, THE Episode_Runner SHALL 체포 우선순위, 탈출 경계 통과 및 최대 스텝 타임아웃을 각각 대표 예제로 검증한다.
10. WHEN 통합 시험 모음을 실행하면, THE OSM_Road_Pursuit_Demo SHALL 고정된 소형 캐시 지도에서 로드·전처리·검증·에피소드·메트릭·렌더링·API 응답의 전체 흐름을 검증한다.
11. WHEN 외부 OSM 연결 시험을 실행하면, THE OSM_Road_Pursuit_Demo SHALL 최대 세 개의 대표 Bounded_Area 예제로 조회와 출처 메타데이터를 검증한다.
12. WHILE 기본 자동 시험 모음을 실행할 때, THE OSM_Road_Pursuit_Demo SHALL 외부 OSM 서비스 호출 없이 Offline_Cache 또는 시험 자료를 사용한다.
13. WHEN Property_Based_Test가 Raw OSM 식별자만 일대일로 변경한 동형 Model_Network와 대응 차량 상태를 생성하면, THE Inference_Adapter SHALL OSM_Observation_Profile에서 동일한 관측 벡터와 행동 마스크를 생성한다.

### Requirement 14: 실험 검증 및 경진대회 보고

**User Story:** 연구 책임자로서, 실제 OSM 성능 주장을 사전에 정의한 실험으로 검증하고 싶다. 그래야 경진대회 자료가 재현 가능하고 과장되지 않는다.

#### Acceptance Criteria

1. WHEN OSM 성능 주장을 `verified`로 표시하기 전에, THE OSM_Road_Pursuit_Demo SHALL 평가 영역, Checkpoint, 도주자 규칙, 경찰 수, 체포 반경, 최대 스텝, 에피소드 수, Deterministic_Seed 목록 및 성공 기준을 고정한 실험 계획을 기록한다.
2. WHEN OSM 평가를 실행하면, THE Episode_Runner SHALL 학습에 사용된 영역과 사용되지 않은 영역의 결과를 분리한다.
3. WHEN OSM 평가를 실행하면, THE Episode_Runner SHALL 학습된 경찰 정책과 Baseline_Police를 동일한 초기 배치, 도주자 행동 난수열 및 종료 설정에서 비교한다.
4. WHEN 비교 결과를 보고하면, THE Metrics_Collector SHALL 정책별 `capture`, `escape`, `timeout`, 에피소드 길이 및 지연시간을 동일한 산식으로 제시한다.
5. IF 사전 성공 기준을 충족하지 못하면, THEN THE OSM_Road_Pursuit_Demo SHALL 결과를 실패 또는 예비 결과로 보고하고 OSM 성능 주장을 `verified`로 변경하지 않는다.
6. IF 실험 코드, 입력 자료 또는 Checkpoint를 재현할 수 없으면, THEN THE OSM_Road_Pursuit_Demo SHALL 해당 결과를 경진대회 성능 근거에서 제외한다.
7. WHEN Competition_Export를 완성하면, THE OSM_Road_Pursuit_Demo SHALL 현재 구현 기능, 검증된 결과, 알려진 제한, 미검증 가설 및 후속 OSM_Training_Path를 별도 절로 구분한다.
8. THE OSM_Road_Pursuit_Demo SHALL 경찰 지휘 보조 연구 데모이며 실제 현장 지휘 결정을 대체하지 않는다는 제한 문구를 Competition_Export에 포함한다.