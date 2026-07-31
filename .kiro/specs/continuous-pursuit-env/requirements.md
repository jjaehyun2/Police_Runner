# Requirements Document

## Introduction

본 문서는 기존 이산 그래프 기반 추격-도주 RL 시뮬레이션을 2단계로 확장하는 요구사항을 정의한다. 1단계에서는 기존 이산 환경의 경찰 모델을 커리큘럼 학습과 장기 학습을 통해 강화하고, 2단계에서는 연속 2D 좌표 공간에서 속도와 방향을 제어하는 새로운 환경을 구현한다. 최종 목적은 실시간으로 경찰에게 최적 배치 및 이동 방향을 제시하는 모델을 학습하는 것이다.

## Glossary

- **Discrete_Environment**: 기존 그래프 기반 이산 노드 이동 추격-도주 시뮬레이션 환경이다.
- **Continuous_Environment**: 연속 2D 좌표 공간에서 에이전트가 속도와 방향을 제어하며 이동하는 새로운 추격-도주 시뮬레이션 환경이다.
- **Road_Segment**: 2D 평면 위의 선분 또는 곡선으로 표현된 도로 구간이다. 에이전트는 Road_Segment 위에서만 이동할 수 있다.
- **Road_Map**: 복수의 Road_Segment와 교차점(Intersection)으로 구성된 2D 도로 네트워크이다.
- **Intersection**: 2개 이상의 Road_Segment가 만나는 점이다. 에이전트가 방향을 전환할 수 있는 지점이다.
- **Police_Agent**: 도망자를 추격하여 체포하는 것을 목표로 하는 경찰 차량 에이전트이다.
- **Fugitive_Agent**: 경찰의 추격을 피해 맵 경계를 벗어나 탈출하는 것을 목표로 하는 도망자 차량 에이전트이다.
- **Capture_Radius**: 경찰이 도망자를 체포한 것으로 판정하는 최소 접근 거리(유클리드 거리)이다.
- **Map_Boundary**: Road_Map의 외곽 경계선이다. 도망자가 이 경계를 넘으면 탈출 성공이다.
- **Curriculum_Learning**: 쉬운 맵에서 어려운 맵으로 점진적으로 난이도를 높여가며 학습하는 방법이다.
- **MAPPO**: Multi-Agent Proximal Policy Optimization. 본 시스템에서 사용하는 멀티 에이전트 강화학습 알고리즘이다.
- **Self_Play**: 경찰 팀과 도망자가 동일 환경에서 상호 대전하며 동시에 학습하는 방식이다.
- **Action_Space**: 에이전트가 선택할 수 있는 행동의 집합이다. 연속 환경에서는 속도와 방향의 연속 값이다.
- **Observation_Space**: 에이전트가 관측할 수 있는 상태 정보의 집합이다.
- **Episode**: 시뮬레이션의 한 에피소드. 초기 상태에서 시작하여 종료 조건이 달성될 때까지의 전체 과정이다.
- **Max_Steps**: 하나의 Episode에서 허용되는 최대 시뮬레이션 스텝 수이다.
- **Transfer_Learning**: 이산 환경에서 학습된 정책 지식을 연속 환경으로 전이하는 학습 기법이다.

## Requirements

### Requirement 1: 이산 환경 커리큘럼 학습

**User Story:** 개발자로서, 기존 이산 환경에서 경찰 모델의 성능을 커리큘럼 학습을 통해 단계적으로 향상시키고 싶다. 이를 통해 복잡한 맵에서도 높은 승률을 달성할 수 있다.

#### Acceptance Criteria

1. THE Curriculum_Manager SHALL 학습 난이도를 3단계 이상으로 구분하고, 각 단계별 맵 파라미터(노드 수, 연결 밀도, 경계 노드 비율)를 설정 파일에서 로드한다.
2. WHEN 현재 난이도 단계에서 경찰 승률이 설정된 임계값(기본값 90%)을 일정 에피소드 수(기본값 500 에피소드) 동안 유지하면, THE Curriculum_Manager SHALL 다음 난이도 단계로 자동 전환한다.
3. THE Curriculum_Manager SHALL 난이도 전환 시 이전 단계에서 학습된 모델 가중치를 유지하고, 새로운 난이도 환경에서 학습을 지속한다.
4. THE Curriculum_Manager SHALL 각 난이도 단계의 시작 에피소드, 전환 시점, 해당 단계 승률을 로그 파일에 기록한다.
5. IF 최종 난이도 단계에서 목표 승률 임계값에 도달하면, THEN THE Curriculum_Manager SHALL 학습 완료를 보고하고 최종 모델 체크포인트를 저장한다.

### Requirement 2: 이산 환경 장기 학습 실행

**User Story:** 개발자로서, 10,000 에피소드 이상의 장기 학습을 안정적으로 수행하고 싶다. 이를 통해 경찰 모델의 완전한 수렴을 보장할 수 있다.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL 설정 파일에 지정된 max_episodes(최소 10,000) 동안 학습을 수행하며, 에피소드 간 메모리 누수 없이 안정적으로 실행한다.
2. THE Training_Pipeline SHALL 학습 진행 중 매 checkpoint_interval(기본 100) 에피소드마다 모델 체크포인트를 저장하고, 학습 중단 시 마지막 체크포인트에서 재개할 수 있는 기능을 제공한다.
3. WHEN 학습이 외부 요인으로 중단되면, THE Training_Pipeline SHALL 마지막으로 저장된 체크포인트와 에피소드 번호를 기반으로 학습을 이어서 재개한다.
4. THE Training_Pipeline SHALL 매 에피소드마다 경찰 승률, 도망자 승률, 평균 에피소드 길이, 평균 보상을 추적하고, 설정된 로깅 간격(기본 50 에피소드)마다 이 통계를 로그 파일과 표준 출력에 기록한다.
5. WHILE Self_Play 학습이 진행되는 동안, THE Training_Pipeline SHALL 경찰 팀과 도망자 팀의 정책을 매 에피소드마다 동시에 업데이트하여 양측 모두 점진적으로 강화된다.

### Requirement 3: 연속 2D 도로 맵 정의

**User Story:** 개발자로서, 연속 2D 좌표 공간에서 도로 네트워크를 정의하고 싶다. 이를 통해 실제 도로 형상에 가까운 시뮬레이션이 가능하다.

#### Acceptance Criteria

1. THE Road_Map SHALL 2D 평면 위의 선분(직선 도로)과 곡선(베지어 곡선, 최대 3차)으로 구성된 Road_Segment의 집합으로 표현된다.
2. THE Road_Map SHALL 각 Road_Segment에 대해 시작점 좌표(x, y), 끝점 좌표(x, y), 도로 폭(width), 제한 속도(speed_limit)를 속성으로 포함한다.
3. THE Road_Map SHALL 2개 이상의 Road_Segment가 만나는 점을 Intersection으로 자동 식별하고, Intersection 목록을 유지한다.
4. THE Road_Map SHALL 맵의 외곽 경계를 직사각형 영역(min_x, min_y, max_x, max_y)으로 정의하며, Road_Segment의 끝점 중 경계에 접하는 점을 경계 출구(Boundary_Exit)로 표시한다.
5. THE Road_Map SHALL 설정 파일(YAML 또는 JSON)에서 도로 구조를 로드하거나, 프로그래밍 방식으로 그리드형, 방사형, 또는 랜덤 도로 네트워크를 생성하는 기능을 제공한다.
6. THE Road_Map SHALL 모든 Road_Segment가 하나 이상의 Intersection 또는 Boundary_Exit와 연결되어 연결 그래프를 형성함을 보장한다.
7. IF Road_Map 생성 시 연결되지 않은 Road_Segment가 존재하면, THEN THE Road_Map SHALL 해당 세그먼트를 제외하고 경고 메시지를 출력한다.

### Requirement 4: 연속 환경 에이전트 이동 모델

**User Story:** 개발자로서, 에이전트가 연속 2D 공간에서 속도와 방향을 제어하며 도로 위를 이동하게 하고 싶다. 이를 통해 현실적인 차량 이동을 시뮬레이션할 수 있다.

#### Acceptance Criteria

1. THE Continuous_Environment SHALL 각 에이전트의 상태를 위치(x, y), 속도(speed), 방향(heading, 0~2π 라디안)으로 정의한다.
2. WHEN 에이전트가 속도와 방향을 지정하면, THE Continuous_Environment SHALL 에이전트를 현재 위치에서 해당 방향으로 속도에 비례한 거리만큼 이동시키되, 이동 경로가 Road_Segment 위에 있을 때만 이동을 허용한다.
3. IF 에이전트의 이동 경로가 Road_Segment를 벗어나면, THEN THE Continuous_Environment SHALL 에이전트를 가장 가까운 Road_Segment 위의 유효한 위치로 보정(클램핑)하고 속도를 0으로 설정한다.
4. WHEN 에이전트가 Intersection에 도달하면, THE Continuous_Environment SHALL 에이전트가 해당 Intersection에 연결된 모든 Road_Segment 중 하나를 선택하여 방향을 전환할 수 있도록 허용한다.
5. THE Continuous_Environment SHALL 에이전트의 최대 속도를 현재 위치한 Road_Segment의 speed_limit 이하로 제한한다.
6. THE Continuous_Environment SHALL 시뮬레이션 시간 스텝(dt)을 고정값(기본 0.1초)으로 설정하고, 매 스텝마다 에이전트의 위치를 position += speed * direction * dt 공식으로 갱신한다.

### Requirement 5: 연속 환경 행동 공간

**User Story:** 개발자로서, 에이전트의 행동을 연속 값(속도, 방향)으로 정의하고 싶다. 이를 통해 세밀한 이동 제어가 가능하다.

#### Acceptance Criteria

1. THE Continuous_Environment SHALL 각 에이전트의 행동 공간을 2차원 연속 공간으로 정의한다: 속도 제어(0.0 ~ max_speed)와 방향 제어(0.0 ~ 2π 라디안).
2. THE Continuous_Environment SHALL 행동 공간을 gymnasium.spaces.Box로 정의하며, 하한은 [0.0, 0.0], 상한은 [max_speed, 2π]로 설정한다.
3. WHEN 에이전트가 max_speed를 초과하는 속도 값을 출력하면, THE Continuous_Environment SHALL 해당 값을 max_speed로 클리핑한다.
4. WHEN 에이전트가 2π를 초과하거나 0 미만의 방향 값을 출력하면, THE Continuous_Environment SHALL 해당 값을 0~2π 범위로 정규화(modulo 연산)한다.
5. THE Continuous_Environment SHALL 경찰과 도망자에게 동일한 행동 공간 구조를 적용하되, max_speed는 설정 파일을 통해 에이전트 유형별로 개별 설정 가능하게 한다.

### Requirement 6: 연속 환경 관측 공간

**User Story:** 개발자로서, 에이전트가 연속 환경에서 의사결정에 필요한 풍부한 상태 정보를 관측할 수 있게 하고 싶다. 이를 통해 효과적인 추격-도주 전략 학습이 가능하다.

#### Acceptance Criteria

1. THE Continuous_Environment SHALL 각 에이전트에게 자신의 현재 위치(x, y), 현재 속도(speed), 현재 방향(heading)을 관측값으로 제공한다.
2. THE Continuous_Environment SHALL 각 Police_Agent에게 Fugitive_Agent의 현재 위치(x, y)를 관측값으로 제공한다(전역 관측).
3. THE Continuous_Environment SHALL 각 에이전트에게 다른 모든 Police_Agent의 현재 위치(x, y)를 고정 길이 배열로 제공하며, 설정된 최대 경찰 수보다 실제 경찰 수가 적을 경우 나머지를 (-1, -1)로 패딩한다.
4. THE Continuous_Environment SHALL 각 에이전트에게 맵 경계(상하좌우 4방향)까지의 최단 거리를 관측값으로 제공한다.
5. THE Continuous_Environment SHALL 각 에이전트에게 현재 위치에서 가장 가까운 Boundary_Exit까지의 유클리드 거리를 관측값으로 제공한다.
6. THE Continuous_Environment SHALL 관측 공간을 gymnasium.spaces.Box로 정의하며, 모든 관측 값은 정규화된 부동소수점(float32) 배열로 표현한다.
7. THE Continuous_Environment SHALL 경찰 수가 가변적(3~20대)일 때, 관측 공간의 크기를 최대 경찰 수(20대) 기준으로 고정하여 신경망 입력 차원을 일정하게 유지한다.

### Requirement 7: 연속 환경 종료 조건

**User Story:** 개발자로서, 연속 환경에서의 체포와 탈출 조건을 명확히 정의하고 싶다. 이를 통해 에이전트가 목표 지향적 학습을 수행할 수 있다.

#### Acceptance Criteria

1. WHEN Police_Agent 중 하나 이상이 Fugitive_Agent와의 유클리드 거리가 Capture_Radius(기본값 5.0 단위) 이내에 도달하면, THE Continuous_Environment SHALL 경찰 승리로 Episode를 종료한다(terminated=True).
2. WHEN Fugitive_Agent의 위치(x, y)가 Map_Boundary를 벗어나면, THE Continuous_Environment SHALL 도망자 승리로 Episode를 종료한다(terminated=True).
3. WHEN 현재 Step 수가 Max_Steps(기본값 1000)에 도달하면, THE Continuous_Environment SHALL 타임아웃으로 Episode를 종료한다(truncated=True).
4. IF 동일 Step에서 복수의 종료 조건이 동시에 충족되면, THEN THE Continuous_Environment SHALL 도망자 탈출 > 경찰 체포 > 타임아웃 순의 우선순위에 따라 하나의 종료 원인만 판정한다.
5. WHEN Episode가 종료되면, THE Continuous_Environment SHALL 종료 원인(police_capture, fugitive_escape, timeout)과 최종 에이전트 위치를 info 딕셔너리에 포함하여 반환한다.

### Requirement 8: 연속 환경 보상 설계

**User Story:** 개발자로서, 연속 환경에서 에이전트의 추격-도주 전략 학습을 유도하는 보상 구조를 설계하고 싶다. 이를 통해 경찰은 빠른 포위를, 도망자는 효율적 탈출을 학습한다.

#### Acceptance Criteria

1. WHEN 경찰 체포로 Episode가 종료되면, THE Continuous_Environment SHALL Police_Agent에게 +1.0의 보상을, Fugitive_Agent에게 -1.0의 보상을 부여한다.
2. WHEN 도망자 탈출로 Episode가 종료되면, THE Continuous_Environment SHALL Fugitive_Agent에게 +1.0의 보상을, Police_Agent에게 -1.0의 보상을 부여한다.
3. WHEN 타임아웃으로 Episode가 종료되면, THE Continuous_Environment SHALL 모든 에이전트에게 -0.5의 보상을 부여한다.
4. WHILE Episode가 진행 중일 때, THE Continuous_Environment SHALL 매 스텝마다 Police_Agent에게 이전 스텝 대비 Fugitive_Agent와의 유클리드 거리 감소량에 비례하는 shaping 보상을 부여하되, 단일 스텝 shaping 보상의 절대값은 0.1을 초과하지 않는다.
5. WHILE Episode가 진행 중일 때, THE Continuous_Environment SHALL 매 스텝마다 Fugitive_Agent에게 이전 스텝 대비 가장 가까운 Boundary_Exit까지의 거리 감소량에 비례하는 shaping 보상을 부여하되, 단일 스텝 shaping 보상의 절대값은 0.1을 초과하지 않는다.
6. WHILE Episode가 진행 중일 때, THE Continuous_Environment SHALL Police_Agent들이 Fugitive_Agent를 기준으로 서로 다른 방향(120도 이상 간격)에서 3대 이상 접근하고 있을 경우 각 Police_Agent에게 0.05의 포위 협력 보상을 부여한다.
7. IF 에이전트가 Road_Segment를 벗어나는 이동을 시도하면, THEN THE Continuous_Environment SHALL 해당 에이전트에게 -0.02의 페널티를 부여한다.

### Requirement 9: 연속 환경 Gymnasium 호환성

**User Story:** 개발자로서, 연속 환경이 Gymnasium 인터페이스를 완전히 준수하도록 하고 싶다. 이를 통해 기존 RL 라이브러리와 원활하게 통합할 수 있다.

#### Acceptance Criteria

1. THE Continuous_Environment SHALL gymnasium.Env를 상속하고, reset(), step(), render() 메서드와 observation_space, action_space 속성을 구현한다.
2. WHEN reset()이 호출되면, THE Continuous_Environment SHALL 모든 에이전트를 Road_Map 위의 유효한 위치에 배치하고, seed 파라미터가 제공된 경우 동일 seed에 대해 동일한 초기 배치를 보장하며, (observations, info) 튜플을 반환한다.
3. WHEN step()이 호출되면, THE Continuous_Environment SHALL 모든 에이전트의 행동을 동시에 처리하고, (observations, rewards, terminated, truncated, info) 튜플을 반환한다.
4. THE Continuous_Environment SHALL 경찰 수를 설정 파일을 통해 3대에서 20대까지 조정할 수 있으며, 환경 생성 시 지정된 경찰 수만큼 Police_Agent를 생성한다.
5. THE Continuous_Environment SHALL gymnasium.utils.env_checker를 사용한 호환성 검증을 통과한다.
6. IF reset() 호출 없이 step()이 호출되면, THEN THE Continuous_Environment SHALL RuntimeError를 발생시킨다.

### Requirement 10: MAPPO 기반 연속 환경 학습

**User Story:** 개발자로서, 연속 환경에서 MAPPO 알고리즘을 사용하여 경찰 팀과 도망자를 학습시키고 싶다. 이를 통해 최적 추격 전략을 도출한다.

#### Acceptance Criteria

1. THE Continuous_Training_Pipeline SHALL 연속 행동 공간에 적합한 MAPPO 알고리즘(Gaussian 정책)을 구현하여 경찰 팀의 파라미터 공유 학습과 도망자의 개별 학습을 수행한다.
2. THE Continuous_Training_Pipeline SHALL Self_Play 방식으로 경찰 팀과 도망자를 동시에 학습시키며, 매 에피소드마다 양측의 정책을 업데이트한다.
3. THE Continuous_Training_Pipeline SHALL 학습 하이퍼파라미터(learning_rate, discount_factor, clip_epsilon, entropy_coefficient, max_episodes)를 설정 파일에서 로드한다.
4. THE Continuous_Training_Pipeline SHALL 매 checkpoint_interval 에피소드마다 모델 체크포인트를 저장하고, 학습 중단 시 마지막 체크포인트에서 재개할 수 있는 기능을 제공한다.
5. WHILE 학습이 진행되는 동안, THE Continuous_Training_Pipeline SHALL 매 에피소드 종료 시 경찰 승률, 도망자 탈출률, 평균 에피소드 길이, 평균 보상을 로그 파일에 기록한다.

### Requirement 11: 이산-연속 환경 전이 학습

**User Story:** 개발자로서, 이산 환경에서 학습된 정책 지식을 연속 환경으로 전이하고 싶다. 이를 통해 연속 환경의 학습 초기 수렴 속도를 향상시킬 수 있다.

#### Acceptance Criteria

1. THE Transfer_Module SHALL 이산 환경에서 학습된 MAPPO 모델의 가치 함수(Critic) 가중치를 연속 환경 모델의 Critic 네트워크 초기값으로 전이할 수 있는 인터페이스를 제공한다.
2. WHEN 전이 학습이 요청되면, THE Transfer_Module SHALL 이산 환경 체크포인트 파일 경로를 입력받아, 호환 가능한 레이어의 가중치를 연속 환경 모델에 복사하고, 차원이 불일치하는 레이어는 건너뛰며 경고를 출력한다.
3. THE Transfer_Module SHALL 전이 학습 적용 여부를 설정 파일의 옵션(transfer_from 경로)으로 지정할 수 있으며, 경로가 비어 있으면 전이 없이 처음부터 학습한다.
4. IF 지정된 이산 환경 체크포인트 파일이 존재하지 않거나 로드에 실패하면, THEN THE Transfer_Module SHALL 경고 메시지를 출력하고 전이 없이 처음부터 학습을 진행한다.

### Requirement 12: 연속 환경 설정 관리

**User Story:** 개발자로서, 연속 환경의 모든 파라미터를 설정 파일로 관리하고 싶다. 이를 통해 다양한 실험 조건을 코드 수정 없이 변경할 수 있다.

#### Acceptance Criteria

1. THE Configuration_Manager SHALL 연속 환경 전용 설정 파일(YAML)을 파싱하여 Road_Map 구조, 에이전트 파라미터, 학습 하이퍼파라미터를 로드한다.
2. THE Configuration_Manager SHALL 다음 연속 환경 설정 항목을 포함한다: map_width(양의 실수), map_height(양의 실수), num_police(3~20 정수), max_speed_police(양의 실수), max_speed_fugitive(양의 실수), capture_radius(양의 실수), dt(양의 실수), max_steps(양의 정수).
3. IF 설정 파일에 항목이 누락된 경우, THEN THE Configuration_Manager SHALL 해당 항목에 사전 정의된 기본값을 적용하고 누락된 항목명을 포함하는 경고 메시지를 출력한다.
4. IF 설정 항목의 값이 허용 범위를 벗어나는 경우(예: 음수 capture_radius, 0 이하의 dt), THEN THE Configuration_Manager SHALL 해당 항목명과 허용 범위를 포함하는 오류 메시지를 출력하고 프로그램을 종료한다.
5. THE Configuration_Manager SHALL 유효한 연속 환경 설정 파일을 파싱한 후 YAML로 직렬화하고 재파싱했을 때 동일한 설정 객체가 생성됨을 보장한다(round-trip 속성).

