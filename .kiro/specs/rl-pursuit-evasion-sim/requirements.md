# Requirements Document

## Introduction

강화학습(Reinforcement Learning) 기반 멀티 에이전트 추격-도주 시뮬레이션 시스템이다. 추상화된 그래프 구조의 도로 네트워크 위에서 경찰 차량 에이전트(복수)와 도망자 차량 에이전트(1대)가 각자의 승리 조건을 달성하기 위한 전략을 자율적으로 학습한다. 경찰은 도망자를 포위하거나 탈출 경로를 차단해야 하며, 도망자는 외곽 경계 노드에 도달하여 탈출해야 한다.

## Glossary

- **Road_Network**: networkx 기반 그래프 구조로 표현된 도로 네트워크. 노드(교차로)와 엣지(도로 구간)로 구성된다.
- **Node**: 도로 네트워크 그래프의 꼭짓점. 교차로 또는 도로 분기점을 나타낸다.
- **Edge**: 두 노드를 연결하는 도로 구간. 에이전트가 이동할 수 있는 경로이다.
- **Boundary_Node**: 도로 네트워크의 외곽 경계에 위치한 노드. 도망자의 탈출 목표 지점이다.
- **Police_Agent**: 도망자를 포위하거나 탈출 경로를 차단하는 것을 목표로 하는 경찰 차량 에이전트이다.
- **Fugitive_Agent**: 경찰의 포위를 피해 Boundary_Node에 도달하여 탈출하는 것을 목표로 하는 도망자 차량 에이전트이다.
- **Pursuit_Environment**: Gymnasium 인터페이스를 따르는 멀티 에이전트 추격-도주 시뮬레이션 환경이다.
- **Episode**: 시뮬레이션의 한 에피소드. 초기 상태에서 시작하여 종료 조건이 달성될 때까지의 전체 과정이다.
- **Step**: Episode 내에서 모든 에이전트가 한 번씩 행동을 취하는 단위 시간이다.
- **Observation_Space**: 에이전트가 관측할 수 있는 상태 정보의 집합이다.
- **Action_Space**: 에이전트가 선택할 수 있는 행동의 집합이다.
- **Self_Play**: 경찰 팀과 도망자가 서로 상대하며 동시에 학습하는 방식이다.
- **Reward_Shaping**: 최종 승패 보상 외에 중간 단계에서 학습을 유도하는 추가 보상 신호이다.
- **Terminal_Reward**: Episode 종료 시 승패에 따라 부여되는 최종 보상이다.
- **MAPPO**: Multi-Agent Proximal Policy Optimization. 멀티 에이전트 강화학습 알고리즘이다.
- **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient. 멀티 에이전트 강화학습 알고리즘이다.
- **Max_Steps**: 하나의 Episode에서 허용되는 최대 Step 수이다.

## Requirements

### Requirement 1: 도로 네트워크 생성

**User Story:** 개발자로서, 추상화된 그래프 구조의 도로 네트워크를 생성하고 싶다. 이를 통해 시뮬레이션 환경의 맵을 유연하게 정의할 수 있다.

#### Acceptance Criteria

1. THE Road_Network SHALL 노드와 엣지로 구성된 무향 연결 그래프를 생성한다.
2. WHEN 고정 네트워크 모드가 선택되면, THE Road_Network SHALL 설정 파일에 정의된 구조로 동일한 그래프를 생성한다.
3. WHEN 랜덤 생성 모드가 선택되면, THE Road_Network SHALL 지정된 노드 수(최소 4개, 최대 200개)와 연결 밀도(0.0 초과 1.0 이하)에 따라 무작위 연결 그래프를 생성한다.
4. THE Road_Network SHALL 그래프 내에서 degree가 1 또는 2이며 그래프의 중심성(closeness centrality)이 하위 25%에 해당하는 노드를 Boundary_Node로 지정한다.
5. THE Road_Network SHALL 최소 1개 이상의 Boundary_Node를 포함한다.
6. THE Road_Network SHALL networkx 라이브러리를 사용하여 그래프를 표현한다.
7. WHEN 랜덤 생성된 그래프가 연결 그래프가 아닌 경우, THE Road_Network SHALL 최대 100회까지 연결 그래프가 될 때까지 재생성한다.
8. IF 최대 재생성 시도 횟수(100회) 이내에 연결 그래프가 생성되지 않으면, THEN THE Road_Network SHALL 생성 실패를 나타내는 예외를 발생시키고 그래프를 생성하지 않는다.
9. IF 랜덤 생성 모드에서 노드 수 또는 연결 밀도가 유효 범위를 벗어나면, THEN THE Road_Network SHALL 유효하지 않은 파라미터를 나타내는 예외를 발생시키고 그래프를 생성하지 않는다.

### Requirement 2: 시뮬레이션 환경 구현

**User Story:** 개발자로서, Gymnasium 인터페이스를 따르는 멀티 에이전트 환경을 구현하고 싶다. 이를 통해 표준 강화학습 라이브러리와 호환되는 학습이 가능하다.

#### Acceptance Criteria

1. THE Pursuit_Environment SHALL Gymnasium 환경 인터페이스(reset, step, render)를 구현하며, observation_space와 action_space 속성을 정의하여 Gymnasium API 호환성 검증(gymnasium.utils.env_checker)을 통과한다.
2. WHEN reset이 호출되면, THE Pursuit_Environment SHALL 모든 에이전트를 Road_Network의 유효한 Node 위에 랜덤 배치하고, seed 파라미터가 제공된 경우 동일 seed에 대해 동일한 초기 배치를 보장하며, 각 에이전트의 관측값을 포함한 딕셔너리를 반환한다.
3. WHEN step이 호출되면, THE Pursuit_Environment SHALL 모든 에이전트의 행동을 동일 타임스텝 내에서 적용하고, 각 에이전트별 관측값, 보상, 종료 여부(terminated, truncated)를 딕셔너리 형태로 반환한다.
4. THE Pursuit_Environment SHALL 기본 3대의 Police_Agent와 1대의 Fugitive_Agent를 포함한다.
5. THE Pursuit_Environment SHALL Police_Agent의 수를 설정 파일을 통해 최소 3대에서 최대 20대까지 조정할 수 있도록 한다.
6. THE Pursuit_Environment SHALL 각 에이전트의 위치를 Road_Network의 Node 위에서만 관리한다.
7. WHEN Police_Agent 중 하나 이상이 Fugitive_Agent와 동일한 Node에 위치하면, THE Pursuit_Environment SHALL 에피소드를 종료(terminated=True)로 판정한다.
8. IF step 호출 시 에이전트의 행동이 action_space 범위를 벗어나면, THEN THE Pursuit_Environment SHALL 해당 행동을 무시하고 에이전트를 현재 위치에 유지한다.
9. IF 에피소드가 최대 타임스텝 수(기본값: 500스텝)에 도달하면, THEN THE Pursuit_Environment SHALL 에피소드를 절단(truncated=True)으로 종료한다.

### Requirement 3: 관측 공간 정의

**User Story:** 개발자로서, 에이전트가 의사결정에 필요한 정보를 관측할 수 있도록 하고 싶다. 이를 통해 에이전트가 환경 상태에 기반한 전략적 행동을 학습할 수 있다.

#### Acceptance Criteria

1. THE Pursuit_Environment SHALL 각 에이전트에게 자신의 현재 위치를 노드 ID(정수값)로 관측값에 포함하여 제공한다.
2. THE Pursuit_Environment SHALL 각 에이전트에게 자신의 인접 노드 목록을 고정 길이 배열로 제공하며, 실제 인접 노드 수가 최대 차수보다 적을 경우 나머지를 -1로 패딩한다.
3. THE Pursuit_Environment SHALL 각 에이전트에게 모든 다른 에이전트의 현재 위치를 노드 ID(정수값) 배열로 관측값에 포함하여 제공한다.
4. THE Pursuit_Environment SHALL 각 에이전트에게 가장 가까운 Boundary_Node까지의 최단 경로 거리를 홉(hop) 수로 관측값에 포함하여 제공한다.
5. THE Pursuit_Environment SHALL 각 에이전트에게 현재 Episode의 경과 Step 수를 0부터 시작하는 정수값으로 관측값에 포함하여 제공한다.
6. IF 에이전트가 환경에서 제거된 상태라면, THEN THE Pursuit_Environment SHALL 해당 에이전트의 위치를 다른 에이전트의 관측값에서 -1로 표시한다.
7. THE Pursuit_Environment SHALL 각 에이전트의 관측값을 매 Step 시작 시 행동 선택 전에 갱신하여 제공한다.

### Requirement 4: 행동 공간 정의

**User Story:** 개발자로서, 에이전트의 행동 공간을 이산적으로 정의하고 싶다. 이를 통해 그래프 구조에 적합한 행동 선택이 가능하다.

#### Acceptance Criteria

1. THE Pursuit_Environment SHALL 각 에이전트의 행동 공간을 최대 인접 노드 수 기반의 고정 크기 이산 공간으로 정의하며, 인접 노드 선택(인덱스 0부터 최대 인접 노드 수 - 1)과 정지(stay) 행동(별도 인덱스)으로 구성한다.
2. WHEN 에이전트가 인접 노드를 선택하면, THE Pursuit_Environment SHALL 해당 에이전트를 선택된 인접 노드로 이동시킨다.
3. WHEN 에이전트가 정지 행동을 선택하면, THE Pursuit_Environment SHALL 해당 에이전트를 현재 노드에 유지한다.
4. IF 에이전트가 유효하지 않은 행동(현재 노드와 연결되지 않은 노드에 해당하는 인덱스)을 선택하면, THEN THE Pursuit_Environment SHALL 해당 에이전트를 현재 노드에 유지하고 해당 스텝의 행동을 무효 처리하여 정지(stay)와 동일한 결과를 적용한다.
5. THE Pursuit_Environment SHALL 각 스텝마다 에이전트별로 현재 노드 기준 유효한 행동을 나타내는 행동 마스크(boolean 배열, 크기는 행동 공간 크기와 동일)를 제공한다.
6. THE Pursuit_Environment SHALL 행동 마스크에서 유효한 행동은 True, 유효하지 않은 행동은 False로 표시하며, 정지(stay) 행동은 항상 True로 표시한다.

### Requirement 5: 승리 조건 및 에피소드 종료

**User Story:** 개발자로서, 명확한 승리 조건과 종료 조건을 정의하고 싶다. 이를 통해 에이전트가 목표 지향적 학습을 수행할 수 있다.

#### Acceptance Criteria

1. WHEN Fugitive_Agent의 모든 인접 이동 가능 노드가 Police_Agent에 의해 점유되면, THE Pursuit_Environment SHALL 경찰 승리로 Episode를 종료하고, done 플래그를 True로 설정한다.
2. WHEN Fugitive_Agent가 이동할 수 있는 모든 Boundary_Node까지의 경로가 Police_Agent의 노드 점유에 의해 완전히 차단되면, THE Pursuit_Environment SHALL 경찰 승리로 Episode를 종료하고, done 플래그를 True로 설정한다.
3. WHEN Fugitive_Agent가 Boundary_Node에 도달하면, THE Pursuit_Environment SHALL 도망자 승리로 Episode를 종료하고, done 플래그를 True로 설정한다.
4. WHEN 현재 Step 수가 Max_Steps를 초과하면, THE Pursuit_Environment SHALL 타임아웃으로 Episode를 종료하고, done 플래그를 True로 설정한다.
5. WHEN Episode가 종료되면, THE Pursuit_Environment SHALL 종료 원인(경찰 승리, 도망자 승리, 타임아웃)과 각 에이전트별 보상(reward) 값을 info 딕셔너리에 포함하여 반환한다.
6. IF 동일 Step에서 복수의 종료 조건이 동시에 충족되면, THEN THE Pursuit_Environment SHALL 도망자 승리 > 경찰 승리 > 타임아웃 순의 우선순위에 따라 하나의 종료 원인만 판정한다.
7. WHEN Episode가 경찰 승리로 종료되면, THE Pursuit_Environment SHALL Police_Agent에게 양의 보상을, Fugitive_Agent에게 음의 보상을 반환하고, 타임아웃 종료 시에는 양측 모두에게 음의 보상을 반환한다.

### Requirement 6: 보상 설계

**User Story:** 개발자로서, 에이전트의 학습을 효과적으로 유도하는 보상 구조를 설계하고 싶다. 이를 통해 원하는 전략적 행동이 자연스럽게 학습된다.

#### Acceptance Criteria

1. WHEN 경찰 승리로 Episode가 종료되면, THE Pursuit_Environment SHALL Police_Agent에게 +1.0의 Terminal_Reward를 부여하고 Fugitive_Agent에게 -1.0의 Terminal_Reward를 부여한다.
2. WHEN 도망자 승리로 Episode가 종료되면, THE Pursuit_Environment SHALL Fugitive_Agent에게 +1.0의 Terminal_Reward를 부여하고 Police_Agent에게 -1.0의 Terminal_Reward를 부여한다.
3. WHEN 타임아웃으로 Episode가 종료되면, THE Pursuit_Environment SHALL 모든 에이전트에게 -0.5의 Terminal_Reward를 부여한다.
4. WHILE Episode가 진행 중일 때, THE Pursuit_Environment SHALL 매 타임스텝마다 Police_Agent에게 이전 타임스텝 대비 Fugitive_Agent와의 최단 경로 거리 감소량에 비례하는 Reward_Shaping 보상을 부여하되, 단일 스텝 Shaping 보상의 절대값은 0.1을 초과하지 않는다.
5. WHILE Episode가 진행 중일 때, THE Pursuit_Environment SHALL 매 타임스텝마다 Fugitive_Agent에게 이전 타임스텝 대비 가장 가까운 Boundary_Node와의 최단 경로 거리 감소량에 비례하는 Reward_Shaping 보상을 부여하되, 단일 스텝 Shaping 보상의 절대값은 0.1을 초과하지 않는다.
6. WHILE Episode가 진행 중일 때, THE Pursuit_Environment SHALL 매 타임스텝마다 Police_Agent들이 Fugitive_Agent를 기준으로 3개 이상의 서로 다른 인접 방향(Fugitive_Agent에 연결된 간선 방향 기준)을 점유하고 있을 경우 각 Police_Agent에게 0.05의 협력 보상을 추가로 부여한다.
7. THE Pursuit_Environment SHALL Terminal_Reward의 절대값이 Reward_Shaping 보상 누적 합의 절대값보다 항상 크도록 보상 스케일을 유지한다.
8. IF 에이전트가 유효하지 않은 행동(이동 불가능한 노드로의 이동 시도)을 선택하면, THEN THE Pursuit_Environment SHALL 해당 에이전트에게 -0.01의 페널티를 부여하고 해당 에이전트의 위치를 변경하지 않는다.

### Requirement 7: 강화학습 학습 파이프라인

**User Story:** 개발자로서, 멀티 에이전트 강화학습 알고리즘을 적용하여 에이전트를 학습시키고 싶다. 이를 통해 경찰과 도망자 모두 효과적인 전략을 습득한다.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL 설정 파일(configs/default.yaml)에 지정된 알고리즘 파라미터에 따라 MAPPO 또는 MADDPG 알고리즘 중 하나를 선택하여 학습을 수행한다.
2. THE Training_Pipeline SHALL Self_Play 방식으로 동일 에피소드 환경 내에서 Police_Agent와 Fugitive_Agent를 상호 대전시키며 동시에 학습시킨다.
3. THE Training_Pipeline SHALL 학습 하이퍼파라미터를 설정 파일(configs/default.yaml)에서 로드한다.
4. IF 설정 파일(configs/default.yaml)이 존재하지 않거나 파싱에 실패하면, THEN THE Training_Pipeline SHALL 학습을 시작하지 않고 오류 원인을 나타내는 메시지를 출력한다.
5. WHILE 학습이 진행 중인 동안, THE Training_Pipeline SHALL 매 에피소드 종료 시 에피소드 보상, 승률, 에피소드 길이를 로그 파일에 기록한다.
6. THE Training_Pipeline SHALL 설정 파일에 지정된 에피소드 간격(미지정 시 기본값 100 에피소드)마다 학습된 모델의 체크포인트를 저장한다.
7. THE Training_Pipeline SHALL stable-baselines3 또는 ray[rllib] 라이브러리를 사용한다.
8. IF 설정 파일에 지정된 최대 에피소드 수에 도달하면, THEN THE Training_Pipeline SHALL 학습을 종료하고 최종 체크포인트를 저장한다.

### Requirement 8: 평가 및 시각화

**User Story:** 개발자로서, 학습된 에이전트의 성능을 평가하고 시각적으로 확인하고 싶다. 이를 통해 학습 효과를 분석하고 전략을 이해할 수 있다.

#### Acceptance Criteria

1. WHEN 평가 실행이 요청되면, THE Evaluator SHALL 학습된 모델을 로드하여 지정된 수(기본 100, 최소 1, 최대 10,000)의 Episode를 실행하고 승률, 평균 누적 보상, 평균 에피소드 길이를 포함한 통계를 출력한다.
2. IF 지정된 모델 파일이 존재하지 않거나 로드에 실패하면, THEN THE Evaluator SHALL 오류 메시지를 출력하고 평가를 중단한다.
3. THE Evaluator SHALL 학습된 에이전트와 휴리스틱 에이전트(최단 경로 추격, 랜덤 이동)를 동일한 에피소드 수 및 환경 조건에서 실행하여 각 에이전트별 승률을 산출하고 비교표 형태로 출력한다.
4. THE Visualizer SHALL 도로 네트워크를 노드와 엣지로 구성된 2D 그래프로 렌더링하고, 경찰 에이전트와 도망자 에이전트의 위치를 서로 다른 색상으로 구분하여 표시한다.
5. THE Visualizer SHALL Episode의 진행 과정을 스텝 단위로 재생하되, 자동 재생(기본 프레임 간격 500ms) 및 단계별 수동 전진 기능을 제공한다.
6. THE Visualizer SHALL matplotlib 또는 pygame 라이브러리를 사용하여 렌더링한다.

### Requirement 9: 설정 파일 관리

**User Story:** 개발자로서, 시뮬레이션의 주요 파라미터를 설정 파일로 관리하고 싶다. 이를 통해 코드 수정 없이 실험 조건을 변경할 수 있다.

#### Acceptance Criteria

1. THE Configuration_Manager SHALL YAML 형식의 설정 파일을 파싱하여 시뮬레이션 파라미터를 로드한다.
2. THE Configuration_Manager SHALL 다음 설정 항목을 포함한다: 도로 네트워크 크기(grid_size, 양의 정수), 에이전트 수(num_agents, 1 이상의 정수), Max_Steps(max_steps, 1 이상의 정수), 학습 하이퍼파라미터(learning_rate, discount_factor, epsilon, batch_size, replay_buffer_size).
3. IF 설정 파일에 항목이 누락된 경우, THEN THE Configuration_Manager SHALL 해당 항목에 대해 사전 정의된 기본값을 적용하고 누락된 항목명을 포함하는 경고 메시지를 표준 출력 또는 로그로 출력한다.
4. IF 설정 파일이 YAML 문법에 맞지 않는 경우, THEN THE Configuration_Manager SHALL 파싱 실패 위치(행 번호)와 원인을 포함하는 오류 메시지를 출력하고 프로그램을 종료한다.
5. THE Configuration_Manager SHALL 유효한 설정 파일을 파싱한 후 다시 YAML로 직렬화하고 재파싱했을 때 동일한 설정 객체가 생성됨을 보장한다(round-trip 속성).
6. IF 설정 항목의 값이 허용 범위를 벗어나는 경우(예: 음수 에이전트 수, 0 이하의 learning_rate), THEN THE Configuration_Manager SHALL 해당 항목명과 허용 범위를 포함하는 오류 메시지를 출력하고 프로그램을 종료한다.

### Requirement 10: 휴리스틱 에이전트

**User Story:** 개발자로서, RL 학습 전에 환경을 검증하기 위한 휴리스틱 에이전트를 구현하고 싶다. 이를 통해 환경의 정상 작동을 확인하고 기준선을 설정할 수 있다.

#### Acceptance Criteria

1. THE Heuristic_Police_Agent SHALL 각 Step에서 networkx의 shortest_path를 사용하여 Fugitive_Agent까지의 최단 경로 상 다음 노드로 이동하며, 최단 경로가 복수일 경우 networkx가 반환하는 첫 번째 경로를 따른다.
2. THE Heuristic_Fugitive_Agent SHALL 각 Step에서 networkx의 shortest_path를 사용하여 가장 가까운 Boundary_Node까지의 최단 경로 상 다음 노드로 이동하며, 동일 거리의 Boundary_Node가 복수일 경우 노드 ID가 가장 작은 것을 선택한다.
3. IF Heuristic_Police_Agent에서 Fugitive_Agent까지의 경로가 존재하지 않으면, THEN THE Heuristic_Police_Agent SHALL 정지(stay) 행동을 선택한다.
4. IF Heuristic_Fugitive_Agent에서 모든 Boundary_Node까지의 경로가 존재하지 않으면, THEN THE Heuristic_Fugitive_Agent SHALL 정지(stay) 행동을 선택한다.
5. WHEN 휴리스틱 에이전트 간 대전으로 환경 검증을 수행할 때, THE Pursuit_Environment SHALL 모든 Episode가 Max_Steps 이내에 종료 조건(경찰 승리, 도망자 승리, 타임아웃) 중 하나에 도달하여 종료되며, 각 Episode의 종료 원인과 소요 Step 수를 기록한다.
6. WHEN 휴리스틱 에이전트 간 대전이 지정된 Episode 수만큼 완료되면, THE Pursuit_Environment SHALL 경찰 승률, 도망자 승률, 평균 Episode 길이를 기준선 통계로 출력한다.
