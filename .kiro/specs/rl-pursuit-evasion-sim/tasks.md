# Implementation Plan: RL 추격-도주 시뮬레이션

## Overview

강화학습 기반 멀티 에이전트 추격-도주 시뮬레이션의 구현 계획이다. 의존성 순서에 따라 설정 관리 → 도로 네트워크 → 환경 구성 요소 → Gymnasium 환경 → 휴리스틱 에이전트 → 학습 파이프라인 → 평가/시각화 순으로 구현한다. Python 3.10+ 기반이며, hypothesis 라이브러리를 사용한 Property-Based Testing을 포함한다.

## Tasks

- [x] 1. 프로젝트 구조 및 설정 관리 구현
  - [x] 1.1 프로젝트 디렉토리 구조 및 의존성 설정
    - `pursuit_evasion_rl/` 디렉토리 구조 생성 (env/, training/, eval/, utils/, configs/)
    - 각 디렉토리에 `__init__.py` 생성
    - `pyproject.toml` 또는 `requirements.txt` 작성 (networkx, gymnasium, stable-baselines3, ray[rllib], matplotlib, pygame, PyYAML, hypothesis, pytest)
    - `configs/default.yaml` 기본 설정 파일 작성 (Design의 Data Models 섹션 참조)
    - _Requirements: 9.1, 9.2_

  - [x] 1.2 ConfigManager 및 SimulationConfig 구현
    - `utils/config_manager.py` 구현
    - `SimulationConfig` dataclass 정의 (모든 필드 기본값 포함)
    - `ConfigManager.load()`: YAML 파싱, 누락 항목 기본값 적용 + 경고, 문법 오류 시 행 번호 포함 에러 + 종료, 범위 초과 시 항목명/범위 에러 + 종료
    - `ConfigManager.validate()`: 모든 필드의 범위/타입 검증
    - `ConfigManager.serialize()`: SimulationConfig → YAML 문자열
    - `ConfigManager.round_trip()`: serialize 후 재파싱 동일성 확인
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 1.3 ConfigManager Property 테스트 작성
    - **Property 21: 설정 파일 round-trip** — 유효한 SimulationConfig를 YAML 직렬화 후 재파싱하면 원본과 동일한 객체 생성 확인
    - **Property 22: 설정 누락 시 기본값 적용** — 임의 항목 누락 시 기본값 적용 및 경고 메시지 출력 확인
    - **Property 23: 설정값 범위 검증** — 범위 밖 값에 대해 validate()가 오류 반환 확인
    - **Validates: Requirements 9.3, 9.5, 9.6**

- [x] 2. 도로 네트워크 구현
  - [x] 2.1 RoadNetwork 클래스 구현
    - `env/road_network.py` 구현
    - `generate()`: 설정 mode에 따라 고정/랜덤 그래프 생성
    - `_generate_random()`: erdos_renyi 기반 연결 그래프 생성, 최대 100회 재시도
    - `_generate_fixed()`: 설정 파일의 엣지 리스트 기반 고정 그래프 생성
    - `_identify_boundary_nodes()`: degree ≤ 2 AND closeness centrality 하위 25% 노드 식별
    - `get_neighbors()`, `shortest_path_length()`, `shortest_path()` 유틸리티 메서드
    - 파라미터 범위 검증 (num_nodes: 4~200, density: 0.0 < x ≤ 1.0)
    - 커스텀 예외 `NetworkGenerationError` 정의
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [ ]* 2.2 RoadNetwork Property 테스트 작성
    - **Property 1: 그래프 생성 불변량** — 유효 파라미터로 생성된 그래프가 무향 연결 그래프이며 노드 수 일치 확인
    - **Property 2: 고정 네트워크 멱등성** — 동일 설정으로 여러 번 생성 시 동일 그래프 확인
    - **Property 3: 경계 노드 조건 충족** — boundary 노드의 degree ≤ 2, closeness centrality 하위 25%, 최소 1개 존재 확인
    - **Property 4: 유효하지 않은 파라미터 거부** — 범위 밖 파라미터 시 예외 발생 확인
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9**

- [x] 3. 체크포인트 - 기반 모듈 검증
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. 환경 구성 요소 구현
  - [ ] 4.1 ObservationBuilder 구현
    - `env/observations.py` 구현
    - `build_observation_space()`: gymnasium.spaces.Dict 기반 에이전트별 관측 공간 정의
    - `get_observation()`: my_position, neighbors(패딩), other_positions, nearest_boundary_dist, current_step 계산
    - 제거된 에이전트 위치 -1 처리
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 4.2 ObservationBuilder Property 테스트 작성
    - **Property 8: 관측값 위치 정확성** — my_position과 other_positions가 실제 위치와 일치 확인
    - **Property 9: 인접 노드 관측 패딩** — neighbors 배열 길이 = max_degree, 빈 슬롯 -1 확인
    - **Property 10: 최단 경로 거리 관측 정확성** — nearest_boundary_dist와 networkx 계산값 일치 확인
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

  - [ ] 4.3 ActionHandler 구현
    - `env/actions.py` 구현
    - `build_action_space()`: Discrete(max_degree + 1) 행동 공간 정의 (인접 노드 인덱스 + stay)
    - `execute_action()`: 유효 행동 시 이동, 무효 행동/stay 시 현재 위치 유지
    - `get_action_mask()`: boolean 배열, stay 항상 True
    - `is_valid_action()`: 행동 유효성 검사
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 4.4 ActionHandler Property 테스트 작성
    - **Property 12: 행동 실행 정확성** — 유효 인접 노드 인덱스 선택 시 이동, stay/무효 시 위치 유지 확인
    - **Property 13: 행동 마스크 정확성** — 마스크 True 인덱스가 실제 이동 가능 노드, stay 항상 True 확인
    - **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**

  - [ ] 4.5 RewardCalculator 구현
    - `env/rewards.py` 구현
    - `compute_step_rewards()`: 종료 보상, shaping 보상, 협력 보상, 무효 행동 페널티 통합 계산
    - `_compute_shaping_reward_police()`: 도망자와의 거리 감소 비례 보상 (|값| ≤ 0.1)
    - `_compute_shaping_reward_fugitive()`: 경계 노드와의 거리 감소 비례 보상 (|값| ≤ 0.1)
    - `_compute_cooperation_bonus()`: 3방향 이상 포위 시 0.05
    - Terminal_Reward: 경찰 승리 (+1.0/-1.0), 도망자 승리 (-1.0/+1.0), 타임아웃 (-0.5)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [ ]* 4.6 RewardCalculator Property 테스트 작성
    - **Property 17: Shaping 보상 범위 제한** — 단일 스텝 Shaping 보상 |값| ≤ 0.1 확인
    - **Property 18: 협력 보상 조건** — 3방향 이상 포위 시 0.05 부여 확인
    - **Property 19: Terminal > Shaping 스케일 불변량** — 전체 에피소드에서 terminal > 누적 shaping 확인
    - **Property 20: 무효 행동 페널티** — 무효 행동 시 정확히 -0.01 페널티, 위치 불변 확인
    - **Validates: Requirements 6.4, 6.5, 6.6, 6.7, 6.8**

  - [ ] 4.7 TerminationChecker 구현
    - `env/termination.py` 구현
    - `check()`: (종료 원인, terminated, truncated) 반환
    - `_check_fugitive_escape()`: 도망자 boundary 노드 도달 확인
    - `_check_police_capture()`: 경찰-도망자 동일 노드 확인
    - `_check_police_surround()`: 도망자 인접 노드 전체 경찰 점유 확인
    - `_check_path_blocked()`: 도망자→경계노드 경로 완전 차단 확인
    - 우선순위: 도망자 승리 > 경찰 승리 > 타임아웃
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6_

  - [ ]* 4.8 TerminationChecker Property 테스트 작성
    - **Property 14: 도망자 탈출 판정** — Fugitive가 Boundary_Node 위치 시 도망자 승리 종료 확인
    - **Property 15: 경찰 포위 판정** — 도망자 인접 노드 전체 경찰 점유 시 경찰 승리 확인
    - **Property 16: 종료 조건 우선순위** — 복수 조건 동시 충족 시 우선순위 준수 확인
    - **Validates: Requirements 5.1, 5.3, 5.6**

- [ ] 5. 체크포인트 - 환경 구성 요소 검증
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Gymnasium 환경 통합 구현
  - [ ] 6.1 PursuitEnvironment 클래스 구현
    - `env/pursuit_env.py` 구현
    - RoadNetwork, ObservationBuilder, ActionHandler, RewardCalculator, TerminationChecker 통합
    - `__init__()`: 설정 로드, 컴포넌트 초기화, observation_space/action_space 정의
    - `reset()`: seed 기반 에이전트 랜덤 배치, 상태 초기화, (observations, info) 반환
    - `step()`: 전 에이전트 동시 행동 적용, 보상 계산, 종료 판정, 스텝 카운터 증가
    - `render()`: render_mode에 따른 시각화 호출
    - `get_action_masks()`: 에이전트별 행동 마스크 딕셔너리 반환
    - reset 전 step 호출 시 RuntimeError 발생
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 5.5, 5.7_

  - [ ]* 6.2 PursuitEnvironment Property 테스트 작성
    - **Property 5: Reset seed 재현성** — 동일 seed로 reset 두 번 호출 시 동일 초기 배치 확인
    - **Property 6: 에이전트 위치 불변량** — 모든 step에서 에이전트 위치가 유효 Node ID 확인
    - **Property 7: 에이전트 수 설정 반영** — num_police 설정값과 실제 에이전트 수 일치 확인
    - **Property 11: 스텝 카운터 정확성** — N번 step 후 current_step = N 확인
    - **Validates: Requirements 2.2, 2.5, 2.6, 3.5**

  - [ ]* 6.3 PursuitEnvironment 단위 테스트 작성
    - 기본 설정(3 Police, 1 Fugitive) 환경 생성 확인
    - 경찰 승리 시 Terminal_Reward 값 (+1.0, -1.0) 확인
    - 도망자 승리 시 Terminal_Reward 값 (+1.0, -1.0) 확인
    - 타임아웃 시 Terminal_Reward 값 (-0.5) 확인
    - Max_Steps 도달 시 truncated=True 확인
    - Gymnasium env_checker 통과 테스트 (Integration)
    - _Requirements: 2.1, 2.4, 6.1, 6.2, 6.3_

- [ ] 7. 체크포인트 - Gymnasium 환경 통합 검증
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. 휴리스틱 에이전트 구현
  - [ ] 8.1 HeuristicPoliceAgent 및 HeuristicFugitiveAgent 구현
    - `eval/heuristic_agents.py` 구현
    - `HeuristicPoliceAgent.act()`: networkx shortest_path로 도망자 방향 이동, 경로 미존재 시 stay
    - `HeuristicFugitiveAgent.act()`: 가장 가까운 boundary 노드 방향 이동, 동일 거리 시 ID 최소 선택, 경로 미존재 시 stay
    - 행동을 ActionHandler 인덱스로 변환하는 로직
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ]* 8.2 휴리스틱 에이전트 Property 테스트 작성
    - **Property 24: 휴리스틱 경찰 최단 경로 추적** — 경로 존재 시 shortest_path 두 번째 노드 인덱스 확인
    - **Property 25: 휴리스틱 도망자 탈출 경로 추적** — 가장 가까운 boundary(동일 거리 시 ID 최소) 방향 이동 확인
    - **Property 26: 휴리스틱 대전 에피소드 종료 보장** — Max_Steps 이내 종료 조건 도달 확인
    - **Validates: Requirements 10.1, 10.2, 10.5**

- [ ] 9. 학습 파이프라인 구현
  - [ ] 9.1 알고리즘 래퍼 구현
    - `training/algorithms.py` 구현
    - MAPPO 래퍼: stable-baselines3 또는 ray[rllib] 기반 멀티 에이전트 PPO
    - MADDPG 래퍼: 중앙 집중식 critic + 분산 actor 구조
    - 공통 인터페이스: `train_step()`, `get_action()`, `save()`, `load()`
    - _Requirements: 7.1, 7.7_

  - [ ] 9.2 Self-Play 매니저 구현
    - `training/self_play.py` 구현
    - 경찰 팀과 도망자의 동시 학습 관리
    - 에이전트 모델 업데이트 스케줄링
    - 상대 모델 풀(pool) 관리 (선택적)
    - _Requirements: 7.2_

  - [ ] 9.3 TrainingPipeline 구현
    - `training/train.py` 구현
    - 설정 파일 로드 및 알고리즘 선택 (MAPPO/MADDPG)
    - Self-Play 방식 학습 루프 구현
    - 에피소드별 보상, 승률, 길이 로깅
    - 체크포인트 간격별 모델 저장
    - 최대 에피소드 수 도달 시 학습 종료
    - 설정 파일 로드 실패 시 학습 미시작 + 오류 메시지
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.8_

  - [ ]* 9.4 학습 파이프라인 단위 테스트 작성
    - 알고리즘 선택 (MAPPO/MADDPG) 분기 확인
    - 설정 파일 미존재 시 학습 미시작 확인
    - 체크포인트 저장/로드 사이클 확인
    - 짧은 에피소드 수(10회)로 end-to-end 학습 실행 확인
    - _Requirements: 7.1, 7.4, 7.6_

- [ ] 10. 체크포인트 - 학습 파이프라인 검증
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. 평가 및 시각화 구현
  - [ ] 11.1 Evaluator 구현
    - `eval/evaluate.py` 구현
    - `evaluate()`: 모델 로드 → N 에피소드 실행 → 승률, 평균 보상, 평균 길이 산출
    - `compare_with_heuristic()`: RL 에이전트 vs 휴리스틱 에이전트 비교 평가, 비교표 출력
    - 모델 파일 미존재/로드 실패 시 오류 메시지 + 평가 중단
    - _Requirements: 8.1, 8.2, 8.3, 10.5, 10.6_

  - [ ] 11.2 Visualizer 구현
    - `eval/visualizer.py` 구현
    - `render_frame()`: matplotlib/pygame 기반 2D 그래프 렌더링 (경찰: 파랑, 도망자: 빨강, 경계: 초록)
    - `replay_episode()`: 에피소드 재생 (자동 500ms 간격 / 수동 모드)
    - render_backend 선택 (matplotlib 또는 pygame)
    - _Requirements: 8.4, 8.5, 8.6_

  - [ ]* 11.3 평가/시각화 통합 테스트 작성
    - 휴리스틱 에이전트 간 대전 평가 실행 및 통계 출력 확인
    - Visualizer render_frame 예외 없이 실행 확인 (SMOKE)
    - _Requirements: 8.1, 8.4, 10.5, 10.6_

- [ ] 12. 최종 체크포인트 - 전체 시스템 검증
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- `*` 표시된 태스크는 선택 사항이며, 빠른 MVP 구현을 위해 건너뛸 수 있습니다.
- 각 태스크는 구체적인 Requirements를 참조하여 추적 가능합니다.
- Property 테스트는 `hypothesis` 라이브러리를 사용하며, 각 속성당 최소 100회 이상 실행합니다.
- 단위 테스트는 `pytest`를 사용하며, 핵심 로직 90% 이상 라인 커버리지를 목표로 합니다.
- 학습 파이프라인 테스트는 짧은 에피소드 수로 실행하여 기능 검증에 집중합니다.
- 시각화 테스트는 SMOKE 수준(예외 없이 실행 확인)으로 진행합니다.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["1.3", "2.1"] },
    { "id": 3, "tasks": ["2.2", "4.1", "4.3", "4.5", "4.7"] },
    { "id": 4, "tasks": ["4.2", "4.4", "4.6", "4.8"] },
    { "id": 5, "tasks": ["6.1"] },
    { "id": 6, "tasks": ["6.2", "6.3", "8.1"] },
    { "id": 7, "tasks": ["8.2", "9.1", "9.2"] },
    { "id": 8, "tasks": ["9.3"] },
    { "id": 9, "tasks": ["9.4", "11.1", "11.2"] },
    { "id": 10, "tasks": ["11.3"] }
  ]
}
```
