# Implementation Plan: Continuous Pursuit Environment

## Overview

기존 이산 그래프 기반 추격-도주 RL 시뮬레이션을 2단계로 확장한다. Phase 1에서 커리큘럼 학습 매니저와 장기 학습 파이프라인을 구현하고, Phase 2에서 연속 2D 도로 환경, Gaussian MAPPO, 전이 학습 모듈을 구현한다. 구현 언어는 Python이며, 테스트에는 pytest와 hypothesis를 사용한다.

## Tasks

- [ ] 1. Phase 1: 커리큘럼 학습 및 장기 학습 파이프라인
  - [ ] 1.1 CurriculumManager 구현
    - `pursuit_evasion_rl/curriculum/__init__.py` 생성
    - `pursuit_evasion_rl/curriculum/curriculum_manager.py`에 `CurriculumLevel` 데이터클래스와 `CurriculumManager` 클래스 구현
    - 난이도 레벨 로드, 승률 추적, 승급 판정(`should_promote`), 레벨 전환(`promote`), 상태 저장/복원 기능 구현
    - 최소 3단계 레벨 검증, 최종 레벨 도달 시 완료 보고 로직 포함
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.2 CurriculumManager property 테스트 작성
    - `tests/test_curriculum_manager.py` 생성
    - **Property 1: 커리큘럼 승률 임계값 도달 시 자동 승급**
    - **Property 2: 난이도 전환 시 모델 가중치 보존**
    - **Validates: Requirements 1.2, 1.3**

  - [ ] 1.3 LongTrainPipeline 구현
    - `pursuit_evasion_rl/training/long_train_pipeline.py`에 `LongTrainPipeline` 클래스 구현
    - 최대 에피소드 학습 루프, 체크포인트 저장/로드, 학습 재개(resume) 기능 구현
    - 에피소드별 통계 추적(승률, 평균 길이, 평균 보상), 로깅 간격 지원
    - CurriculumManager 연동(선택적), Self-play 양팀 동시 업데이트 지원
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 1.4 체크포인트 라운드트립 property 테스트 작성
    - `tests/test_checkpoint_roundtrip.py` 생성
    - **Property 3: 체크포인트 라운드트립**
    - **Validates: Requirements 2.2, 2.3, 10.4**

  - [ ]* 1.5 CurriculumManager 및 LongTrainPipeline 단위 테스트 작성
    - 최종 레벨 도달 시 체크포인트 저장 확인 테스트
    - Self-play 양팀 업데이트 확인 테스트
    - _Requirements: 1.5, 2.5_

- [ ] 2. Checkpoint - Phase 1 검증
  - Phase 1 모든 테스트 통과 확인, 문제 발생 시 사용자에게 질의.

- [ ] 3. Phase 2: 연속 2D 도로 맵 구현
  - [ ] 3.1 RoadMap 모듈 구현
    - `pursuit_evasion_rl/continuous_env/__init__.py` 생성
    - `pursuit_evasion_rl/continuous_env/road_map.py`에 `RoadSegment`, `Intersection`, `RoadMap` 클래스 구현
    - 직선/베지어 곡선 세그먼트, 교차점 자동 식별, 경계 출구 판별, 연결성 검증 구현
    - 파일 로드(`load_from_file`), 프로그래밍 생성(grid, radial, random) 메서드 구현
    - `is_on_road`, `nearest_point_on_road`, `get_segment_at` 공간 쿼리 구현
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 3.2 RoadMap property 테스트 작성
    - `tests/test_road_map.py` 업데이트 또는 재작성
    - **Property 4: 도로 세그먼트 기하학적 정합성**
    - **Property 5: 교차점 자동 식별 완전성**
    - **Property 6: 도로 맵 직렬화 라운드트립**
    - **Property 7: 생성된 도로 맵의 연결성 보장**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6**

- [ ] 4. Phase 2: 에이전트 이동 모델 구현
  - [ ] 4.1 AgentModel 구현
    - `pursuit_evasion_rl/continuous_env/agent_model.py`에 `AgentState` 데이터클래스 및 이동 물리 로직 구현
    - 위치 갱신 공식(`position + speed * [cos(heading), sin(heading)] * dt`), 도로 위 제한, 클램핑, 방향 정규화 구현
    - 속도 상한(speed_limit) 준수, 교차점 방향 전환 로직 포함
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 4.2 에이전트 이동 property 테스트 작성
    - `tests/test_agent_movement.py` 생성
    - **Property 8: 이동 후 도로 위 위치 불변성**
    - **Property 9: 속도 상한 준수**
    - **Property 10: 위치 갱신 공식 정확성**
    - **Property 11: 방향 정규화 (모듈로 2π)**
    - **Validates: Requirements 4.2, 4.3, 4.5, 4.6, 5.3, 5.4**

- [ ] 5. Phase 2: 관측 공간 및 행동 공간 구현
  - [ ] 5.1 Observations 모듈 구현
    - `pursuit_evasion_rl/continuous_env/observations.py`에 연속 관측 벡터 빌더 구현
    - 자기 위치/속도/방향, 도망자 위치(경찰만), 다른 경찰 위치(패딩), 경계 거리, 최근접 출구 거리 포함
    - 고정 차원(51) 관측 벡터, float32 배열, gymnasium.spaces.Box 정의
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 5.2 관측 공간 property 테스트 작성
    - `tests/test_observations.py` 생성
    - **Property 12: 관측 벡터 정확성**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.7**

  - [ ] 5.3 행동 공간 정의
    - `observations.py`와 함께 또는 `continuous_pursuit_env.py` 내에서 행동 공간 정의
    - gymnasium.spaces.Box: [0, max_speed] × [0, 2π], 속도 클리핑, 방향 정규화 포함
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 6. Phase 2: 보상 시스템 구현
  - [ ] 6.1 Rewards 모듈 구현
    - `pursuit_evasion_rl/continuous_env/rewards.py`에 연속 보상 계산기 구현
    - 종료 보상(체포 +1/-1, 탈출 +1/-1, 타임아웃 -0.5), shaping 보상(거리 감소, 절대값 ≤ 0.1)
    - 포위 협력 보상(3대 이상, 120도 간격, +0.05), 오프로드 페널티(-0.02) 구현
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 6.2 보상 시스템 property 테스트 작성
    - `tests/test_rewards.py` 생성
    - **Property 15: Shaping 보상 상한 제한**
    - **Property 16: 포위 협력 보상 조건**
    - **Validates: Requirements 8.4, 8.5, 8.6**

- [ ] 7. Phase 2: ContinuousPursuitEnv 통합 구현
  - [ ] 7.1 ContinuousPursuitEnv 클래스 구현
    - `pursuit_evasion_rl/continuous_env/continuous_pursuit_env.py`에 `ContinuousPursuitEnv(gymnasium.Env)` 구현
    - `reset()`, `step()`, `render()` 메서드 및 `observation_space`, `action_space` 속성 구현
    - RoadMap, AgentModel, Observations, Rewards 모듈 통합
    - 종료 조건 판정(체포, 탈출, 타임아웃), 우선순위 적용, info 딕셔너리 반환
    - seed 기반 재현 가능한 초기 배치, reset() 없이 step() 호출 시 RuntimeError
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 7.2 종료 조건 property 테스트 작성
    - `tests/test_termination.py` 생성
    - **Property 13: 종료 조건 정확 판정**
    - **Property 14: 종료 조건 우선순위**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

  - [ ]* 7.3 Gymnasium API 호환성 테스트 작성
    - `tests/test_gymnasium_api.py` 생성
    - **Property 17: Gymnasium API 구조 정합성**
    - gymnasium.utils.env_checker 통과 확인 단위 테스트
    - RuntimeError 발생 테스트 (reset 없이 step 호출)
    - **Validates: Requirements 9.2, 9.3, 9.5, 9.6**

- [ ] 8. Checkpoint - Phase 2 환경 검증
  - 연속 환경 모듈 전체 테스트 통과 확인, 문제 발생 시 사용자에게 질의.

- [ ] 9. Phase 2: GaussianMAPPO 알고리즘 구현
  - [ ] 9.1 GaussianMAPPO 알고리즘 구현
    - `pursuit_evasion_rl/training/gaussian_mappo.py`에 `GaussianMLPActor`, `GaussianMAPPOAlgorithm` 구현
    - Gaussian 정책 네트워크(평균 + 로그 표준편차 출력), PPO 클리핑 업데이트
    - 경찰 팀 파라미터 공유, 도망자 개별 학습, 행동 공간 범위 내 출력 보장
    - get_action, get_log_prob, train_step, save/load 메서드 구현
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ]* 9.2 GaussianMAPPO property 테스트 작성
    - `tests/test_gaussian_mappo.py` 생성
    - **Property 18: Gaussian 정책 출력 유효 범위**
    - **Validates: Requirements 10.1**

- [ ] 10. Phase 2: 전이 학습 모듈 구현
  - [ ] 10.1 TransferModule 구현
    - `pursuit_evasion_rl/transfer/__init__.py` 생성
    - `pursuit_evasion_rl/transfer/transfer_module.py`에 `TransferModule` 클래스 구현
    - Critic 가중치 전이(차원 일치 레이어만), 전이 활성화 여부 판별, 소스 파일 미존재 시 경고 후 skip
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 10.2 전이 학습 property 테스트 작성
    - `tests/test_transfer_module.py` 생성
    - **Property 19: 전이 학습 호환 레이어 가중치 일치**
    - **Validates: Requirements 11.1, 11.2**

- [ ] 11. Phase 2: 설정 파일 및 설정 관리 구현
  - [ ] 11.1 연속 환경 설정 관리자 확장 및 설정 파일 생성
    - `configs/continuous_config.yaml` 생성 (연속 환경, 도로 맵, 학습, 전이 학습 설정 포함)
    - `pursuit_evasion_rl/utils/config_manager.py` 확장하여 연속 환경 설정 파싱, 기본값 적용, 범위 검증 로직 추가
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 11.2 설정 관리 property 테스트 작성
    - `tests/test_continuous_config.py` 생성
    - **Property 20: 설정 파일 라운드트립**
    - **Property 21: 설정 범위 검증 거부**
    - **Validates: Requirements 12.4, 12.5**

- [ ] 12. Phase 2: 전체 통합 및 파이프라인 연결
  - [ ] 12.1 연속 환경 학습 파이프라인 통합
    - `LongTrainPipeline`이 `ContinuousPursuitEnv` + `GaussianMAPPOAlgorithm`과 연동되도록 통합
    - `TransferModule`을 학습 시작 시 호출하여 초기 가중치 전이 수행
    - Self-play 양팀 동시 학습 플로우 완성
    - _Requirements: 10.2, 10.4, 10.5, 11.1_

  - [ ]* 12.2 통합 테스트 작성
    - 전체 에피소드 시뮬레이션 테스트 (reset → 다수 step → 종료)
    - 전이 학습 파이프라인 테스트 (이산 체크포인트 → 연속 모델 초기화)
    - _Requirements: 9.2, 9.3, 10.4, 11.1_

- [ ] 13. Final Checkpoint - 전체 테스트 통과 확인
  - 모든 테스트 통과 확인, 문제 발생 시 사용자에게 질의.

## Notes

- `*` 표시된 태스크는 선택적이며 빠른 MVP를 위해 건너뛸 수 있습니다.
- 각 태스크는 추적 가능성을 위해 특정 요구사항을 참조합니다.
- Checkpoint 태스크는 점진적 검증을 보장합니다.
- Property 테스트는 설계 문서의 보편적 정확성 속성을 검증합니다.
- 단위 테스트는 특정 예시와 엣지 케이스를 검증합니다.
- 구현 순서: Phase 1(커리큘럼 + 장기 학습) → Phase 2(연속 환경) 순서를 따릅니다.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4", "1.5"] },
    { "id": 3, "tasks": ["3.1"] },
    { "id": 4, "tasks": ["3.2", "4.1"] },
    { "id": 5, "tasks": ["4.2", "5.1", "5.3", "6.1"] },
    { "id": 6, "tasks": ["5.2", "6.2", "7.1"] },
    { "id": 7, "tasks": ["7.2", "7.3", "9.1"] },
    { "id": 8, "tasks": ["9.2", "10.1", "11.1"] },
    { "id": 9, "tasks": ["10.2", "11.2", "12.1"] },
    { "id": 10, "tasks": ["12.2"] }
  ]
}
```
