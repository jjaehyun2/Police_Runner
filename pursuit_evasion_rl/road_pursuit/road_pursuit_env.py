"""하이브리드 도로 추격 환경 메인 모듈.

RoadPursuitEnv: Gymnasium 환경.
차량은 도로 세그먼트 위를 자동으로 전진하며,
교차로 도달 시 이산 행동으로 다음 도로를 선택한다.
기존 MAPPOAlgorithm과 호환 가능한 이산 행동 공간을 사용한다.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from pursuit_evasion_rl.road_pursuit.observations import RoadObservationBuilder
from pursuit_evasion_rl.road_pursuit.rewards import RoadRewardCalculator
from pursuit_evasion_rl.road_pursuit.road_network_2d import RoadNetwork2D
from pursuit_evasion_rl.road_pursuit.vehicle_model import (
    VehicleState,
    advance_vehicle,
    place_vehicle_at_intersection,
)

logger = logging.getLogger(__name__)


class RoadPursuitEnv(gymnasium.Env):
    """하이브리드 도로 추격 환경.

    차량이 도로 세그먼트 위를 자동으로 전진하며,
    교차로 도달 시 이산 행동으로 다음 도로를 선택하는 환경.

    행동 공간: Discrete(fixed_max_degree + 1) per agent
    - 0 ~ fixed_max_degree-1: outgoing 세그먼트 인덱스 선택
    - fixed_max_degree: 교차로에서 대기 (1스텝)

    차량이 세그먼트 중간(progress < 1.0)에 있을 때:
    - 행동은 무시되고 차량은 자동 전진만 한다.
    - 에이전트는 여전히 유효한 행동을 출력해야 한다 (어떤 값이든 가능).

    Config:
        grid_rows: 그리드 행 수 (기본 4)
        grid_cols: 그리드 열 수 (기본 4)
        block_size: 블록 크기 미터 (기본 100.0)
        num_police: 경찰 에이전트 수 (기본 4)
        speed: 전체 차량 속도 (기본 50.0)
        capture_radius: 체포 반경 (기본 15.0)
        max_steps: 최대 스텝 수 (기본 200)
        dt: 시간 스텝 (기본 0.1)
        fixed_max_degree: 고정 최대 차수 (기본 5)

    Attributes:
        metadata: 렌더링 모드
        network: 2D 도로 네트워크
        num_police: 경찰 에이전트 수
        max_steps: 에피소드 최대 스텝 수
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None) -> None:
        """RoadPursuitEnv 초기화.

        Args:
            config: 환경 설정 딕셔너리
            render_mode: 렌더링 모드
        """
        super().__init__()

        if config is None:
            config = {}

        self.render_mode = render_mode
        self._config = config

        # 환경 파라미터
        self.num_police: int = config.get("num_police", 4)
        self.max_steps: int = config.get("max_steps", 200)
        self._speed: float = config.get("speed", 50.0)
        self._dt: float = config.get("dt", 0.1)
        self._capture_radius: float = config.get("capture_radius", 15.0)
        self._fixed_max_degree: int = config.get("fixed_max_degree", 5)

        # 에이전트 ID
        self._police_ids: list[str] = [f"police_{i}" for i in range(self.num_police)]
        self._fugitive_id: str = "fugitive"
        self._agent_ids: list[str] = self._police_ids + [self._fugitive_id]

        # 도로 네트워크 생성
        network_config = {
            "grid_rows": config.get("grid_rows", 4),
            "grid_cols": config.get("grid_cols", 4),
            "block_size": config.get("block_size", 100.0),
            "speed_limit": self._speed,
            "road_width": config.get("road_width", 10.0),
            "boundary_ratio": config.get("boundary_ratio", 0.5),
            "num_dead_ends": config.get("num_dead_ends", 2),
            "seed": config.get("seed", None),
        }
        self.network = RoadNetwork2D(network_config)

        # 컴포넌트 초기화
        self._obs_builder = RoadObservationBuilder(
            network=self.network,
            num_police=self.num_police,
            max_degree=self._fixed_max_degree,
        )
        self._reward_calculator = RoadRewardCalculator(
            network=self.network,
            config={
                "terminal_reward": config.get("terminal_reward", 1.0),
                "shaping_scale": config.get("shaping_scale", 0.1),
                "cooperation_bonus": config.get("cooperation_bonus", 0.05),
                "capture_radius": self._capture_radius,
            },
        )

        # 관측/행동 공간 정의
        individual_obs_space = self._obs_builder.build_observation_space()
        self.observation_space = spaces.Dict(
            {agent_id: individual_obs_space for agent_id in self._agent_ids}
        )

        # 행동 공간: Discrete(fixed_max_degree + 1)
        # 마지막 인덱스 = stay(대기)
        individual_action_space = spaces.Discrete(self._fixed_max_degree + 1)
        self.action_space = spaces.Dict(
            {agent_id: individual_action_space for agent_id in self._agent_ids}
        )

        # 내부 상태
        self._vehicle_states: dict[str, VehicleState] | None = None
        self._current_step: int = 0
        self._done: bool = False
        self._rng: np.random.Generator | None = None

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """환경을 초기화한다.

        Args:
            seed: 랜덤 시드
            options: 추가 옵션

        Returns:
            (observations, info) 튜플
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self._current_step = 0
        self._done = False

        # 도메인 랜덤화 적용
        self._apply_domain_randomization()

        # 에이전트 배치
        self._vehicle_states = self._place_agents()

        # 관측값 생성
        observations = self._build_observations()
        info = {agent_id: {} for agent_id in self._agent_ids}

        return observations, info

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict, dict, dict, dict, dict]:
        """한 스텝을 실행한다.

        1. 모든 차량을 전진시킨다 (advance)
        2. 교차로에 도달한 차량에 대해 행동을 적용한다
        3. 종료 조건을 검사한다
        4. 보상을 계산한다

        Args:
            actions: 에이전트별 행동 딕셔너리

        Returns:
            (observations, rewards, terminated, truncated, info) 튜플
        """
        if self._vehicle_states is None:
            raise RuntimeError("step() 전에 reset()을 호출하세요.")

        # 이전 상태 저장
        prev_states = dict(self._vehicle_states)

        # 1. 모든 차량 전진
        for aid in self._agent_ids:
            self._vehicle_states[aid] = advance_vehicle(
                self._vehicle_states[aid], self.network, self._dt
            )

        # 2. 종료 조건 검사 (행동 적용 전에! → 탈출 판정 누락 방지)
        termination_result = self._check_termination()

        # 3. 교차로 도달한 차량에 행동 적용 (종료가 아닌 경우에만)
        if termination_result is None:
            for aid in self._agent_ids:
                state = self._vehicle_states[aid]
                if state.at_intersection:
                    action = actions.get(aid, self._fixed_max_degree)
                    self._vehicle_states[aid] = self._apply_intersection_action(
                        state, action
                    )

        # 4. 스텝 카운터 증가
        self._current_step += 1

        # 종료 판정 결과 사용 (위에서 이미 계산)
        terminated_flag = termination_result in ("police_win", "fugitive_win")
        truncated_flag = termination_result == "timeout"

        # 타임아웃은 별도 체크 (스텝 증가 후)
        if not terminated_flag and not truncated_flag and self._current_step >= self.max_steps:
            termination_result = "timeout"
            truncated_flag = True

        self._done = terminated_flag or truncated_flag

        # 5. 보상 계산
        rewards = self._reward_calculator.compute_step_rewards(
            vehicle_states=self._vehicle_states,
            prev_vehicle_states=prev_states,
            termination_result=termination_result,
        )

        # 6. 관측값 생성
        observations = self._build_observations()

        terminated = {aid: terminated_flag for aid in self._agent_ids}
        truncated = {aid: truncated_flag for aid in self._agent_ids}
        info = {
            aid: {"termination_reason": termination_result}
            for aid in self._agent_ids
        }

        return observations, rewards, terminated, truncated, info

    def get_action_masks(self) -> dict[str, np.ndarray]:
        """각 에이전트의 행동 마스크를 반환한다.

        - 세그먼트 중간(progress < 1.0): 모든 행동 허용 (어차피 무시됨)
        - 교차로 도달(progress >= 1.0): 유효한 outgoing 인덱스 + stay만 허용

        Returns:
            에이전트별 행동 마스크 딕셔너리
        """
        action_size = self._fixed_max_degree + 1

        if self._vehicle_states is None:
            mask = np.ones(action_size, dtype=bool)
            return {aid: mask.copy() for aid in self._agent_ids}

        masks: dict[str, np.ndarray] = {}
        for aid in self._agent_ids:
            state = self._vehicle_states[aid]
            mask = np.zeros(action_size, dtype=bool)

            if not state.at_intersection:
                # 세그먼트 중간: 모든 행동 허용 (무시됨)
                mask[:] = True
            else:
                # 교차로: 유효한 outgoing만 허용
                end_inter = state.end_intersection(self.network)
                outgoing = self.network.get_outgoing_segments(end_inter)
                for i in range(min(len(outgoing), self._fixed_max_degree)):
                    mask[i] = True
                # stay 행동은 항상 허용
                mask[self._fixed_max_degree] = True

            masks[aid] = mask

        return masks

    def render(self) -> None:
        """현재 상태의 텍스트 표현을 출력한다."""
        if self._vehicle_states is None:
            print("환경이 초기화되지 않았습니다.")
            return

        print(f"=== Step {self._current_step} ===")
        print(f"네트워크: {self.network.num_intersections} 교차로, "
              f"{self.network.num_segments} 세그먼트")
        print("차량 상태:")
        for aid in self._agent_ids:
            state = self._vehicle_states[aid]
            pos = state.position_2d(self.network)
            inter = state.nearest_intersection(self.network)
            at_inter = "★교차로" if state.at_intersection else f"진행중({state.progress:.2f})"
            print(f"  {aid}: seg={state.segment_id}, {at_inter}, "
                  f"교차로={inter}, 좌표=({pos[0]:.1f}, {pos[1]:.1f})")
        print()

    def _place_agents(self) -> dict[str, VehicleState]:
        """에이전트를 초기 위치에 배치한다.

        도망자: 네트워크 중심부 교차로에서 출발
        경찰: 경계 근처 교차로에서 출발

        Returns:
            에이전트별 VehicleState 딕셔너리
        """
        states: dict[str, VehicleState] = {}
        used_intersections: set[int] = set()

        all_intersections = list(self.network.intersections.keys())
        boundary = set(self.network.get_boundary_intersections())
        center = [i for i in all_intersections if i not in boundary]

        if not center:
            center = all_intersections

        # 도망자 배치: 중심부 교차로
        fug_inter = int(self._rng.choice(center))
        used_intersections.add(fug_inter)
        fug_outgoing = self.network.get_outgoing_segments(fug_inter)
        if fug_outgoing:
            fug_seg = fug_outgoing[int(self._rng.integers(len(fug_outgoing)))]
            states[self._fugitive_id] = place_vehicle_at_intersection(
                self._fugitive_id, fug_inter, fug_seg.segment_id,
                self.network, "fugitive"
            )
        else:
            # fallback: 첫 번째 세그먼트
            first_seg_id = next(iter(self.network.segments.keys()))
            states[self._fugitive_id] = VehicleState(
                agent_id=self._fugitive_id,
                segment_id=first_seg_id,
                progress=0.0,
                speed=self._speed,
                agent_type="fugitive",
            )

        # 경찰 배치: 경계 또는 나머지 교차로
        police_candidates = [i for i in all_intersections if i not in used_intersections]
        # 경계 교차로 우선
        boundary_candidates = [i for i in police_candidates if i in boundary]
        other_candidates = [i for i in police_candidates if i not in boundary]
        ordered = boundary_candidates + other_candidates

        for pid in self._police_ids:
            available = [i for i in ordered if i not in used_intersections]
            if not available:
                available = all_intersections  # 중복 허용 fallback

            chosen_inter = int(self._rng.choice(available))
            used_intersections.add(chosen_inter)

            outgoing = self.network.get_outgoing_segments(chosen_inter)
            if outgoing:
                seg = outgoing[int(self._rng.integers(len(outgoing)))]
                states[pid] = place_vehicle_at_intersection(
                    pid, chosen_inter, seg.segment_id, self.network, "police"
                )
            else:
                # fallback
                first_seg_id = next(iter(self.network.segments.keys()))
                states[pid] = VehicleState(
                    agent_id=pid,
                    segment_id=first_seg_id,
                    progress=0.0,
                    speed=self._speed,
                    agent_type="police",
                )

        return states

    def _apply_intersection_action(
        self, state: VehicleState, action: int
    ) -> VehicleState:
        """교차로에서 행동을 적용하여 새 세그먼트로 출발시킨다.

        Args:
            state: 현재 차량 상태 (at_intersection=True)
            action: 행동 인덱스

        Returns:
            새 VehicleState
        """
        end_inter = state.end_intersection(self.network)
        outgoing = self.network.get_outgoing_segments(end_inter)

        # stay 행동 또는 유효하지 않은 인덱스 → 대기
        if action >= self._fixed_max_degree or action >= len(outgoing):
            # 대기: progress를 1.0으로 유지 (다음 스텝에서 다시 행동 선택)
            # 실제로는 같은 세그먼트의 끝에 머무름
            return VehicleState(
                agent_id=state.agent_id,
                segment_id=state.segment_id,
                progress=1.0,
                speed=state.speed,
                agent_type=state.agent_type,
            )

        # 선택한 세그먼트로 출발
        chosen_seg = outgoing[action]
        return VehicleState(
            agent_id=state.agent_id,
            segment_id=chosen_seg.segment_id,
            progress=0.0,
            speed=chosen_seg.speed_limit,
            agent_type=state.agent_type,
        )

    def _apply_domain_randomization(self) -> None:
        """매 에피소드 도메인 랜덤화를 적용한다.

        1. 랜덤 도로 차단 (10% 세그먼트 속도 0 → 통과 불가)
        2. 교통 체증 (20% 세그먼트 속도 50% 감소)
        3. 도주자 은신 확률 설정 (5% 확률로 관측에서 사라짐)
        """
        # 도로 차단: 10% 세그먼트를 임시 차단 (speed_limit을 매우 낮게)
        all_seg_ids = list(self.network.segments.keys())
        num_block = max(1, int(len(all_seg_ids) * 0.05))
        blocked = self._rng.choice(all_seg_ids, size=num_block, replace=False)
        
        # 원래 속도 백업 (첫 호출 시)
        if not hasattr(self, '_original_speeds'):
            self._original_speeds = {
                sid: seg.speed_limit for sid, seg in self.network.segments.items()
            }
        
        # 속도 원복
        for sid, speed in self._original_speeds.items():
            if sid in self.network.segments:
                self.network.segments[sid].speed_limit = speed
        
        # 도로 차단 (속도를 5로 → 사실상 매우 느림)
        for sid in blocked:
            if sid in self.network.segments:
                self.network.segments[sid].speed_limit = 5.0
        
        # 교통 체증: 20% 세그먼트 속도 50% 감소
        num_congestion = max(1, int(len(all_seg_ids) * 0.15))
        congested = self._rng.choice(all_seg_ids, size=num_congestion, replace=False)
        for sid in congested:
            if sid in self.network.segments and sid not in blocked:
                self.network.segments[sid].speed_limit *= 0.5
        
        # 도주자 은신 확률 저장 (step에서 사용)
        self._fugitive_hide_prob = 0.05

    def _check_termination(self) -> str | None:
        """종료 조건을 검사한다.

        Returns:
            "police_win", "fugitive_win", "timeout", 또는 None
        """
        # 체포 확인
        if self._reward_calculator.check_capture(self._vehicle_states):
            return "police_win"

        # 탈출 확인
        if self._reward_calculator.check_escape(self._vehicle_states):
            return "fugitive_win"

        # 타임아웃
        if self._current_step >= self.max_steps:
            return "timeout"

        return None

    def _build_observations(self) -> dict[str, dict[str, np.ndarray]]:
        """모든 에이전트의 관측값을 생성한다.
        
        도주자 은신 확률에 따라 경찰의 관측에서 도주자 위치가 -1로 표시될 수 있다.
        """
        # 도주자 은신 판정 (매 스텝)
        fugitive_hidden = False
        if hasattr(self, '_fugitive_hide_prob') and self._fugitive_hide_prob > 0:
            if self._rng is not None and self._rng.random() < self._fugitive_hide_prob:
                fugitive_hidden = True

        observations: dict[str, dict[str, np.ndarray]] = {}
        
        # 은신 중이면 임시로 도주자 위치를 교란
        temp_states = dict(self._vehicle_states)
        if fugitive_hidden:
            # 경찰 관측에서 도주자의 nearest_intersection을 -1로 만들기 위해
            # 임시로 vehicle_states에서 도주자를 제거하지 않고, 
            # 관측 빌더에서 처리하도록 플래그 전달
            pass
        
        for aid in self._agent_ids:
            observations[aid] = self._obs_builder.get_observation(
                agent_id=aid,
                vehicle_states=self._vehicle_states,
                current_step=self._current_step,
            )
            
            # 도주자 은신 중이면 경찰의 other_positions에서 도주자를 -1로
            if fugitive_hidden and aid.startswith("police"):
                # other_positions의 마지막 값이 도주자 (agent 순서상)
                obs = observations[aid]
                positions = obs["other_positions"]
                positions[-1] = -1  # 도주자 위치를 -1로 마스킹

        return observations

    @property
    def agent_ids(self) -> list[str]:
        """모든 에이전트 ID 목록."""
        return list(self._agent_ids)

    @property
    def police_ids(self) -> list[str]:
        """경찰 에이전트 ID 목록."""
        return list(self._police_ids)

    @property
    def fugitive_id(self) -> str:
        """도망자 에이전트 ID."""
        return self._fugitive_id
