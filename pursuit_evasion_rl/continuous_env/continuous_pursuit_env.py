"""연속 2D 추격-도주 Gymnasium 환경.

RoadMap, AgentModel, Observations, Rewards 모듈을 통합하여
연속 좌표 공간에서의 다중 에이전트 추격-도주 시뮬레이션을 구현한다.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from pursuit_evasion_rl.continuous_env.agent_model import (
    AgentState,
    update_position,
)
from pursuit_evasion_rl.continuous_env.observations import ContinuousObservationBuilder
from pursuit_evasion_rl.continuous_env.rewards import ContinuousRewardCalculator
from pursuit_evasion_rl.continuous_env.road_map import RoadMap


class ContinuousPursuitEnv(gymnasium.Env):
    """연속 2D 추격-도주 Gymnasium 환경.

    Config keys:
        map_width (float): 맵 가로 크기 (기본 100.0)
        map_height (float): 맵 세로 크기 (기본 100.0)
        num_police (int): 경찰 수 (기본 5)
        max_speed_police (float): 경찰 최대 속도 (기본 8.0)
        max_speed_fugitive (float): 도망자 최대 속도 (기본 10.0)
        capture_radius (float): 체포 반경 (기본 5.0)
        dt (float): 시뮬레이션 시간 스텝 (기본 0.1)
        max_steps (int): 최대 스텝 수 (기본 1000)
        road_map_config (dict): 도로 맵 설정
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        config = config or {}

        self.map_width: float = config.get("map_width", 100.0)
        self.map_height: float = config.get("map_height", 100.0)
        self.num_police: int = config.get("num_police", 5)
        self.max_speed_police: float = config.get("max_speed_police", 8.0)
        self.max_speed_fugitive: float = config.get("max_speed_fugitive", 10.0)
        self.capture_radius: float = config.get("capture_radius", 5.0)
        self.dt: float = config.get("dt", 0.1)
        self.max_steps: int = config.get("max_steps", 1000)
        self.render_mode = render_mode

        # 도로 맵 설정
        road_config = config.get("road_map_config", {})
        self._road_map_config = road_config

        # 에이전트 ID 정의
        self.police_ids = [f"police_{i}" for i in range(self.num_police)]
        self.fugitive_id = "fugitive"
        self.agent_ids = self.police_ids + [self.fugitive_id]

        # 행동 공간: 각 에이전트 별 Box([speed, heading])
        self.action_space = gymnasium.spaces.Dict(
            {
                aid: gymnasium.spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array(
                        [
                            self.max_speed_police if aid.startswith("police") else self.max_speed_fugitive,
                            2.0 * np.pi,
                        ],
                        dtype=np.float32,
                    ),
                    shape=(2,),
                    dtype=np.float32,
                )
                for aid in self.agent_ids
            }
        )

        # 관측 공간: 각 에이전트 별 Box(shape=(51,))
        obs_low = np.full(ContinuousObservationBuilder.OBS_DIM, -1.0, dtype=np.float32)
        obs_high = np.full(ContinuousObservationBuilder.OBS_DIM, 1.0, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Dict(
            {
                aid: gymnasium.spaces.Box(
                    low=obs_low, high=obs_high, shape=(ContinuousObservationBuilder.OBS_DIM,), dtype=np.float32
                )
                for aid in self.agent_ids
            }
        )

        # 내부 상태
        self._states: dict[str, AgentState] = {}
        self._prev_states: dict[str, AgentState] = {}
        self._road_map: RoadMap | None = None
        self._obs_builder: ContinuousObservationBuilder | None = None
        self._reward_calc: ContinuousRewardCalculator | None = None
        self._current_step: int = 0
        self._is_reset: bool = False

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """환경을 초기화한다.

        Args:
            seed: 랜덤 시드
            options: 추가 옵션

        Returns:
            (observations, info) 튜플
        """
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # 도로 맵 생성
        self._road_map = RoadMap(config=self._road_map_config)
        self._road_map.map_bounds = (0.0, 0.0, self.map_width, self.map_height)

        mode = self._road_map_config.get("mode", "grid")
        if mode == "grid":
            rows = self._road_map_config.get("grid_rows", 5)
            cols = self._road_map_config.get("grid_cols", 5)
            spacing = self._road_map_config.get("grid_spacing", 20.0)
            self._road_map.generate_grid(rows, cols, spacing)
        elif mode == "random":
            num_seg = self._road_map_config.get("num_segments", 20)
            self._road_map.generate_random(num_seg, seed=seed)
        else:
            # 기본 그리드
            self._road_map.generate_grid(5, 5, 20.0)

        # 관측/보상 빌더 초기화
        self._obs_builder = ContinuousObservationBuilder(self._road_map)
        self._reward_calc = ContinuousRewardCalculator(self._road_map)

        # 에이전트를 도로 위에 랜덤 배치
        self._states = {}
        intersections = self._road_map.get_intersections()

        if intersections:
            positions = [inter.position for inter in intersections]
        else:
            # 교차점이 없으면 세그먼트 중간점 사용
            positions = [
                (
                    (seg.start_point[0] + seg.end_point[0]) / 2,
                    (seg.start_point[1] + seg.end_point[1]) / 2,
                )
                for seg in self._road_map.get_segments()
            ]

        if not positions:
            # fallback: 맵 중앙
            positions = [(self.map_width / 2, self.map_height / 2)]

        # 경찰 배치
        for pid in self.police_ids:
            idx = int(rng.integers(0, len(positions)))
            pos = positions[idx]
            self._states[pid] = AgentState(
                agent_id=pid,
                position=np.array(pos, dtype=np.float32),
                speed=0.0,
                heading=float(rng.uniform(0, 2 * np.pi)),
                agent_type="police",
                max_speed=self.max_speed_police,
            )

        # 도망자 배치 (경찰과 일정 거리 떨어진 위치 선호)
        fug_idx = int(rng.integers(0, len(positions)))
        fug_pos = positions[fug_idx]
        self._states[self.fugitive_id] = AgentState(
            agent_id=self.fugitive_id,
            position=np.array(fug_pos, dtype=np.float32),
            speed=0.0,
            heading=float(rng.uniform(0, 2 * np.pi)),
            agent_type="fugitive",
            max_speed=self.max_speed_fugitive,
        )

        self._prev_states = {aid: self._copy_state(s) for aid, s in self._states.items()}
        self._current_step = 0
        self._is_reset = True

        observations = self._get_all_observations()
        info: dict[str, Any] = {"step": 0}

        return observations, info

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """환경을 한 스텝 진행한다.

        Args:
            actions: 에이전트 ID → [speed, heading] 행동 딕셔너리

        Returns:
            (observations, rewards, terminated, truncated, info) 5-튜플
        """
        if not self._is_reset:
            raise RuntimeError("reset()을 먼저 호출해야 합니다.")

        assert self._road_map is not None
        assert self._obs_builder is not None
        assert self._reward_calc is not None

        # 이전 상태 저장
        self._prev_states = {aid: self._copy_state(s) for aid, s in self._states.items()}
        self._current_step += 1

        # 위치 갱신 및 오프로드 추적
        invalid_moves: dict[str, bool] = {}
        for aid in self.agent_ids:
            action = actions.get(aid, np.array([0.0, 0.0], dtype=np.float32))
            action = np.asarray(action, dtype=np.float32)

            old_state = self._states[aid]
            new_state = update_position(old_state, action, self.dt, self._road_map)

            # 오프로드 판정: 속도가 0으로 클램핑되었고 원래 속도 > 0이면 오프로드
            tolerance = self._road_map._config.get("segment_width", 3.0) if self._road_map._config else 3.0
            is_off_road = not self._road_map.is_on_road(tuple(new_state.position), tolerance=tolerance)
            invalid_moves[aid] = is_off_road

            self._states[aid] = new_state

        # 종료 조건 판정 (우선순위: escape > capture > timeout)
        termination_result = self._check_termination()

        # 보상 계산
        rewards = self._reward_calc.compute_rewards(
            self._states, self._prev_states, termination_result, invalid_moves
        )

        # terminated/truncated 딕셔너리
        is_done = termination_result is not None
        terminated = {aid: (termination_result in ("police_capture", "fugitive_escape")) for aid in self.agent_ids}
        truncated = {aid: (termination_result == "timeout") for aid in self.agent_ids}

        if not is_done:
            terminated = {aid: False for aid in self.agent_ids}
            truncated = {aid: False for aid in self.agent_ids}

        # 관측값
        observations = self._get_all_observations()

        info: dict[str, Any] = {
            "step": self._current_step,
            "termination": termination_result,
        }

        return observations, rewards, terminated, truncated, info

    def render(self) -> None:
        """렌더링 (현재 미구현)."""
        pass

    def _check_termination(self) -> str | None:
        """종료 조건을 확인한다. 우선순위: escape > capture > timeout."""
        fugitive_state = self._states.get(self.fugitive_id)
        if fugitive_state is None:
            return None

        # 1. 탈출: 도망자가 맵 경계 밖
        fug_pos = fugitive_state.position
        min_x, min_y, max_x, max_y = self._road_map.map_bounds  # type: ignore
        if fug_pos[0] < min_x or fug_pos[0] > max_x or fug_pos[1] < min_y or fug_pos[1] > max_y:
            return "fugitive_escape"

        # 2. 체포: 경찰-도망자 거리 <= capture_radius
        for pid in self.police_ids:
            police_state = self._states[pid]
            dist = float(np.linalg.norm(police_state.position - fug_pos))
            if dist <= self.capture_radius:
                return "police_capture"

        # 3. 타임아웃
        if self._current_step >= self.max_steps:
            return "timeout"

        return None

    def _get_all_observations(self) -> dict[str, np.ndarray]:
        """모든 에이전트의 관측값을 반환한다."""
        assert self._obs_builder is not None
        observations: dict[str, np.ndarray] = {}
        for aid in self.agent_ids:
            observations[aid] = self._obs_builder.get_observation(
                aid, self._states, self.num_police
            )
        return observations

    @staticmethod
    def _copy_state(state: AgentState) -> AgentState:
        """에이전트 상태의 딥 카피를 반환한다."""
        return AgentState(
            agent_id=state.agent_id,
            position=state.position.copy(),
            speed=state.speed,
            heading=state.heading,
            agent_type=state.agent_type,
            max_speed=state.max_speed,
        )
