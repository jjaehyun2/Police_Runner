"""연속 환경 관측 공간 빌더 모듈.

고정 차원(51) 관측 벡터를 구성하며, gymnasium.spaces.Box를 정의한다.
관측 구성: self(4) + fugitive_pos(2) + police_positions(40) + boundary_dist(4) + nearest_exit(1)
"""

from __future__ import annotations

import numpy as np
import gymnasium

from pursuit_evasion_rl.continuous_env.agent_model import AgentState
from pursuit_evasion_rl.continuous_env.road_map import RoadMap


class ContinuousObservationBuilder:
    """연속 환경 관측 벡터 빌더.

    관측 벡터 구조 (총 51차원, float32):
        [0:4]   자기 정보: [x, y, speed, heading] (정규화 [0,1])
        [4:6]   도망자 위치: [x, y] (경찰만; 도망자는 [0,0])
        [6:46]  경찰 위치: [(x,y) * 20] (최대 20대, 미사용 슬롯은 (-1,-1))
        [46:50] 경계 거리: [up, down, left, right] (정규화)
        [50]    최근접 출구 거리: [dist] (정규화)
    """

    OBS_DIM = 51
    MAX_POLICE = 20

    def __init__(self, road_map: RoadMap) -> None:
        self.road_map = road_map
        min_x, min_y, max_x, max_y = road_map.map_bounds
        self.map_width = max_x - min_x
        self.map_height = max_y - min_y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self._boundary_exits = road_map.get_boundary_exits()

    def build_observation_space(self) -> gymnasium.spaces.Box:
        """관측 공간을 정의한다."""
        low = np.full(self.OBS_DIM, -1.0, dtype=np.float32)
        high = np.full(self.OBS_DIM, 1.0, dtype=np.float32)
        return gymnasium.spaces.Box(low=low, high=high, shape=(self.OBS_DIM,), dtype=np.float32)

    def get_observation(
        self,
        agent_id: str,
        all_states: dict[str, AgentState],
        num_police: int,
        max_police: int = 20,
    ) -> np.ndarray:
        """에이전트의 관측 벡터를 구성한다.

        Args:
            agent_id: 관측 대상 에이전트 ID
            all_states: 모든 에이전트 상태 딕셔너리
            num_police: 현재 경찰 수
            max_police: 최대 경찰 수 (패딩용)

        Returns:
            shape=(51,) float32 관측 벡터
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        state = all_states[agent_id]

        # [0:4] 자기 정보 (정규화 [0,1])
        obs[0] = (state.position[0] - self.min_x) / max(self.map_width, 1e-6)
        obs[1] = (state.position[1] - self.min_y) / max(self.map_height, 1e-6)
        obs[2] = state.speed / max(state.max_speed, 1e-6)
        obs[3] = state.heading / (2.0 * np.pi)

        # [4:6] 도망자 위치 (경찰만 관측 가능; 도망자는 [0,0])
        if state.agent_type == "police" and "fugitive" in all_states:
            fug_state = all_states["fugitive"]
            obs[4] = (fug_state.position[0] - self.min_x) / max(self.map_width, 1e-6)
            obs[5] = (fug_state.position[1] - self.min_y) / max(self.map_height, 1e-6)
        else:
            obs[4] = 0.0
            obs[5] = 0.0

        # [6:46] 경찰 위치 (최대 20대, 패딩 (-1,-1))
        police_idx = 0
        for aid, s in all_states.items():
            if s.agent_type == "police" and aid != agent_id:
                if police_idx < max_police:
                    base = 6 + police_idx * 2
                    obs[base] = (s.position[0] - self.min_x) / max(self.map_width, 1e-6)
                    obs[base + 1] = (s.position[1] - self.min_y) / max(self.map_height, 1e-6)
                    police_idx += 1

        # 미사용 슬롯은 -1로 패딩
        for i in range(police_idx, max_police):
            base = 6 + i * 2
            obs[base] = -1.0
            obs[base + 1] = -1.0

        # [46:50] 경계 거리 (정규화)
        px, py = state.position[0], state.position[1]
        diag = max(np.hypot(self.map_width, self.map_height), 1e-6)
        obs[46] = (self.max_y - py) / diag  # up
        obs[47] = (py - self.min_y) / diag  # down
        obs[48] = (px - self.min_x) / diag  # left
        obs[49] = (self.max_x - px) / diag  # right

        # [50] 최근접 출구 거리 (정규화)
        if self._boundary_exits:
            min_exit_dist = min(
                np.hypot(px - ex[0], py - ex[1]) for ex in self._boundary_exits
            )
            obs[50] = min_exit_dist / diag
        else:
            obs[50] = 1.0

        return obs
