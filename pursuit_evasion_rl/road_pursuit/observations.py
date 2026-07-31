"""도로 추격 환경 관측값 구성 모듈.

RoadObservationBuilder 클래스를 제공한다.
이산 환경과 호환 가능한 관측 벡터를 구성하여
기존 MAPPO 알고리즘을 재활용할 수 있도록 한다.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from pursuit_evasion_rl.road_pursuit.road_network_2d import RoadNetwork2D
from pursuit_evasion_rl.road_pursuit.vehicle_model import VehicleState


class RoadObservationBuilder:
    """도로 추격 환경 관측값 빌더.

    이산 환경과 동일한 구조의 관측값을 생성한다:
    - my_position: 가장 가까운 교차로 ID (정규화)
    - neighbors: 현재/가장가까운 교차로의 outgoing 세그먼트 목적지 ID (패딩)
    - other_positions: 다른 에이전트들의 가장 가까운 교차로 ID
    - nearest_boundary_dist: 경계 교차로까지 최단 홉 수
    - current_step: 현재 스텝
    - my_progress: 세그먼트 내 진행률 (추가 정보)

    Attributes:
        network: 2D 도로 네트워크
        num_police: 경찰 에이전트 수
        max_degree: 최대 outgoing 차수 (NN 입력 크기 고정)
        num_agents: 전체 에이전트 수
    """

    def __init__(
        self,
        network: RoadNetwork2D,
        num_police: int,
        max_degree: int,
    ) -> None:
        """초기화.

        Args:
            network: 2D 도로 네트워크
            num_police: 경찰 에이전트 수
            max_degree: 고정 최대 차수
        """
        self.network = network
        self.num_police = num_police
        self.max_degree = max_degree
        self.num_agents = num_police + 1

    def build_observation_space(self) -> spaces.Dict:
        """관측 공간을 정의한다.

        Returns:
            gymnasium.spaces.Dict: 관측 공간
        """
        n = self.network.num_intersections

        return spaces.Dict(
            {
                "my_position": spaces.Box(
                    low=0, high=n - 1, shape=(1,), dtype=np.int64
                ),
                "my_progress": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "neighbors": spaces.Box(
                    low=-1, high=n - 1, shape=(self.max_degree,), dtype=np.int64
                ),
                "other_positions": spaces.Box(
                    low=-1, high=n - 1, shape=(self.num_agents - 1,), dtype=np.int64
                ),
                "nearest_boundary_dist": spaces.Box(
                    low=0, high=n, shape=(1,), dtype=np.int64
                ),
                "current_step": spaces.Box(
                    low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64
                ),
                "agent_id_onehot": spaces.Box(
                    low=0, high=1, shape=(self.num_police,), dtype=np.float32
                ),
            }
        )

    def get_observation(
        self,
        agent_id: str,
        vehicle_states: dict[str, VehicleState],
        current_step: int,
    ) -> dict[str, np.ndarray]:
        """특정 에이전트의 관측값을 계산한다.

        Args:
            agent_id: 관측 대상 에이전트 ID
            vehicle_states: 전체 차량 상태 딕셔너리
            current_step: 현재 스텝 수

        Returns:
            관측값 딕셔너리
        """
        my_state = vehicle_states[agent_id]

        # 가장 가까운 교차로
        my_intersection = my_state.nearest_intersection(self.network)

        # my_position
        my_position = np.array([my_intersection], dtype=np.int64)

        # my_progress
        my_progress = np.array(
            [max(0.0, min(1.0, my_state.progress))], dtype=np.float32
        )

        # neighbors: 현재 교차로의 outgoing 세그먼트 목적지 ID
        neighbors = self._get_padded_neighbors(my_intersection)

        # other_positions: 다른 에이전트의 가장 가까운 교차로
        other_positions = self._get_other_positions(agent_id, vehicle_states)

        # nearest_boundary_dist
        boundary_dist = self._compute_nearest_boundary_dist(my_intersection)
        nearest_boundary_dist = np.array([boundary_dist], dtype=np.int64)

        # current_step
        step_arr = np.array([current_step], dtype=np.int64)

        # agent_id_onehot: 경찰 ID one-hot 벡터 (도주자는 전부 0)
        agent_id_onehot = np.zeros(self.num_police, dtype=np.float32)
        if agent_id.startswith("police_"):
            try:
                idx = int(agent_id.split("_")[1])
                if 0 <= idx < self.num_police:
                    agent_id_onehot[idx] = 1.0
            except (IndexError, ValueError):
                pass

        return {
            "my_position": my_position,
            "my_progress": my_progress,
            "neighbors": neighbors,
            "other_positions": other_positions,
            "nearest_boundary_dist": nearest_boundary_dist,
            "current_step": step_arr,
            "agent_id_onehot": agent_id_onehot,
        }

    def _get_padded_neighbors(self, intersection_id: int) -> np.ndarray:
        """교차로의 outgoing 목적지 ID를 패딩하여 반환한다.

        Args:
            intersection_id: 교차로 ID

        Returns:
            shape (max_degree,)의 int64 배열, 미사용 슬롯은 -1
        """
        padded = np.full(self.max_degree, -1, dtype=np.int64)
        outgoing = self.network.get_outgoing_segments(intersection_id)
        for i, seg in enumerate(outgoing[: self.max_degree]):
            padded[i] = seg.end_intersection_id
        return padded

    def _get_other_positions(
        self, agent_id: str, vehicle_states: dict[str, VehicleState]
    ) -> np.ndarray:
        """다른 에이전트들의 가장 가까운 교차로 ID를 반환한다.

        Args:
            agent_id: 현재 에이전트 ID
            vehicle_states: 전체 차량 상태

        Returns:
            shape (num_agents - 1,)의 int64 배열
        """
        other_pos = []
        for aid, state in vehicle_states.items():
            if aid != agent_id:
                other_pos.append(state.nearest_intersection(self.network))

        return np.array(other_pos, dtype=np.int64)

    def _compute_nearest_boundary_dist(self, intersection_id: int) -> int:
        """가장 가까운 경계 교차로까지의 최단 홉 수.

        Args:
            intersection_id: 현재 교차로 ID

        Returns:
            최단 홉 수
        """
        boundary = self.network.get_boundary_intersections()
        if not boundary:
            return 0

        min_dist = self.network.num_intersections
        for b_id in boundary:
            dist = self.network.shortest_path_hops(intersection_id, b_id)
            if dist < min_dist:
                min_dist = dist

        return min_dist
