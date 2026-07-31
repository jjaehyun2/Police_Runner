"""도주자 휴리스틱 에이전트.

제한된 시야 내 경찰을 피하면서 최단 경로로 탈출구를 향해 이동한다.
경찰의 RL 학습 상대로 사용된다 (self-play 대신).
"""

from __future__ import annotations

import numpy as np

from pursuit_evasion_rl.road_pursuit.road_network_2d import RoadNetwork2D
from pursuit_evasion_rl.road_pursuit.vehicle_model import VehicleState


class HeuristicFugitive:
    """휴리스틱 도주자 에이전트.

    행동 규칙:
    1. random_rate 확률로 랜덤 행동 (예측 불가능성)
    2. 시야 내 경찰이 있으면 → 경찰 반대 방향 도로 선택
    3. 시야 내 경찰 없으면 → 최단 경로로 가장 가까운 탈출구 향해 이동

    Attributes:
        network: 도로 네트워크
        vision_range: 시야 거리 (유클리드)
        random_rate: 랜덤 행동 확률 (0.0~1.0)
    """

    def __init__(
        self,
        network: RoadNetwork2D,
        vision_range: float = 30.0,
        random_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """초기화.

        Args:
            network: 2D 도로 네트워크
            vision_range: 경찰 감지 거리 (기본 30m = 1블록)
            random_rate: 랜덤 행동 확률 (기본 0.0)
            seed: 랜덤 시드
        """
        self.network = network
        self.vision_range = vision_range
        self.random_rate = random_rate
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        fugitive_state: VehicleState,
        police_states: list[VehicleState],
        fixed_max_degree: int = 5,
    ) -> int:
        """도주자의 행동을 결정한다.

        Args:
            fugitive_state: 도주자 차량 상태
            police_states: 모든 경찰 차량 상태 리스트
            fixed_max_degree: 행동 공간 크기 (stay = fixed_max_degree)

        Returns:
            행동 인덱스 (outgoing 세그먼트 인덱스 또는 stay)
        """
        # 교차로에 있지 않으면 행동 무의미 (자동 전진)
        if not fugitive_state.at_intersection:
            return 0  # 아무 값이나 (무시됨)

        # 현재 교차로
        current_inter_id = fugitive_state.end_intersection(self.network)
        current_pos = self.network.get_intersection(current_inter_id).position

        # outgoing 세그먼트 목록
        outgoing = self.network.get_outgoing_segments(current_inter_id)
        if not outgoing:
            return fixed_max_degree  # stay (갈 데가 없음)

        # 랜덤 행동 (예측 불가능성)
        if self.random_rate > 0 and self._rng.random() < self.random_rate:
            num_valid = min(len(outgoing), fixed_max_degree)
            return int(self._rng.integers(0, num_valid))

        # 시야 내 경찰 찾기
        visible_police_positions = []
        for ps in police_states:
            police_pos = ps.position_2d(self.network)
            dx = police_pos[0] - current_pos[0]
            dy = police_pos[1] - current_pos[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist <= self.vision_range:
                visible_police_positions.append(police_pos)

        if visible_police_positions:
            # 경찰이 보임 → 경찰 반대 방향으로 이동
            return self._flee_from_police(
                current_pos, outgoing, visible_police_positions, fixed_max_degree
            )
        else:
            # 경찰 안 보임 → 가장 가까운 탈출구 방향으로 이동
            return self._move_toward_exit(
                current_inter_id, outgoing, fixed_max_degree
            )

    def _flee_from_police(
        self,
        current_pos: tuple[float, float],
        outgoing: list,
        police_positions: list[tuple[float, float]],
        fixed_max_degree: int,
    ) -> int:
        """경찰 반대 방향의 도로를 선택한다."""
        # 경찰들의 평균 방향 벡터
        avg_police_dx = sum(p[0] - current_pos[0] for p in police_positions) / len(police_positions)
        avg_police_dy = sum(p[1] - current_pos[1] for p in police_positions) / len(police_positions)

        # 경찰 반대 방향
        flee_dx = -avg_police_dx
        flee_dy = -avg_police_dy

        # 각 outgoing 세그먼트의 방향과 flee 방향의 내적이 최대인 것 선택
        best_idx = 0
        best_score = float("-inf")

        for i, seg in enumerate(outgoing):
            if i >= fixed_max_degree:
                break
            seg_dx = seg.end_pos[0] - seg.start_pos[0]
            seg_dy = seg.end_pos[1] - seg.start_pos[1]
            mag = (seg_dx**2 + seg_dy**2) ** 0.5
            if mag < 1e-9:
                continue
            # 내적 (도트 프로덕트)
            score = (flee_dx * seg_dx + flee_dy * seg_dy) / mag
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _move_toward_exit(
        self,
        current_inter_id: int,
        outgoing: list,
        fixed_max_degree: int,
    ) -> int:
        """가장 가까운 탈출구 방향으로 이동한다."""
        boundary = self.network.get_boundary_intersections()
        if not boundary:
            return 0  # fallback

        # 각 outgoing 세그먼트의 도착 교차로에서 가장 가까운 탈출구까지 거리
        best_idx = 0
        best_dist = float("inf")

        for i, seg in enumerate(outgoing):
            if i >= fixed_max_degree:
                break
            dest_id = seg.end_intersection_id
            # 도착 교차로에서 가장 가까운 탈출구까지 최단 홉
            min_hops = min(
                self.network.shortest_path_hops(dest_id, b) for b in boundary
            )
            if min_hops < best_dist:
                best_dist = min_hops
                best_idx = i

        return best_idx
