"""차량 상태 및 이동 모델 모듈.

VehicleState 데이터클래스와 차량 전진 함수를 제공한다.
차량은 도로 세그먼트 위에서 progress(0~1)로 위치가 표현되며,
매 스텝 speed * dt / segment_length만큼 전진한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from pursuit_evasion_rl.road_pursuit.road_network_2d import RoadNetwork2D


@dataclass
class VehicleState:
    """차량 상태.

    Attributes:
        agent_id: 에이전트 고유 ID
        segment_id: 현재 위치한 도로 세그먼트 ID
        progress: 세그먼트 내 진행률 (0.0=출발 교차로, 1.0=도착 교차로)
        speed: 이동 속도 (세그먼트의 speed_limit로 고정)
        agent_type: 에이전트 유형 ("police" | "fugitive")
    """

    agent_id: str
    segment_id: int
    progress: float
    speed: float
    agent_type: str  # "police" | "fugitive"

    @property
    def at_intersection(self) -> bool:
        """도착 교차로에 도달했는지 여부."""
        return self.progress >= 1.0

    def position_2d(self, network: RoadNetwork2D) -> tuple[float, float]:
        """현재 2D 좌표를 세그먼트의 시작/끝 좌표로부터 보간한다.

        Args:
            network: 도로 네트워크

        Returns:
            (x, y) 보간된 좌표
        """
        segment = network.get_segment(self.segment_id)
        t = max(0.0, min(1.0, self.progress))
        x = segment.start_pos[0] + t * (segment.end_pos[0] - segment.start_pos[0])
        y = segment.start_pos[1] + t * (segment.end_pos[1] - segment.start_pos[1])
        return (x, y)

    def nearest_intersection(self, network: RoadNetwork2D) -> int:
        """가장 가까운 교차로 ID를 반환한다.

        progress < 0.5이면 출발 교차로, >= 0.5이면 도착 교차로.

        Args:
            network: 도로 네트워크

        Returns:
            가장 가까운 교차로 ID
        """
        segment = network.get_segment(self.segment_id)
        if self.progress < 0.5:
            return segment.start_intersection_id
        return segment.end_intersection_id

    def end_intersection(self, network: RoadNetwork2D) -> int:
        """현재 세그먼트의 도착 교차로 ID를 반환한다."""
        segment = network.get_segment(self.segment_id)
        return segment.end_intersection_id


def advance_vehicle(
    state: VehicleState, network: RoadNetwork2D, dt: float
) -> VehicleState:
    """차량을 현재 세그먼트에서 전진시킨다.

    progress += speed * dt / segment_length.
    progress가 1.0 이상이 되면 교차로 도달 상태가 된다.

    Args:
        state: 현재 차량 상태
        network: 도로 네트워크
        dt: 시간 스텝

    Returns:
        업데이트된 VehicleState (새 객체)
    """
    segment = network.get_segment(state.segment_id)

    # 이미 교차로에 도달한 상태라면 전진하지 않음 (행동 대기)
    if state.at_intersection:
        return VehicleState(
            agent_id=state.agent_id,
            segment_id=state.segment_id,
            progress=state.progress,
            speed=state.speed,
            agent_type=state.agent_type,
        )

    # progress 갱신
    if segment.length > 0:
        delta = state.speed * dt / segment.length
    else:
        delta = 1.0  # 길이 0 세그먼트는 즉시 통과

    new_progress = state.progress + delta

    return VehicleState(
        agent_id=state.agent_id,
        segment_id=state.segment_id,
        progress=new_progress,
        speed=state.speed,
        agent_type=state.agent_type,
    )


def place_vehicle_at_intersection(
    agent_id: str,
    intersection_id: int,
    segment_id: int,
    network: RoadNetwork2D,
    agent_type: str,
) -> VehicleState:
    """교차로에서 특정 세그먼트로 출발하는 차량 상태를 생성한다.

    Args:
        agent_id: 에이전트 ID
        intersection_id: 출발 교차로 ID
        segment_id: 선택한 세그먼트 ID
        network: 도로 네트워크
        agent_type: 에이전트 유형

    Returns:
        새 VehicleState (progress=0.0)
    """
    segment = network.get_segment(segment_id)
    return VehicleState(
        agent_id=agent_id,
        segment_id=segment_id,
        progress=0.0,
        speed=segment.speed_limit,
        agent_type=agent_type,
    )
