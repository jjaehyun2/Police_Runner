"""연속 환경 에이전트 상태 및 물리 이동 모델.

AgentState 데이터클래스와 위치 갱신, 속도 클램핑, 방향 정규화 함수를 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pursuit_evasion_rl.continuous_env.road_map import RoadMap


@dataclass
class AgentState:
    """연속 환경에서의 에이전트 상태."""

    agent_id: str
    position: np.ndarray  # shape=(2,), [x, y]
    speed: float  # 0.0 ~ max_speed
    heading: float  # 0.0 ~ 2π (라디안)
    agent_type: str  # "police" | "fugitive"
    max_speed: float = 10.0


def normalize_heading(heading: float) -> float:
    """방향을 [0, 2π) 범위로 정규화한다."""
    heading = heading % (2.0 * np.pi)
    if heading < 0:
        heading += 2.0 * np.pi
    return float(heading)


def clamp_speed(speed: float, max_speed: float, segment_speed_limit: float) -> float:
    """속도를 [0, min(max_speed, segment_speed_limit)] 범위로 클램핑한다."""
    effective_max = min(max_speed, segment_speed_limit)
    return float(max(0.0, min(speed, effective_max)))


def update_position(
    state: AgentState, action: np.ndarray, dt: float, road_map: RoadMap
) -> AgentState:
    """에이전트의 위치를 물리 법칙에 따라 갱신한다.

    Physics: new_pos = pos + speed * [cos(heading), sin(heading)] * dt
    도로를 벗어나면 가장 가까운 도로 위 지점으로 클램핑하고 속도를 0으로 설정한다.

    Args:
        state: 현재 에이전트 상태
        action: [speed, heading] 2D 행동 벡터
        dt: 시뮬레이션 시간 스텝
        road_map: 도로 맵 객체

    Returns:
        갱신된 AgentState
    """
    # 행동에서 속도와 방향 추출
    desired_speed = float(action[0])
    desired_heading = normalize_heading(float(action[1]))

    # 현재 세그먼트의 속도 제한 적용
    current_segment = road_map.get_segment_at(tuple(state.position))
    segment_speed_limit = current_segment.speed_limit if current_segment else state.max_speed
    actual_speed = clamp_speed(desired_speed, state.max_speed, segment_speed_limit)

    # 위치 갱신: new_pos = pos + speed * [cos(heading), sin(heading)] * dt
    displacement = actual_speed * np.array(
        [np.cos(desired_heading), np.sin(desired_heading)], dtype=np.float32
    ) * dt
    new_pos = state.position + displacement

    # 도로 위 확인
    tolerance = road_map._config.get("segment_width", 3.0) if road_map._config else 3.0
    if road_map.is_on_road(tuple(new_pos), tolerance=tolerance):
        return AgentState(
            agent_id=state.agent_id,
            position=new_pos.astype(np.float32),
            speed=actual_speed,
            heading=desired_heading,
            agent_type=state.agent_type,
            max_speed=state.max_speed,
        )
    else:
        # 도로를 벗어난 경우: 가장 가까운 도로 위 지점으로 클램핑, 속도 0
        nearest = road_map.nearest_point_on_road(tuple(new_pos))
        return AgentState(
            agent_id=state.agent_id,
            position=np.array(nearest, dtype=np.float32),
            speed=0.0,
            heading=desired_heading,
            agent_type=state.agent_type,
            max_speed=state.max_speed,
        )
