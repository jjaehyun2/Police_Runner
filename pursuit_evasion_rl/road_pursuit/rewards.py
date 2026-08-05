"""도로 추격 환경 보상 계산 모듈.

RoadRewardCalculator 클래스를 제공한다.
이산 환경과 동일한 보상 구조를 사용하여 기존 학습 파이프라인과 호환된다.
"""

from __future__ import annotations

import math
from typing import Any

from pursuit_evasion_rl.road_pursuit.road_network_2d import RoadNetwork2D
from pursuit_evasion_rl.road_pursuit.vehicle_model import VehicleState


class RoadRewardCalculator:
    """도로 추격 환경 보상 계산 클래스.

    보상 구조:
    - 경찰 체포 (capture_radius 이내): police +1.0, fugitive -1.0
    - 도망자 탈출 (경계 교차로 도달): fugitive +1.0, police -1.0
    - 타임아웃: police -0.3, fugitive -0.8
    - Shaping: 최단 경로 홉 수 감소량 * 0.1, clamped ±0.1
    - 협력: 3+ 경찰이 도망자 교차로의 인접 교차로를 점유 →
      해당 인접 교차로를 점유한 경찰에게만, 에피소드당 1회 +0.05
    - 시간 페널티: 비종료 스텝마다 경찰에게 -0.01 (배회/왕복 억제)

    Attributes:
        network: 2D 도로 네트워크
        terminal_reward: 종료 보상 절대값
        shaping_scale: shaping 보상 스케일
        cooperation_bonus_value: 협력 보너스 값
        time_penalty: 비종료 스텝마다 경찰에게 더해지는 값 (음수 = 페널티)
        capture_radius: 체포 반경 (2D 유클리드 거리)
    """

    def __init__(self, network: RoadNetwork2D, config: dict[str, Any]) -> None:
        """초기화.

        Args:
            network: 2D 도로 네트워크
            config: 보상 설정 딕셔너리
                - terminal_reward: 종료 보상 (기본 1.0)
                - shaping_scale: shaping 스케일 (기본 0.1)
                - cooperation_bonus: 협력 보너스 (기본 0.05, 에피소드당 1회)
                - time_penalty: 스텝당 경찰 보상 가산값 (기본 -0.01, 음수가 페널티)
                - capture_radius: 체포 반경 (기본 15.0)
        """
        self.network = network
        self.terminal_reward: float = config.get("terminal_reward", 1.0)
        self.shaping_scale: float = config.get("shaping_scale", 0.1)
        self.cooperation_bonus_value: float = config.get("cooperation_bonus", 0.05)
        self.time_penalty: float = config.get("time_penalty", -0.01)
        self.capture_radius: float = config.get("capture_radius", 15.0)

        # 에피소드 상태: 협력 보너스는 에피소드당 1회만 지급된다.
        self._cooperation_awarded: bool = False

    def reset(self) -> None:
        """에피소드 단위 상태를 초기화한다.

        환경의 reset()에서 호출되어야 한다. 호출하지 않으면 협력 보너스가
        이전 에피소드에서 이미 지급된 것으로 남아 다시 지급되지 않는다.
        """
        self._cooperation_awarded = False

    def compute_step_rewards(
        self,
        vehicle_states: dict[str, VehicleState],
        prev_vehicle_states: dict[str, VehicleState],
        termination_result: str | None,
    ) -> dict[str, float]:
        """스텝별 보상을 통합 계산한다.

        Args:
            vehicle_states: 현재 차량 상태
            prev_vehicle_states: 이전 차량 상태
            termination_result: 종료 결과 ("police_win", "fugitive_win", "timeout", None)

        Returns:
            에이전트별 보상 딕셔너리
        """
        rewards: dict[str, float] = {aid: 0.0 for aid in vehicle_states}

        if termination_result is not None:
            terminal = self._compute_terminal_rewards(vehicle_states, termination_result)
            for aid in rewards:
                rewards[aid] += terminal.get(aid, 0.0)
        else:
            # Shaping rewards
            for aid, state in vehicle_states.items():
                if aid.startswith("police"):
                    shaping = self._compute_shaping_police(
                        aid, vehicle_states, prev_vehicle_states
                    )
                elif aid == "fugitive":
                    shaping = self._compute_shaping_fugitive(
                        vehicle_states, prev_vehicle_states
                    )
                else:
                    shaping = 0.0
                rewards[aid] += shaping

            # 시간 페널티: 비종료 스텝마다 경찰에게 부과 (배회/왕복 억제)
            if self.time_penalty != 0.0:
                for aid in vehicle_states:
                    if aid.startswith("police"):
                        rewards[aid] += self.time_penalty

            # 협력 보너스: 에피소드당 1회, 실제 인접 교차로 점유자에게만 지급
            if not self._cooperation_awarded:
                occupants = self._get_cooperation_occupants(vehicle_states)
                if occupants:
                    self._cooperation_awarded = True
                    for aid in occupants:
                        rewards[aid] += self.cooperation_bonus_value

        return rewards

    def _compute_terminal_rewards(
        self,
        vehicle_states: dict[str, VehicleState],
        termination_result: str,
    ) -> dict[str, float]:
        """종료 보상을 계산한다."""
        terminal: dict[str, float] = {}

        if termination_result == "police_win":
            for aid in vehicle_states:
                if aid.startswith("police"):
                    terminal[aid] = self.terminal_reward
                elif aid == "fugitive":
                    terminal[aid] = -self.terminal_reward

        elif termination_result == "fugitive_win":
            for aid in vehicle_states:
                if aid.startswith("police"):
                    terminal[aid] = -self.terminal_reward
                elif aid == "fugitive":
                    terminal[aid] = self.terminal_reward

        elif termination_result == "timeout":
            for aid in vehicle_states:
                if aid.startswith("police"):
                    terminal[aid] = -0.3
                elif aid == "fugitive":
                    terminal[aid] = -0.8

        return terminal

    def _compute_shaping_police(
        self,
        agent_id: str,
        vehicle_states: dict[str, VehicleState],
        prev_vehicle_states: dict[str, VehicleState],
    ) -> float:
        """경찰의 shaping 보상: 도망자에 대한 최단 홉 감소량.

        potential 기반: 도망자는 양쪽 항 모두 '현재' 교차로로 고정하고,
        경찰 자신의 이전/현재 교차로만 비교한다. 이렇게 해야
        (a) 도망자의 이동이 모든 경찰에게 행동과 무관한 노이즈를 주지 않고,
        (b) A→B→A 왕복이 정확히 0으로 상쇄되어 공짜 왕복 이득이 사라진다.

        Returns:
            shaping 보상 (clamped ±0.1). 경로가 없으면 0.0.
        """
        fugitive_state = vehicle_states.get("fugitive")
        prev_police = prev_vehicle_states.get(agent_id)

        if fugitive_state is None or prev_police is None:
            return 0.0

        curr_police_inter = vehicle_states[agent_id].nearest_intersection(self.network)
        prev_police_inter = prev_police.nearest_intersection(self.network)
        curr_fug_inter = fugitive_state.nearest_intersection(self.network)

        prev_dist = self.network.shortest_path_hops(prev_police_inter, curr_fug_inter)
        curr_dist = self.network.shortest_path_hops(curr_police_inter, curr_fug_inter)

        # 도달 불가 sentinel(num_intersections)끼리 빼면 임의의 ±0.1 임펄스가
        # 생기므로, 한쪽이라도 도달 불가면 shaping을 주지 않는다.
        if self._is_unreachable(prev_dist) or self._is_unreachable(curr_dist):
            return 0.0

        raw = (prev_dist - curr_dist) * self.shaping_scale
        return max(-0.1, min(0.1, raw))

    def _is_unreachable(self, hops: int) -> bool:
        """홉 수가 RoadNetwork2D의 '경로 없음' sentinel인지 판정한다."""
        return hops >= self.network.num_intersections

    def _compute_shaping_fugitive(
        self,
        vehicle_states: dict[str, VehicleState],
        prev_vehicle_states: dict[str, VehicleState],
    ) -> float:
        """도망자의 shaping 보상: 경계 교차로까지 최단 홉 감소량.

        Returns:
            shaping 보상 (clamped ±0.1)
        """
        fugitive_state = vehicle_states.get("fugitive")
        prev_fugitive = prev_vehicle_states.get("fugitive")

        if fugitive_state is None or prev_fugitive is None:
            return 0.0

        boundary = self.network.get_boundary_intersections()
        if not boundary:
            return 0.0

        curr_inter = fugitive_state.nearest_intersection(self.network)
        prev_inter = prev_fugitive.nearest_intersection(self.network)

        prev_dist = min(
            self.network.shortest_path_hops(prev_inter, b) for b in boundary
        )
        curr_dist = min(
            self.network.shortest_path_hops(curr_inter, b) for b in boundary
        )

        # 어느 한쪽이라도 모든 경계 교차로에 도달 불가면 sentinel 차이가
        # 무의미하므로 shaping을 주지 않는다.
        if self._is_unreachable(prev_dist) or self._is_unreachable(curr_dist):
            return 0.0

        raw = (prev_dist - curr_dist) * self.shaping_scale
        return max(-0.1, min(0.1, raw))

    def _compute_cooperation_bonus(
        self, vehicle_states: dict[str, VehicleState]
    ) -> float:
        """협력 포위(3+ 인접 교차로 점유)가 성립하면 보너스 값을 반환한다.

        지급 대상 판정은 _get_cooperation_occupants()가 담당한다.

        Returns:
            cooperation_bonus 또는 0.0
        """
        if self._get_cooperation_occupants(vehicle_states):
            return self.cooperation_bonus_value
        return 0.0

    def _get_cooperation_occupants(
        self, vehicle_states: dict[str, VehicleState]
    ) -> set[str]:
        """협력 보너스를 받을 경찰 ID 집합을 반환한다.

        도망자 교차로의 인접 교차로 중 3곳 이상이 경찰에게 점유된 경우에만
        포위가 성립하며, 그 인접 교차로에 실제로 위치한 경찰만 대상이 된다.
        (포위에 참여하지 않은 원거리 경찰은 받지 않는다.)

        Returns:
            보너스 대상 경찰 ID 집합. 포위 미성립이면 빈 집합.
        """
        fugitive_state = vehicle_states.get("fugitive")
        if fugitive_state is None:
            return set()

        fugitive_inter = fugitive_state.nearest_intersection(self.network)

        # 도망자 교차로의 outgoing 목적지 교차로 목록
        outgoing = self.network.get_outgoing_segments(fugitive_inter)
        adjacent_intersections = set()
        for seg in outgoing:
            adjacent_intersections.add(seg.end_intersection_id)

        # incoming 세그먼트의 출발 교차로도 인접으로 간주
        intersection = self.network.get_intersection(fugitive_inter)
        for sid in intersection.incoming_segments:
            seg = self.network.get_segment(sid)
            adjacent_intersections.add(seg.start_intersection_id)

        if not adjacent_intersections:
            return set()

        # 인접 교차로에 실제로 서 있는 경찰과, 그들이 점유한 교차로 집합
        occupants: set[str] = set()
        occupied_intersections: set[int] = set()
        for aid, state in vehicle_states.items():
            if not aid.startswith("police"):
                continue
            inter = state.nearest_intersection(self.network)
            if inter in adjacent_intersections:
                occupants.add(aid)
                occupied_intersections.add(inter)

        if len(occupied_intersections) >= 3:
            return occupants

        return set()

    def check_capture(self, vehicle_states: dict[str, VehicleState]) -> bool:
        """경찰이 도망자를 체포했는지 확인한다.

        capture_radius 이내에 경찰이 있으면 True.

        Args:
            vehicle_states: 전체 차량 상태

        Returns:
            체포 여부
        """
        fugitive_state = vehicle_states.get("fugitive")
        if fugitive_state is None:
            return False

        fug_pos = fugitive_state.position_2d(self.network)

        for aid, state in vehicle_states.items():
            if aid.startswith("police"):
                police_pos = state.position_2d(self.network)
                dx = fug_pos[0] - police_pos[0]
                dy = fug_pos[1] - police_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= self.capture_radius:
                    return True

        return False

    def check_escape(self, vehicle_states: dict[str, VehicleState]) -> bool:
        """도망자가 경계 교차로에 도달했는지 확인한다.

        Returns:
            탈출 여부
        """
        fugitive_state = vehicle_states.get("fugitive")
        if fugitive_state is None:
            return False

        boundary = self.network.get_boundary_intersections()
        if not boundary:
            return False

        # 도망자가 교차로에 도달했고, 그 교차로가 경계인 경우
        if fugitive_state.at_intersection:
            end_inter = fugitive_state.end_intersection(self.network)
            if end_inter in boundary:
                return True

        return False
