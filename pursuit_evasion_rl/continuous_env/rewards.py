"""연속 환경 보상 계산 모듈.

종료 보상, shaping 보상, 협력 보상, 오프로드 페널티를 계산한다.
"""

from __future__ import annotations

import numpy as np

from pursuit_evasion_rl.continuous_env.agent_model import AgentState
from pursuit_evasion_rl.continuous_env.road_map import RoadMap


class ContinuousRewardCalculator:
    """연속 환경 보상 계산기.

    보상 구성:
        - Terminal: capture +1/-1, escape +1/-1, timeout -0.3/-0.8
        - Shaping police: 도망자에 대한 거리 감소 * scale, |val| <= 0.1
        - Shaping fugitive: 최근접 출구에 대한 거리 감소 * scale, |val| <= 0.1
        - Cooperation: 3+ 경찰이 120도+ 간격으로 접근 시 0.05
        - Off-road penalty: -0.02
    """

    def __init__(self, road_map: RoadMap, config: dict | None = None) -> None:
        self.road_map = road_map
        self._config = config or {}
        self._shaping_scale = self._config.get("shaping_scale", 0.01)
        self._boundary_exits = road_map.get_boundary_exits()

    def compute_rewards(
        self,
        all_states: dict[str, AgentState],
        prev_states: dict[str, AgentState],
        termination_result: str | None,
        invalid_moves: dict[str, bool] | None = None,
    ) -> dict[str, float]:
        """모든 에이전트의 보상을 계산한다.

        Args:
            all_states: 현재 에이전트 상태
            prev_states: 이전 에이전트 상태
            termination_result: "police_capture" | "fugitive_escape" | "timeout" | None
            invalid_moves: 에이전트별 오프로드 여부

        Returns:
            에이전트 ID → 보상 값 딕셔너리
        """
        rewards: dict[str, float] = {aid: 0.0 for aid in all_states}

        if invalid_moves is None:
            invalid_moves = {}

        # 1. 종료 보상
        if termination_result == "police_capture":
            for aid, state in all_states.items():
                if state.agent_type == "police":
                    rewards[aid] += 1.0
                else:
                    rewards[aid] += -1.0
        elif termination_result == "fugitive_escape":
            for aid, state in all_states.items():
                if state.agent_type == "fugitive":
                    rewards[aid] += 1.0
                else:
                    rewards[aid] += -1.0
        elif termination_result == "timeout":
            for aid, state in all_states.items():
                if state.agent_type == "police":
                    rewards[aid] += -0.8
                else:
                    rewards[aid] += -0.3

        # 2. Shaping 보상 (종료하지 않은 경우에만)
        if termination_result is None:
            fugitive_state = None
            prev_fugitive_state = None
            for aid, state in all_states.items():
                if state.agent_type == "fugitive":
                    fugitive_state = state
                    prev_fugitive_state = prev_states.get(aid)
                    break

            # 경찰 shaping: 도망자에 대한 거리 감소
            if fugitive_state is not None:
                fug_pos = fugitive_state.position
                for aid, state in all_states.items():
                    if state.agent_type == "police" and aid in prev_states:
                        prev_dist = float(np.linalg.norm(prev_states[aid].position - prev_states.get(
                            fugitive_state.agent_id, fugitive_state
                        ).position))
                        curr_dist = float(np.linalg.norm(state.position - fug_pos))
                        shaping = (prev_dist - curr_dist) * self._shaping_scale
                        shaping = float(np.clip(shaping, -0.1, 0.1))
                        rewards[aid] += shaping

            # 도망자 shaping: 최근접 출구에 대한 거리 감소
            if fugitive_state is not None and prev_fugitive_state is not None and self._boundary_exits:
                prev_exit_dist = min(
                    float(np.hypot(
                        prev_fugitive_state.position[0] - ex[0],
                        prev_fugitive_state.position[1] - ex[1],
                    ))
                    for ex in self._boundary_exits
                )
                curr_exit_dist = min(
                    float(np.hypot(
                        fugitive_state.position[0] - ex[0],
                        fugitive_state.position[1] - ex[1],
                    ))
                    for ex in self._boundary_exits
                )
                shaping = (prev_exit_dist - curr_exit_dist) * self._shaping_scale
                shaping = float(np.clip(shaping, -0.1, 0.1))
                rewards[fugitive_state.agent_id] += shaping

            # 3. 협력 보상: 3+ 경찰이 120도+ 간격으로 접근
            if fugitive_state is not None:
                coop_bonus = self._compute_cooperation_bonus(all_states, fugitive_state)
                if coop_bonus > 0:
                    for aid, state in all_states.items():
                        if state.agent_type == "police":
                            rewards[aid] += coop_bonus

        # 4. 오프로드 페널티
        for aid, is_invalid in invalid_moves.items():
            if is_invalid and aid in rewards:
                rewards[aid] += -0.02

        return rewards

    def _compute_cooperation_bonus(
        self,
        all_states: dict[str, AgentState],
        fugitive_state: AgentState,
    ) -> float:
        """포위 협력 보상을 계산한다.

        도망자 기준으로 3대 이상의 경찰이 120도 이상 간격으로 접근하면 0.05 반환.
        """
        fug_pos = fugitive_state.position

        # 경찰들의 도망자 기준 각도 계산
        angles: list[float] = []
        for aid, state in all_states.items():
            if state.agent_type == "police":
                diff = state.position - fug_pos
                angle = float(np.arctan2(diff[1], diff[0]))
                if angle < 0:
                    angle += 2.0 * np.pi
                angles.append(angle)

        if len(angles) < 3:
            return 0.0

        # 각도 정렬 후 인접 각도 차이 확인
        angles.sort()
        n = len(angles)

        # 3개 이상의 경찰이 120도(2π/3) 이상 간격으로 분포하는지 확인
        # 최소 각도 간격이 120도 이상인 3개의 조합이 있는지 확인
        min_separation = 2.0 * np.pi / 3.0  # 120 degrees

        # 간단한 접근: 정렬된 각도에서 인접 간격 중 최대 간격이 충분히 작은지 확인
        # 즉, 모든 경찰이 360도를 120도 이내 간격으로 커버하는지
        gaps: list[float] = []
        for i in range(n):
            gap = angles[(i + 1) % n] - angles[i]
            if gap < 0:
                gap += 2.0 * np.pi
            gaps.append(gap)

        # 최대 갭이 (360 - 120*(n경찰-1)) 이하이면 포위 성공
        # 더 간단한 조건: 최대 갭 <= 360 - 120 = 240도 (2대일때), 120도(3대 이상)
        # 3+ 경찰이 있고, 최대 갭 < 240도 (=4π/3)이면 포위로 판정
        max_gap = max(gaps) if gaps else 2.0 * np.pi
        if n >= 3 and max_gap <= (2.0 * np.pi - min_separation):
            return 0.05

        return 0.0
