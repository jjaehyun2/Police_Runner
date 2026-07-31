"""데모용 협력 추격/차단 경찰 정책 (InterceptPolicePolicy).

스펙의 BaselinePolicePolicy(페어드 비교용 단순 기준선, 테스트 고정)는 그대로 두고,
데모에서 "경찰차들이 도주자 주변으로 모여들며 탈출 경로를 차단"하는 모습을 보이도록
별도의 협력 정책을 제공한다.

핵심 아이디어
-------------
1. 탈출 경로 차단: 도주자가 향할 수 있는 경계(crosses_boundary) 진입 교차로를,
   도주자 기준 방향 최단거리가 가까운 순으로 뽑는다(= 도주자가 실제로 쓸 확률이 높은
   탈출구). 각 탈출구에 그곳까지 가장 빨리 갈 수 있는 경찰을 배정해 길목을 막는다.
2. 직접 추격: 배정되지 않은 나머지 경찰은 도주자의 현재 위치로 직접 수렴한다.
3. 절대 멈추지 않음: 목표까지 방향(directed) 경로가 없으면 정지 대신 목표에 유클리드
   거리로 가장 가까워지는 출구를 골라 전진한다(일방통행으로 인한 경찰 정지 문제 해결).

정책은 스펙의 build_action_mask / decode_recommendation 계약을 그대로 준수하므로
러너/메트릭/렌더러와 그대로 호환된다.
"""

from __future__ import annotations

import heapq
import math
from typing import Mapping, Sequence

import numpy as np

from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    DomainValidationError,
    ModelNetwork,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.policies import (
    ACTION_DIM,
    STAY_ACTION,
    ActionRecommendation,
    build_action_mask,
    decode_recommendation,
)

INTERCEPT_PROFILE_ID = "demo_intercept_coordinated"
_INF = float("inf")


class InterceptPolicePolicy:
    """협력 추격/차단 경찰 정책 (데모 전용, 비학습).

    Parameters
    ----------
    network:
        검증된 Model_Network.
    block_ratio:
        6대 중 탈출 경로 차단에 배정할 최대 비율(기본 0.5 → 최대 3대가 길목 차단,
        나머지는 직접 추격).
    """

    def __init__(self, network: ModelNetwork, *, block_ratio: float = 0.5) -> None:
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError("INVALID_NETWORK", "A ModelNetwork instance is required")
        self.network = network
        self.profile = INTERCEPT_PROFILE_ID
        self.experimental = False
        self.police_count = POLICE_COUNT
        self._report_id = INTERCEPT_PROFILE_ID
        self._max_block = max(0, min(POLICE_COUNT, int(round(POLICE_COUNT * block_ratio))))

        self._segments = {s.id: s for s in network.segments}
        self._pos = {i.id: i.position_xy for i in network.intersections}
        # 방향 인접 리스트 (start -> end, 길이) 및 역방향 (end -> start, 길이)
        self._fwd: dict[int, list[tuple[int, float]]] = {i.id: [] for i in network.intersections}
        self._rev: dict[int, list[tuple[int, float]]] = {i.id: [] for i in network.intersections}
        for s in network.segments:
            self._fwd[s.start_id].append((s.end_id, float(s.length_m)))
            self._rev[s.end_id].append((s.start_id, float(s.length_m)))
        # 도주자가 진입할 수 있는 탈출구(경계 세그먼트의 시작 교차로)
        self._escape_entries = sorted({s.start_id for s in network.segments if s.crosses_boundary})
        self._fwd_cache: dict[int, dict[int, float]] = {}
        self._rev_cache: dict[int, dict[int, float]] = {}

    # ------------------------------------------------------------------
    # 거리 계산 (방향 그래프 Dijkstra)
    # ------------------------------------------------------------------
    def _dijkstra(self, source: int, adjacency: Mapping[int, list[tuple[int, float]]],
                  cache: dict[int, dict[int, float]]) -> dict[int, float]:
        cached = cache.get(source)
        if cached is not None:
            return cached
        dist: dict[int, float] = {source: 0.0}
        heap: list[tuple[float, int]] = [(0.0, source)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, _INF):
                continue
            for nbr, w in adjacency.get(node, ()):  # 방향 간선만
                cand = d + w
                if cand < dist.get(nbr, _INF):
                    dist[nbr] = cand
                    heapq.heappush(heap, (cand, nbr))
        cache[source] = dist
        return dist

    def _dist_from(self, source: int) -> dict[int, float]:
        """source에서 각 노드까지의 방향 최단거리 (추격/전진 목표 도달용)."""
        return self._dijkstra(source, self._fwd, self._fwd_cache)

    def _dist_to(self, target: int) -> dict[int, float]:
        """각 노드에서 target까지의 방향 최단거리 (역방향 Dijkstra)."""
        return self._dijkstra(target, self._rev, self._rev_cache)

    # ------------------------------------------------------------------
    # 위치 → 결정 교차로
    # ------------------------------------------------------------------
    def _decision_intersection(self, placement: VehiclePlacement) -> int:
        if placement.segment_id is not None:
            return int(self._segments[placement.segment_id].end_id)
        return int(placement.intersection_id)

    # ------------------------------------------------------------------
    # 출구 선택: 목표까지 방향거리 최소, 없으면 유클리드 거리로 전진
    # ------------------------------------------------------------------
    def _select_slot_toward(self, ordered: Sequence[int], target: int) -> int:
        dist_to_target = self._dist_to(target)
        target_xy = self._pos[target]

        best_slot = STAY_ACTION
        best_directed = _INF
        best_euclid_slot = STAY_ACTION
        best_euclid = _INF
        for slot, seg_id in enumerate(ordered):
            seg = self._segments[seg_id]
            end_xy = self._pos[seg.end_id]
            # 유클리드 fallback 후보 (항상 계산 → 절대 멈추지 않게)
            euclid = math.dist(end_xy, target_xy)
            if euclid < best_euclid:
                best_euclid = euclid
                best_euclid_slot = slot
            # 방향 도달 가능하면 우선
            downstream = dist_to_target.get(seg.end_id)
            if downstream is None:
                continue
            cost = float(seg.length_m) + downstream
            if cost < best_directed:
                best_directed = cost
                best_slot = slot

        if best_directed < _INF:
            return best_slot
        # 목표까지 방향 경로가 전혀 없으면 유클리드로라도 접근 (정지 방지)
        return best_euclid_slot

    # ------------------------------------------------------------------
    # 팀 목표 배정: 일부는 탈출구 차단, 나머지는 직접 추격
    # ------------------------------------------------------------------
    def _assign_targets(self, decision_ids: Sequence[int], fugitive_entry: int) -> list[int]:
        targets: list[int | None] = [None] * self.police_count

        # 도주자 기준 가까운 탈출구 = 가장 유력한 도주 경로
        block_targets: list[int] = []
        if self._escape_entries and self._max_block > 0:
            fug_dist = self._dist_from(fugitive_entry)
            reachable = [(fug_dist.get(e, _INF), e) for e in self._escape_entries]
            reachable = [(d, e) for d, e in reachable if d < _INF]
            reachable.sort()
            block_targets = [e for _, e in reachable[: self._max_block]]

        # 각 차단 목표에 '가장 빨리 도달 가능한' 미배정 경찰을 그리디로 배정
        assigned: set[int] = set()
        for target in block_targets:
            dist_to = self._dist_to(target)
            best_officer, best_cost = None, _INF
            for idx in range(self.police_count):
                if idx in assigned:
                    continue
                cost = dist_to.get(decision_ids[idx], _INF)
                if cost == _INF:
                    # 방향 도달 불가면 유클리드로 근사(그래도 배정은 함)
                    cost = math.dist(self._pos[decision_ids[idx]], self._pos[target]) + 1e6
                if cost < best_cost:
                    best_cost, best_officer = cost, idx
            if best_officer is not None:
                targets[best_officer] = target
                assigned.add(best_officer)

        # 나머지 경찰은 도주자를 직접 추격
        for idx in range(self.police_count):
            if targets[idx] is None:
                targets[idx] = fugitive_entry
        return [int(t) for t in targets]

    # ------------------------------------------------------------------
    # 팀 추천
    # ------------------------------------------------------------------
    def recommend(
        self,
        *,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_headings: Sequence[float | None] | None = None,
    ) -> tuple[ActionRecommendation, ...]:
        if len(police) != self.police_count:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT",
                f"Exactly {self.police_count} police placements are required",
                expected=self.police_count,
                actual=len(police),
            )

        fugitive_entry = self._decision_intersection(fugitive)
        decision_ids = [self._decision_intersection(police[i]) for i in range(self.police_count)]
        targets = self._assign_targets(decision_ids, fugitive_entry)

        recommendations: list[ActionRecommendation] = []
        for index in range(self.police_count):
            heading = None if incoming_headings is None else incoming_headings[index]
            decision_id = decision_ids[index]
            mask, ordered = build_action_mask(self.network, decision_id, heading)
            chosen = self._select_slot_toward(ordered, targets[index]) if ordered else STAY_ACTION
            probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
            probabilities[chosen] = 1.0
            recommendations.append(
                decode_recommendation(
                    agent_id=f"police_{index}",
                    network=self.network,
                    decision_intersection_id=decision_id,
                    mask=mask,
                    ordered_segment_ids=ordered,
                    probabilities=probabilities,
                    profile=self.profile,
                    compatibility_report_id=self._report_id,
                    action_index=chosen,
                )
            )
        return tuple(recommendations)


__all__ = ("INTERCEPT_PROFILE_ID", "InterceptPolicePolicy")
