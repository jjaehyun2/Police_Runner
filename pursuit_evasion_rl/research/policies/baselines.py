"""Audited deterministic pursuit baselines and their immutable schemas.

The implementations migrated from :mod:`demo_pursuit`.  This package module is
the sole implementation owner; the root module is a warning compatibility shim
for legacy imports and pickle globals.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from pursuit_evasion_rl.osm_demo.metrics import placement_position
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
    BaselinePolicePolicy,
    build_action_mask,
    decode_recommendation,
)
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.domain import ExecutionStatus, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.variants.observations import (
    OBSERVATION_21D_DIM,
    OBSERVATION_28D_DIM,
    actor_capacity,
    find_capacity_matched_hidden_width,
)

_INF = float("inf")
BASELINE_SCHEMA_VERSION = "1.0"
_DIRECT_PURSUIT = "direct_pursuit"
_CORDON = "cordon"


def _positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise DomainValidationError(
            "INVALID_BASELINE_PARAMETER", f"{name} must be positive and finite", path=name, actual=value
        )
    return numeric


def _finite_tuple(values, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise DomainValidationError(
            "INVALID_BASELINE_PARAMETER", f"{name} must contain finite values", path=name, actual=values
        )
    return result


@dataclass(frozen=True, slots=True)
class GoalEvaderParameters:
    """Complete, immutable parameter contract for :class:`GoalEvader`."""

    vision_range_m: float = 150.0
    replan_every: int = 8
    safe_band_m: float = 700.0
    safe_goal_distance_weight: float = 0.1
    forward_distance_weight: float = 0.2
    reverse_penalty_m: float = 400.0
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("vision_range_m", "safe_band_m", "reverse_penalty_m"):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        for name in ("safe_goal_distance_weight", "forward_distance_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise DomainValidationError(
                    "INVALID_BASELINE_PARAMETER", f"{name} must be finite and nonnegative", path=name, actual=value
                )
            object.__setattr__(self, name, value)
        if isinstance(self.replan_every, bool) or not isinstance(self.replan_every, int) or self.replan_every <= 0:
            raise DomainValidationError(
                "INVALID_BASELINE_PARAMETER", "replan_every must be a positive integer",
                path="replan_every", actual=self.replan_every,
            )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class EncirclementParameters:
    """Complete, immutable parameter contract for the audited encirclement baseline."""

    fugitive_speed_mps: float = 10.0
    police_speed_mps: float = 14.0
    lead_time_s: float = 25.0
    lead_stagger_s: tuple[float, ...] = (6.0, 14.0, 22.0)
    lateral_offsets_m: tuple[float, ...] = (0.0, 120.0, -120.0)
    fan_degrees: tuple[float, ...] = (0.0, 25.0, -25.0, 50.0, -50.0, 80.0, -80.0)
    intercept_margin: float = 0.9
    reverse_penalty_m: float = 300.0
    velocity_history_size: int = 6
    cordon_ring_min_m: float = 120.0
    cordon_ring_max_m: float = 380.0
    cordon_degrees: tuple[float, ...] = (0.0, 55.0, -55.0, 110.0, -110.0)
    close_pursuit_m: float = 220.0
    direct_pursuit_count: int = 2
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "fugitive_speed_mps", "police_speed_mps", "lead_time_s", "intercept_margin",
            "reverse_penalty_m", "cordon_ring_min_m", "cordon_ring_max_m", "close_pursuit_m",
        ):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        for name in ("lead_stagger_s", "lateral_offsets_m", "fan_degrees", "cordon_degrees"):
            object.__setattr__(self, name, _finite_tuple(getattr(self, name), name))
        if self.cordon_ring_min_m >= self.cordon_ring_max_m:
            raise DomainValidationError(
                "INVALID_BASELINE_PARAMETER", "cordon ring minimum must be below maximum",
                path="cordon_ring_min_m", actual=self.cordon_ring_min_m,
            )
        for name in ("velocity_history_size", "direct_pursuit_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DomainValidationError(
                    "INVALID_BASELINE_PARAMETER", f"{name} must be a positive integer", path=name, actual=value
                )
        if self.direct_pursuit_count > POLICE_COUNT:
            raise DomainValidationError(
                "INVALID_BASELINE_PARAMETER", "direct_pursuit_count exceeds police count",
                path="direct_pursuit_count", expected=POLICE_COUNT, actual=self.direct_pursuit_count,
            )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class GoalAssignment:
    """One officer's deterministic role/target assignment at a decision epoch."""

    officer_id: int
    proximity_rank: int
    decision_intersection_id: int
    target_intersection_id: int
    role: str
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0 <= self.officer_id < POLICE_COUNT or not 0 <= self.proximity_rank < POLICE_COUNT:
            raise DomainValidationError("INVALID_GOAL_ASSIGNMENT", "Officer and rank must be in the six-officer domain")
        if self.decision_intersection_id < 0 or self.target_intersection_id < 0:
            raise DomainValidationError("INVALID_GOAL_ASSIGNMENT", "Assignment node IDs must be nonnegative")
        if self.role not in {_DIRECT_PURSUIT, _CORDON}:
            raise DomainValidationError(
                "INVALID_GOAL_ASSIGNMENT", "Unknown assignment role", expected=[_DIRECT_PURSUIT, _CORDON], actual=self.role
            )


@dataclass(frozen=True, slots=True)
class GoalAssignmentPlan:
    """Canonical six-officer assignment schema emitted by the baseline."""

    fugitive_intersection_id: int
    fugitive_bearing_rad: float
    assignments: tuple[GoalAssignment, ...]
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        bearing = float(self.fugitive_bearing_rad)
        if self.fugitive_intersection_id < 0 or not math.isfinite(bearing):
            raise DomainValidationError("INVALID_GOAL_ASSIGNMENT", "Plan fugitive node/bearing is invalid")
        object.__setattr__(self, "fugitive_bearing_rad", bearing)
        ids = [assignment.officer_id for assignment in self.assignments]
        ranks = [assignment.proximity_rank for assignment in self.assignments]
        if len(ids) != POLICE_COUNT or set(ids) != set(range(POLICE_COUNT)) or set(ranks) != set(range(POLICE_COUNT)):
            raise DomainValidationError(
                "INVALID_GOAL_ASSIGNMENT", "Plan must assign every officer and proximity rank exactly once"
            )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def make_interior_network(network: ModelNetwork) -> ModelNetwork:
    """경계 탈출 제거(내부 추격): capture/timeout만 남긴다."""
    segs = tuple(replace(s, crosses_boundary=False) for s in network.segments)
    inters = tuple(replace(i, boundary_kind=None) for i in network.intersections)
    return ModelNetwork(intersections=inters, segments=segs)


class _Graph:
    """방향 그래프 거리(양방향) 캐시. 네트워크 불변 → 시드 간 재사용."""

    def __init__(self, network: ModelNetwork, weight=None) -> None:
        """``weight`` maps a Segment to its edge cost (default: length_m in
        meters).  A travel-time weight (seconds) turns every cached distance
        into a travel time without touching the dijkstra itself."""
        self.network = network
        self.pos = {i.id: i.position_xy for i in network.intersections}
        self.seg = {s.id: s for s in network.segments}
        edge_cost = weight if weight is not None else (lambda s: float(s.length_m))
        self.fwd: dict[int, list[tuple[int, float]]] = {i.id: [] for i in network.intersections}
        self.rev: dict[int, list[tuple[int, float]]] = {i.id: [] for i in network.intersections}
        for s in network.segments:
            cost = float(edge_cost(s))
            self.fwd[s.start_id].append((s.end_id, cost))
            self.rev[s.end_id].append((s.start_id, cost))
        self._to: dict[int, dict[int, float]] = {}
        self._from: dict[int, dict[int, float]] = {}
        self._nodes = list(self.pos)
        self._xy = np.array([self.pos[n] for n in self._nodes], dtype=float)

    def _dijkstra(self, src: int, adj) -> dict[int, float]:
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, _INF):
                continue
            for nb, w in adj[n]:
                nd = d + w
                if nd < dist.get(nb, _INF):
                    dist[nb] = nd
                    heapq.heappush(heap, (nd, nb))
        return dist

    def dist_to(self, target: int) -> dict[int, float]:
        c = self._to.get(target)
        if c is None:
            c = self._dijkstra(target, self.rev)
            self._to[target] = c
        return c

    def dist_from(self, source: int) -> dict[int, float]:
        c = self._from.get(source)
        if c is None:
            c = self._dijkstra(source, self.fwd)
            self._from[source] = c
        return c

    def nearest_node(self, xy) -> int:
        d2 = ((self._xy[:, 0] - xy[0]) ** 2 + (self._xy[:, 1] - xy[1]) ** 2)
        return self._nodes[int(np.argmin(d2))]


# ---------------------------------------------------------------------------
# 도주자: 안전 지역 목표를 정해 일관되게 도주(진동 제거)
# ---------------------------------------------------------------------------
class GoalEvader:
    """현실적(시야 제한) 도주자.

    - 시야(vision_range_m) 안의 경찰만 인지한다(전지적 아님).
    - 보이는 경찰이 있으면 그들에게서 멀어지는 방향의 안전 노드로 도주(일관 커밋).
    - 아무도 안 보이면 현재 진행 방향(모멘텀)을 유지해 계속 도망친다(진동 없음).
    """

    def __init__(self, network, *, rng, graph: _Graph,
                 vision_range_m: float = 150.0, replan_every: int = 8,
                 safe_band_m: float = 700.0) -> None:
        self.g = graph
        self.rng = rng
        self.parameters = GoalEvaderParameters(
            vision_range_m=vision_range_m,
            replan_every=replan_every,
            safe_band_m=safe_band_m,
        )
        self.vision = self.parameters.vision_range_m
        self.replan_every = self.parameters.replan_every
        self.safe_band = self.parameters.safe_band_m
        self.goal: int | None = None
        self.since = 10 ** 9
        self.prev_node: int | None = None
        self.heading: float | None = None

    @property
    def parameter_hash(self) -> str:
        return self.parameters.content_hash

    def _safe_goal(self, fnode: int, visible_xy) -> int:
        """보이는 경찰들로부터 멀어지는, 도달 가능한 안전 노드."""
        d_from_f = self.g.dist_from(fnode)
        best, best_score = fnode, -_INF
        for node, df in d_from_f.items():
            if df > self.safe_band or df <= 1e-9:
                continue
            xy = self.g.pos[node]
            min_pol = min(math.dist(xy, q) for q in visible_xy)
            score = min_pol + self.parameters.safe_goal_distance_weight * df
            if score > best_score:
                best_score, best = score, node
        return best

    def _forward_goal(self, fnode: int) -> int:
        """진행 방향(모멘텀) 앞쪽의 도달 가능한 먼 노드 — 계속 도망치기 위함."""
        d_from_f = self.g.dist_from(fnode)
        here = self.g.pos[fnode]
        if self.heading is None:
            # 방향 정보 없으면 가장 먼 도달 노드
            return max(d_from_f, key=lambda n: d_from_f[n]) if d_from_f else fnode
        hx, hy = math.cos(self.heading), math.sin(self.heading)
        best, best_score = fnode, -_INF
        for node, df in d_from_f.items():
            if df <= 1e-9 or df > self.safe_band:
                continue
            xy = self.g.pos[node]
            fwd = (xy[0] - here[0]) * hx + (xy[1] - here[1]) * hy  # 진행방향 투영
            score = fwd + self.parameters.forward_distance_weight * df
            if score > best_score:
                best_score, best = score, node
        return best

    def env_provider(self, network, police_xy):
        captured = [(float(x), float(y)) for x, y in police_xy]

        def provider(agent_id, intersection_id, ordered, hop):
            if not ordered:
                return STAY_ACTION
            here = self.g.pos[intersection_id]
            visible = [q for q in captured if math.dist(here, q) <= self.vision]
            self.since += 1

            if visible:
                # 위협 인지 → 안전 방향으로 즉시(또는 주기적으로) 재계획
                if self.goal is None or self.since >= self.replan_every or self.goal == intersection_id:
                    self.goal = self._safe_goal(intersection_id, visible)
                    self.since = 0
            else:
                # 미인지 → 진행 방향 유지 목표(도착했거나 없을 때만 갱신)
                if self.goal is None or self.goal == intersection_id:
                    self.goal = self._forward_goal(intersection_id)
                    self.since = 0

            dist_to_goal = self.g.dist_to(self.goal)
            goal_xy = self.g.pos[self.goal]
            best, best_cost, eu_best, eu_cost = None, _INF, None, _INF
            for slot, seg_id in enumerate(ordered):
                end = self.g.seg[seg_id].end_id
                back = self.parameters.reverse_penalty_m if (self.prev_node is not None and end == self.prev_node) else 0.0
                eu = math.dist(self.g.pos[end], goal_xy) + back
                if eu < eu_cost:
                    eu_cost, eu_best = eu, slot
                dn = dist_to_goal.get(end)
                if dn is None:
                    continue
                cost = float(self.g.seg[seg_id].length_m) + dn + back
                if cost < best_cost:
                    best_cost, best = cost, slot
            slot = best if best is not None else eu_best
            seg = self.g.seg[ordered[slot]]
            if not seg.virtual:
                end_xy = self.g.pos[seg.end_id]
                self.heading = math.atan2(end_xy[1] - here[1], end_xy[0] - here[0])
                self.prev_node = intersection_id
            return slot

        return provider


# ---------------------------------------------------------------------------
# 경찰: 도주자보다 먼저 도달 가능한 길목을 다방향으로 선점(포위) + 직접 추격
# ---------------------------------------------------------------------------
class EncirclementPolice:
    def __init__(self, network, *, graph: _Graph, fugitive_speed_mps: float = 10.0,
                 police_speed_mps: float = 14.0, lead_time_s: float = 25.0,
                 lead_stagger_s=(6.0, 14.0, 22.0), lateral_offsets_m=(0.0, 120.0, -120.0),
                 fan_degrees=(0.0, 25.0, -25.0, 50.0, -50.0, 80.0, -80.0),
                 intercept_margin: float = 0.9) -> None:
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError("INVALID_NETWORK", "A ModelNetwork instance is required")
        self.network = network
        self.g = graph
        self.profile = "demo_encirclement_v2"
        self.experimental = False
        self.pc = POLICE_COUNT
        self.parameters = EncirclementParameters(
            fugitive_speed_mps=fugitive_speed_mps,
            police_speed_mps=police_speed_mps,
            lead_time_s=lead_time_s,
            lead_stagger_s=tuple(lead_stagger_s),
            lateral_offsets_m=tuple(lateral_offsets_m),
            fan_degrees=tuple(fan_degrees),
            intercept_margin=intercept_margin,
        )
        self.vf = self.parameters.fugitive_speed_mps
        self.vp = self.parameters.police_speed_mps
        self.lead = self.parameters.lead_time_s * self.parameters.fugitive_speed_mps
        self.lead_stagger = self.parameters.lead_stagger_s
        self.lateral = self.parameters.lateral_offsets_m
        self.fan = self.parameters.fan_degrees
        self.margin = self.parameters.intercept_margin
        self._prev = {k: None for k in range(POLICE_COUNT)}
        self._fug_hist: list[tuple[float, float]] = []  # 도주자 위치 이력(속도 추정)
        self.last_goal_assignment: GoalAssignmentPlan | None = None

    @property
    def parameter_hash(self) -> str:
        return self.parameters.content_hash

    def _decision(self, pl: VehiclePlacement) -> int:
        if pl.segment_id is not None:
            return int(self.g.seg[pl.segment_id].end_id)
        return int(pl.intersection_id)

    def _route(self, ordered, target, officer) -> int:
        dist_to = self.g.dist_to(target)
        txy = self.g.pos[target]
        prev = self._prev[officer]
        best, best_cost, eu_best, eu_cost = None, _INF, None, _INF
        for slot, seg_id in enumerate(ordered):
            end = self.g.seg[seg_id].end_id
            back = self.parameters.reverse_penalty_m if (prev is not None and end == prev) else 0.0
            eu = math.dist(self.g.pos[end], txy) + back
            if eu < eu_cost:
                eu_cost, eu_best = eu, slot
            dn = dist_to.get(end)
            if dn is None:
                continue
            c = float(self.g.seg[seg_id].length_m) + dn + back
            if c < best_cost:
                best_cost, best = c, slot
        return best if best is not None else (eu_best if eu_best is not None else STAY_ACTION)

    def _fug_velocity(self, fug_xy):
        """도주자 위치 이력으로 이동 방향(단위 벡터) 추정."""
        self._fug_hist.append(fug_xy)
        if len(self._fug_hist) > self.parameters.velocity_history_size:
            self._fug_hist.pop(0)
        if len(self._fug_hist) < 2:
            return None
        ax, ay = self._fug_hist[0]
        bx, by = self._fug_hist[-1]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        return (dx / norm, dy / norm)

    def _assign(self, decisions, fnode, police_xy, fug_xy):
        # 도주 방향: 이동 이력 기반(정확) → 없으면 경찰 무게중심 반대 방향
        vel = self._fug_velocity(fug_xy)
        if vel is None:
            cx = sum(p[0] for p in police_xy) / len(police_xy)
            cy = sum(p[1] for p in police_xy) / len(police_xy)
            b = math.atan2(fug_xy[1] - cy, fug_xy[0] - cx) if fug_xy != (cx, cy) else 0.0
            vel = (math.cos(b), math.sin(b))
        base = math.atan2(vel[1], vel[0])          # 도주 진행 방위

        # 경찰을 도주자 현재 위치까지 유클리드 근접 순으로 정렬
        order = sorted(range(self.pc), key=lambda k: math.dist(self.g.pos[decisions[k]], fug_xy))

        # 도주자 기준 '도로 거리' 링(코르돈) 후보 노드: 실제 도로 위에서 도주자를 감싼다.
        d_from_f = self.g.dist_from(fnode)
        ring = []
        for node, df in d_from_f.items():
            if self.parameters.cordon_ring_min_m <= df <= self.parameters.cordon_ring_max_m:
                xy = self.g.pos[node]
                bearing = math.atan2(xy[1] - fug_xy[1], xy[0] - fug_xy[0])
                ring.append((bearing, node))

        targets = [fnode] * self.pc
        assignments: list[GoalAssignment | None] = [None] * self.pc
        taken: set[int] = set()
        for rank, k in enumerate(order):
            here = self.g.pos[decisions[k]]
            # 가장 가까운 경찰 + 근접 경찰은 순수 추격으로 갭 봉쇄, 나머지는 코르돈
            if rank < self.parameters.direct_pursuit_count or math.dist(here, fug_xy) <= self.parameters.close_pursuit_m:
                assignments[k] = GoalAssignment(k, rank, decisions[k], fnode, _DIRECT_PURSUIT)
                continue
            want = base + math.radians(
                self.parameters.cordon_degrees[(rank - 1) % len(self.parameters.cordon_degrees)]
            )
            # 원하는 방위에 가장 가까운 미사용 링 노드를 배정(없으면 직접 추격)
            best_node, best_diff = None, _INF
            for bearing, node in ring:
                if node in taken:
                    continue
                diff = abs(math.atan2(math.sin(bearing - want), math.cos(bearing - want)))
                if diff < best_diff:
                    best_diff, best_node = diff, node
            if best_node is not None:
                taken.add(best_node)
                targets[k] = best_node
                assignments[k] = GoalAssignment(k, rank, decisions[k], best_node, _CORDON)
            else:
                assignments[k] = GoalAssignment(k, rank, decisions[k], fnode, _DIRECT_PURSUIT)
        self.last_goal_assignment = GoalAssignmentPlan(
            fugitive_intersection_id=fnode,
            fugitive_bearing_rad=base,
            assignments=tuple(assignment for assignment in assignments if assignment is not None),
        )
        return targets

    def recommend(self, *, police, fugitive, step, max_steps, incoming_headings=None):
        if len(police) != self.pc:
            raise DomainValidationError("INVALID_POLICE_COUNT", "6 police required",
                                        expected=self.pc, actual=len(police))
        decisions = [self._decision(police[k]) for k in range(self.pc)]
        fnode = self._decision(fugitive)
        police_xy = [self.g.pos[d] for d in decisions]
        fug_xy = self.g.pos[fnode]
        targets = self._assign(decisions, fnode, police_xy, fug_xy)

        recs = []
        for k in range(self.pc):
            heading = None if incoming_headings is None else incoming_headings[k]
            did = decisions[k]
            mask, ordered = build_action_mask(self.network, did, heading)
            chosen = self._route(ordered, targets[k], k) if ordered else STAY_ACTION
            if ordered and chosen != STAY_ACTION and not self.g.seg[ordered[chosen]].virtual:
                self._prev[k] = did
            probs = np.zeros(ACTION_DIM, dtype=np.float64)
            probs[chosen] = 1.0
            recs.append(decode_recommendation(
                agent_id=f"police_{k}", network=self.network, decision_intersection_id=did,
                mask=mask, ordered_segment_ids=ordered, probabilities=probs,
                profile=self.profile, compatibility_report_id=self.profile, action_index=chosen,
            ))
        return tuple(recs)


# 하위호환 별칭
SmartEvader = GoalEvader


# ===========================================================================
# Requirement 11: numeric baseline registry, fairness contracts, validation.
#
# This section owns the *declarative* side of Requirement 11.  It registers the
# minimum baseline set, records how each baseline's hyperparameters were chosen,
# accounts for each baseline's cost, and validates a stratum's result matrix.
# It deliberately does not train or evaluate anything: producing a trained
# checkpoint is the trainer's job, and the registry only needs the resulting
# budget/cost declarations to police fairness.
# ===========================================================================

GREEDY_INTERCEPT_PROFILE_ID = "baseline_greedy_intercept"
DEFAULT_ACTOR_HIDDEN_DIMS: tuple[int, ...] = (128, 128)

PROPOSED_POLICY_ID = "proposed_masked_mappo_28d"
LEGACY_SHARED_PPO_ID = "legacy_shared_ppo"
ENCIRCLEMENT_ID = "encirclement_police"
DIRECTED_SHORTEST_PATH_ID = "directed_shortest_path"
GREEDY_INTERCEPT_ID = "greedy_intercept"

# Requirement 11.1: always required in every primary evaluation stratum.
MINIMUM_REQUIRED_BASELINE_IDS: tuple[str, ...] = (
    PROPOSED_POLICY_ID,
    LEGACY_SHARED_PPO_ID,
    ENCIRCLEMENT_ID,
)
# Requirements 11.2-11.4: required only once the interface conformance check passes.
CONDITIONAL_BASELINE_IDS: tuple[str, ...] = (
    DIRECTED_SHORTEST_PATH_ID,
    GREEDY_INTERCEPT_ID,
)

# Requirement 11.7: the metric contract every baseline row must carry.
REQUIRED_METRIC_FIELDS: tuple[str, ...] = (
    "capture",
    "containment",
    "anti_oscillation",
    "idleness",
    "individual_contribution",
    "physical_plausibility",
)

# Requirement 11.6: hyperparameters may be selected on these splits only.
ALLOWED_SELECTION_SPLITS = frozenset({"train", "validation"})

POLICY_INTERFACE_CONTRACT = (
    "A conforming police-team policy exposes a non-empty ``profile`` string and a "
    "boolean ``experimental`` attribute, and implements "
    "``recommend(*, police, fugitive, step, max_steps, incoming_headings=None)`` "
    "returning exactly POLICE_COUNT ActionRecommendation values. Each returned "
    "recommendation must be ``valid`` and its ``action_index`` must be set in "
    "``build_action_mask(network, decision_intersection_id, incoming_heading)`` for "
    "the officer's own decision intersection, and two calls on an identical state "
    "must select identical actions."
)


def decision_intersection_id(network: ModelNetwork, placement: VehiclePlacement) -> int:
    """The intersection at which a placement's next decision is taken.

    A vehicle already on a segment commits to that arc and decides at its end;
    a vehicle at an intersection decides there.  This is the same rule used by
    :class:`EncirclementPolice` and :class:`BaselinePolicePolicy`, restated here
    so the conformance check can derive legality independently of the policy.
    """
    if placement.segment_id is not None:
        for segment in network.segments:
            if segment.id == placement.segment_id:
                return int(segment.end_id)
        raise DomainValidationError(
            "INVALID_POSITION", "Placement references an unknown segment", actual=placement.segment_id
        )
    return int(placement.intersection_id)


class GreedyInterceptPolice:
    """Deterministic greedy-intercept police baseline (Requirement 11.2).

    Every officer independently selects the legal exit whose destination
    intersection minimizes the straight-line distance to the fugitive's current
    position.  This is intentionally the *locally* greedy form: it uses no
    trajectory prediction and no coordination, which is what makes it a
    meaningful contrast against both :class:`BaselinePolicePolicy` (which
    minimizes directed *road* distance) and the coordinating
    :class:`EncirclementPolice`.  Ties keep the lowest action slot, so two runs
    on identical placements produce identical recommendations.
    """

    def __init__(
        self,
        network: ModelNetwork,
        *,
        profile: str = GREEDY_INTERCEPT_PROFILE_ID,
        police_count: int = POLICE_COUNT,
        report_id: str = GREEDY_INTERCEPT_PROFILE_ID,
    ) -> None:
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError("INVALID_NETWORK", "A ModelNetwork instance is required")
        self.network = network
        self.profile = str(profile)
        self.experimental = False
        self.police_count = int(police_count)
        self._report_id = str(report_id)
        self._segments = {segment.id: segment for segment in network.segments}
        self._positions = {item.id: item.position_xy for item in network.intersections}

    def _select_slot(self, ordered: Sequence[int], fugitive_xy: tuple[float, float]) -> int:
        best_slot, best_cost = STAY_ACTION, _INF
        for slot, segment_id in enumerate(ordered):
            destination = self._positions[self._segments[segment_id].end_id]
            cost = math.dist(destination, fugitive_xy)
            if cost < best_cost:  # strict keeps the lowest action slot on ties
                best_cost, best_slot = cost, slot
        return best_slot

    def recommend(
        self,
        *,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_headings: Sequence[float | None] | None = None,
    ) -> tuple[ActionRecommendation, ...]:
        """Produce a complete team recommendation or fail the whole request."""
        if len(police) != self.police_count:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT", f"Exactly {self.police_count} police placements are required",
                expected=self.police_count, actual=len(police),
            )
        if incoming_headings is not None and len(incoming_headings) != self.police_count:
            raise DomainValidationError(
                "INVALID_HEADINGS", "One incoming heading per officer is required when headings are supplied",
                expected=self.police_count, actual=len(incoming_headings),
            )
        fugitive_xy = placement_position(self.network, fugitive)
        recommendations = []
        for index in range(self.police_count):
            heading = None if incoming_headings is None else incoming_headings[index]
            decision_id = decision_intersection_id(self.network, police[index])
            mask, ordered = build_action_mask(self.network, decision_id, heading)
            chosen = self._select_slot(ordered, fugitive_xy)
            probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
            probabilities[chosen] = 1.0
            recommendations.append(decode_recommendation(
                agent_id=f"police_{index}", network=self.network, decision_intersection_id=decision_id,
                mask=mask, ordered_segment_ids=ordered, probabilities=probabilities,
                profile=self.profile, compatibility_report_id=self._report_id, action_index=chosen,
            ))
        return tuple(recommendations)


def directed_shortest_path_policy(network: ModelNetwork) -> BaselinePolicePolicy:
    """The audited directed shortest-path baseline (Requirement 11.2)."""
    return BaselinePolicePolicy(network)


def greedy_intercept_policy(network: ModelNetwork) -> GreedyInterceptPolice:
    """The greedy-intercept baseline (Requirement 11.2)."""
    return GreedyInterceptPolice(network)


# ---------------------------------------------------------------------------
# Common policy interface conformance (Requirements 11.2-11.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InterfaceCheck:
    """One named conformance probe and its outcome."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PolicyInterfaceReport:
    """Recorded conformance verdict for one candidate baseline (Requirement 11.2).

    A failing verdict is a *record*, never an omission: the caller registers the
    baseline as ``not_run`` carrying :attr:`failure_reason` (Requirement 11.4).
    """

    baseline_id: str
    contract: str
    checks: tuple[InterfaceCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_id, str) or not self.baseline_id.strip():
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "baseline_id must be non-empty", path="baseline_id"
            )
        object.__setattr__(self, "checks", tuple(self.checks))
        if not self.checks:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "a conformance report must contain at least one check", path="checks"
            )

    @property
    def conforms(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def status(self) -> ExecutionStatus:
        return ExecutionStatus.COMPLETED if self.conforms else ExecutionStatus.NOT_RUN

    @property
    def failure_reason(self) -> str:
        failed = [f"{check.name}: {check.detail}" for check in self.checks if not check.passed]
        return "; ".join(failed)

    @property
    def config_hash(self) -> str:
        return content_hash(self)


def check_policy_interface_conformance(
    baseline_id: str,
    policy: Any,
    *,
    network: ModelNetwork,
    police: Sequence[VehiclePlacement],
    fugitive: VehiclePlacement,
    step: int = 0,
    max_steps: int = 100,
) -> PolicyInterfaceReport:
    """Probe ``policy`` against :data:`POLICY_INTERFACE_CONTRACT` and record the verdict.

    This never raises for a non-conforming candidate -- classifying arbitrary
    candidate policies is the whole point, so any exception raised by the policy
    is captured as a failed check rather than propagated (Requirement 11.4).
    """
    checks: list[InterfaceCheck] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append(InterfaceCheck(name=name, passed=passed, detail=detail))

    profile = getattr(policy, "profile", None)
    record(
        "profile_attribute",
        isinstance(profile, str) and bool(profile.strip()),
        f"profile must be a non-empty string, got {profile!r}",
    )
    experimental = getattr(policy, "experimental", None)
    record(
        "experimental_attribute",
        isinstance(experimental, bool),
        f"experimental must be a bool, got {experimental!r}",
    )

    recommend = getattr(policy, "recommend", None)
    if not callable(recommend):
        record("recommend_callable", False, "policy exposes no callable recommend()")
        return PolicyInterfaceReport(baseline_id, POLICY_INTERFACE_CONTRACT, tuple(checks))
    record("recommend_callable", True)

    def invoke() -> tuple[ActionRecommendation, ...]:
        return recommend(police=police, fugitive=fugitive, step=step, max_steps=max_steps)

    try:
        first, second = invoke(), invoke()
    except Exception as exc:  # noqa: BLE001 - classification requires catching everything
        record("recommend_invocation", False, f"{type(exc).__name__}: {exc}")
        return PolicyInterfaceReport(baseline_id, POLICY_INTERFACE_CONTRACT, tuple(checks))
    record("recommend_invocation", True)

    cardinality_ok = len(first) == POLICE_COUNT
    record("team_cardinality", cardinality_ok, f"expected {POLICE_COUNT} recommendations, got {len(first)}")
    types_ok = all(isinstance(item, ActionRecommendation) for item in first)
    record("recommendation_type", types_ok, "every element must be an ActionRecommendation")

    if cardinality_ok and types_ok:
        illegal: list[str] = []
        for index, recommendation in enumerate(first):
            if not recommendation.valid:
                illegal.append(f"officer {index} returned an invalid recommendation")
                continue
            decision_id = decision_intersection_id(network, police[index])
            mask, _ = build_action_mask(network, decision_id, None)
            if not (0 <= recommendation.action_index < len(mask) and bool(mask[recommendation.action_index])):
                illegal.append(f"officer {index} chose masked action {recommendation.action_index}")
        record("legal_actions", not illegal, "; ".join(illegal))
        deterministic = [item.action_index for item in first] == [item.action_index for item in second]
        record("determinism", deterministic, "two identical calls selected different actions")

    return PolicyInterfaceReport(baseline_id, POLICY_INTERFACE_CONTRACT, tuple(checks))


# ---------------------------------------------------------------------------
# Cost, budget and selection schemas (Requirements 11.5, 11.6, 11.9)
# ---------------------------------------------------------------------------


class CostAvailability(str, Enum):
    """Why a cost field holds (or does not hold) a value.

    The three states are deliberately distinct.  Conflating them is exactly the
    Correctness Property 17 failure mode: a rule-based baseline genuinely has no
    training wall clock (:attr:`NOT_APPLICABLE`), which must never read the same
    as a learned policy whose wall clock nobody recorded (:attr:`NOT_MEASURED`).
    """

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    NOT_MEASURED = "not_measured"


@dataclass(frozen=True, slots=True)
class CostField:
    """One cost quantity plus the method that produced (or excused) it."""

    availability: CostAvailability
    unit: str
    value: float | None = None
    method: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "availability", CostAvailability(self.availability))
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ResearchValidationError("MISSING_REQUIRED_FIELD", "unit must be non-empty", path="unit")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ResearchValidationError(
                "MISSING_COST_METHOD",
                "every cost field requires a measurement method or an explicit reason",
                path="method",
            )
        if self.availability is CostAvailability.MEASURED:
            value = float(self.value) if self.value is not None else None
            if value is None or not math.isfinite(value) or value < 0.0:
                raise ResearchValidationError(
                    "INVALID_COST_VALUE", "a measured cost requires a finite nonnegative value",
                    path="value", actual=self.value,
                )
            object.__setattr__(self, "value", value)
        elif self.value is not None:
            raise ResearchValidationError(
                "INVALID_COST_VALUE", "only measured costs may carry a value",
                path="value", expected=None, actual=self.value,
            )


def measured_cost(value: float, unit: str, method: str) -> CostField:
    return CostField(CostAvailability.MEASURED, unit, value=value, method=method)


def not_applicable_cost(unit: str, reason: str) -> CostField:
    """A cost this policy kind cannot have -- distinct from an unrecorded one."""
    return CostField(CostAvailability.NOT_APPLICABLE, unit, method=reason)


def not_measured_cost(unit: str, reason: str) -> CostField:
    """A cost this policy *does* have but which nobody recorded."""
    return CostField(CostAvailability.NOT_MEASURED, unit, method=reason)


COST_FIELD_NAMES: tuple[str, ...] = (
    "train_env_steps",
    "train_wall_clock_s",
    "accelerator_time_s",
    "parameter_count",
    "flop_estimate",
    "inference_latency_per_episode_s",
)


@dataclass(frozen=True, slots=True)
class PolicyCostSchema:
    """The six cost quantities Requirement 11.9 requires for every policy."""

    train_env_steps: CostField
    train_wall_clock_s: CostField
    accelerator_time_s: CostField
    parameter_count: CostField
    flop_estimate: CostField
    inference_latency_per_episode_s: CostField
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in COST_FIELD_NAMES:
            if not isinstance(getattr(self, name), CostField):
                raise ResearchValidationError(
                    "INVALID_COST_FIELD", f"{name} must be a CostField", path=name, actual=getattr(self, name)
                )

    @property
    def unrecorded_fields(self) -> tuple[str, ...]:
        return tuple(
            name for name in COST_FIELD_NAMES
            if getattr(self, name).availability is CostAvailability.NOT_MEASURED
        )

    @property
    def config_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class TrainingBudget:
    """The training budget Requirement 11.5 requires learned policies to share."""

    env_steps: int
    optimizer_updates: int
    resource_ceiling_id: str
    model_selection_data: str
    checkpoint_selection_rule: str
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("env_steps", "optimizer_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ResearchValidationError(
                    "INVALID_TRAINING_BUDGET", f"{name} must be a positive integer", path=name, actual=value
                )
        for name in ("resource_ceiling_id", "model_selection_data", "checkpoint_selection_rule"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ResearchValidationError(
                    "INVALID_TRAINING_BUDGET", f"{name} must be non-empty", path=name
                )

    @property
    def budget_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class HyperparameterSelection:
    """Declared search space and selection rule provenance (Requirement 11.6)."""

    search_space: str
    selection_rule: str
    selection_data_splits: tuple[str, ...]
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("search_space", "selection_rule"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ResearchValidationError(
                    "MISSING_REQUIRED_FIELD", f"{name} must be non-empty", path=name
                )
        splits = tuple(str(item) for item in self.selection_data_splits)
        if not splits:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "selection_data_splits must not be empty", path="selection_data_splits"
            )
        leaked = sorted(set(splits) - ALLOWED_SELECTION_SPLITS)
        if leaked:
            raise ResearchValidationError(
                "SELECTION_DATA_LEAKAGE",
                "baseline hyperparameters may only be selected on train/validation data",
                path="selection_data_splits", expected=sorted(ALLOWED_SELECTION_SPLITS), actual=leaked,
            )
        object.__setattr__(self, "selection_data_splits", splits)


FIXED_HEURISTIC_SELECTION = HyperparameterSelection(
    search_space="none: audited constants frozen in the baseline parameter dataclass",
    selection_rule="no search performed; parameters are fixed by the audited baseline contract",
    selection_data_splits=("train",),
)


# ---------------------------------------------------------------------------
# Registration schema (Requirements 11.1-11.6, 11.9)
# ---------------------------------------------------------------------------


class PolicyKind(str, Enum):
    LEARNED = "learned"
    RULE_BASED = "rule_based"


class BaselineRequirement(str, Enum):
    """Whether a baseline is unconditionally required in every stratum."""

    MINIMUM_REQUIRED = "minimum_required"
    CONDITIONAL_ON_CONFORMANCE = "conditional_on_conformance"


@dataclass(frozen=True, slots=True)
class BaselineRegistration:
    """One registered baseline's parameter, selection and cost contract."""

    baseline_id: str
    policy_kind: PolicyKind
    requirement: BaselineRequirement
    parameters: Mapping[str, Any]
    selection: HyperparameterSelection
    cost: PolicyCostSchema
    status: ExecutionStatus
    status_reason: str
    training_budget: TrainingBudget | None = None
    interface_report: PolicyInterfaceReport | None = None
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_id, str) or not self.baseline_id.strip():
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "baseline_id must be non-empty", path="baseline_id"
            )
        object.__setattr__(self, "policy_kind", PolicyKind(self.policy_kind))
        object.__setattr__(self, "requirement", BaselineRequirement(self.requirement))
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        if not isinstance(self.status_reason, str) or not self.status_reason.strip():
            raise ResearchValidationError(
                "MISSING_STATUS_REASON",
                "every registration records why it holds its execution status",
                path="status_reason",
            )
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        if self.policy_kind is PolicyKind.LEARNED and self.training_budget is None:
            raise ResearchValidationError(
                "MISSING_TRAINING_BUDGET",
                "learned policies must declare the budget they were trained under",
                path="training_budget",
            )
        if self.policy_kind is PolicyKind.RULE_BASED and self.training_budget is not None:
            raise ResearchValidationError(
                "UNEXPECTED_TRAINING_BUDGET",
                "rule-based baselines have no training budget to declare",
                path="training_budget",
            )

    @property
    def config_hash(self) -> str:
        return content_hash(self)


def _learned_cost(
    *,
    budget: TrainingBudget,
    observation_dim: int,
    hidden_dims: Sequence[int],
    wall_clock_s: float,
    accelerator_time_s: float,
    inference_latency_s: float,
    measurement_method: str,
) -> PolicyCostSchema:
    capacity = actor_capacity(
        observation_dim, action_dim=ACTION_DIM, num_officers=POLICE_COUNT, hidden_dims=hidden_dims
    )
    analytic = (
        f"analytic Linear/Tanh accounting over hidden_dims={tuple(hidden_dims)} and "
        f"input_dim={observation_dim}+{POLICE_COUNT} (research.variants.observations.actor_capacity)"
    )
    return PolicyCostSchema(
        train_env_steps=measured_cost(budget.env_steps, "environment_steps", "declared TrainingBudget.env_steps"),
        train_wall_clock_s=measured_cost(wall_clock_s, "seconds", measurement_method),
        accelerator_time_s=measured_cost(accelerator_time_s, "seconds", measurement_method),
        parameter_count=measured_cost(capacity.parameter_count, "parameters", analytic),
        flop_estimate=measured_cost(capacity.flop_estimate, "flops_per_forward", analytic),
        inference_latency_per_episode_s=measured_cost(inference_latency_s, "seconds_per_episode", measurement_method),
    )


def _rule_based_cost(*, inference_latency_s: float, measurement_method: str) -> PolicyCostSchema:
    no_training = "rule-based baseline: no training phase exists, so this cost cannot be incurred"
    no_parameters = "rule-based baseline: no learned parameters exist"
    return PolicyCostSchema(
        train_env_steps=not_applicable_cost("environment_steps", no_training),
        train_wall_clock_s=not_applicable_cost("seconds", no_training),
        accelerator_time_s=not_applicable_cost("seconds", no_training),
        parameter_count=not_applicable_cost("parameters", no_parameters),
        flop_estimate=not_applicable_cost("flops_per_forward", no_parameters),
        inference_latency_per_episode_s=measured_cost(
            inference_latency_s, "seconds_per_episode", measurement_method
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    """Measured runtime costs for one policy (Requirement 11.9).

    ``train_wall_clock_s``/``accelerator_time_s`` are ``None`` only for
    rule-based baselines, where :func:`numeric_baseline_registry` turns them into
    explicit ``NOT_APPLICABLE`` cost fields.  A learned policy that leaves them
    ``None`` is rejected rather than silently downgraded, so an unrecorded cost
    can never masquerade as an inapplicable one.
    """

    inference_latency_per_episode_s: float
    method: str
    train_wall_clock_s: float | None = None
    accelerator_time_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ResearchValidationError(
                "MISSING_COST_METHOD", "runtime measurements require a measurement method", path="method"
            )


def numeric_baseline_registry(
    *,
    budget: TrainingBudget,
    measurements: Mapping[str, RuntimeMeasurement],
    conformance: Mapping[str, PolicyInterfaceReport] | None = None,
    proposed_observation_dim: int = OBSERVATION_28D_DIM,
    legacy_observation_dim: int = OBSERVATION_21D_DIM,
    hidden_dims: Sequence[int] = DEFAULT_ACTOR_HIDDEN_DIMS,
) -> dict[str, BaselineRegistration]:
    """Build the primary baseline registry for one Research_Protocol.

    ``budget`` is shared verbatim by the proposed policy and ``legacy_shared_ppo``,
    which is what makes the latter budget-matched (Requirement 11.5).  The legacy
    21D actor's hidden width is capacity-matched against the proposed 28D actor so
    the two also carry comparable parameter counts.

    ``conformance`` supplies the interface verdict for the two conditional
    baselines.  A missing or failing verdict registers that baseline as
    ``not_run`` with a recorded reason instead of dropping it (Requirement 11.4).
    """
    reports = dict(conformance or {})
    _, legacy_hidden = find_capacity_matched_hidden_width(
        base_observation_dim=proposed_observation_dim,
        base_hidden_dims=hidden_dims,
        target_observation_dim=legacy_observation_dim,
        action_dim=ACTION_DIM,
        num_officers=POLICE_COUNT,
    )

    def measurement(baseline_id: str, *, learned: bool) -> RuntimeMeasurement:
        found = measurements.get(baseline_id)
        if found is None:
            raise ResearchValidationError(
                "MISSING_COST_MEASUREMENT",
                f"no runtime cost measurement was supplied for {baseline_id!r}",
                path="measurements", actual=baseline_id,
            )
        if learned and (found.train_wall_clock_s is None or found.accelerator_time_s is None):
            raise ResearchValidationError(
                "MISSING_COST_MEASUREMENT",
                f"learned policy {baseline_id!r} must record training wall clock and accelerator time",
                path="measurements", actual=baseline_id,
            )
        return found

    registry: dict[str, BaselineRegistration] = {}

    for baseline_id, observation_dim, actor_hidden, description in (
        (PROPOSED_POLICY_ID, proposed_observation_dim, tuple(hidden_dims), "proposed masked MAPPO"),
        (LEGACY_SHARED_PPO_ID, legacy_observation_dim, legacy_hidden, "budget-matched legacy shared-parameter PPO"),
    ):
        found = measurement(baseline_id, learned=True)
        registry[baseline_id] = BaselineRegistration(
            baseline_id=baseline_id,
            policy_kind=PolicyKind.LEARNED,
            requirement=BaselineRequirement.MINIMUM_REQUIRED,
            parameters={
                "observation_dim": observation_dim,
                "hidden_dims": actor_hidden,
                "action_dim": ACTION_DIM,
                "num_officers": POLICE_COUNT,
            },
            selection=HyperparameterSelection(
                search_space=(
                    "learning_rate in {1e-4, 3e-4, 1e-3} x entropy_coefficient in {0.0, 0.01} "
                    "at fixed TrainerConfig budget"
                ),
                selection_rule=(
                    "highest validation capture rate at the budget's checkpoint selection rule; "
                    "ties broken toward the lower learning rate"
                ),
                selection_data_splits=("train", "validation"),
            ),
            cost=_learned_cost(
                budget=budget,
                observation_dim=observation_dim,
                hidden_dims=actor_hidden,
                wall_clock_s=found.train_wall_clock_s,
                accelerator_time_s=found.accelerator_time_s,
                inference_latency_s=found.inference_latency_per_episode_s,
                measurement_method=found.method,
            ),
            status=ExecutionStatus.COMPLETED,
            status_reason=f"{description} trained and evaluated under the declared shared budget",
            training_budget=budget,
        )

    encirclement = measurement(ENCIRCLEMENT_ID, learned=False)
    registry[ENCIRCLEMENT_ID] = BaselineRegistration(
        baseline_id=ENCIRCLEMENT_ID,
        policy_kind=PolicyKind.RULE_BASED,
        requirement=BaselineRequirement.MINIMUM_REQUIRED,
        parameters={"parameter_hash": EncirclementParameters().content_hash},
        selection=FIXED_HEURISTIC_SELECTION,
        cost=_rule_based_cost(
            inference_latency_s=encirclement.inference_latency_per_episode_s,
            measurement_method=encirclement.method,
        ),
        status=ExecutionStatus.COMPLETED,
        status_reason="audited coordinating heuristic; runs without training",
    )

    for baseline_id, label in (
        (DIRECTED_SHORTEST_PATH_ID, "directed shortest-path"),
        (GREEDY_INTERCEPT_ID, "greedy-intercept"),
    ):
        report = reports.get(baseline_id)
        if report is None:
            status = ExecutionStatus.NOT_RUN
            reason = f"{label} baseline: no common-interface conformance check was recorded"
            cost = _rule_based_cost(
                inference_latency_s=0.0,
                measurement_method="not executed: baseline is not_run pending a conformance verdict",
            )
        elif not report.conforms:
            status = ExecutionStatus.NOT_RUN
            reason = f"{label} baseline failed the common policy interface check -- {report.failure_reason}"
            cost = _rule_based_cost(
                inference_latency_s=0.0,
                measurement_method="not executed: baseline failed the common interface conformance check",
            )
        else:
            status = ExecutionStatus.COMPLETED
            reason = f"{label} baseline passed the common policy interface check"
            found = measurement(baseline_id, learned=False)
            cost = _rule_based_cost(
                inference_latency_s=found.inference_latency_per_episode_s,
                measurement_method=found.method,
            )
        registry[baseline_id] = BaselineRegistration(
            baseline_id=baseline_id,
            policy_kind=PolicyKind.RULE_BASED,
            requirement=BaselineRequirement.CONDITIONAL_ON_CONFORMANCE,
            parameters={"deterministic": True, "tie_break": "lowest_action_slot"},
            selection=FIXED_HEURISTIC_SELECTION,
            cost=cost,
            status=status,
            status_reason=reason,
            interface_report=report,
        )
    return registry


def required_baseline_ids(registry: Mapping[str, BaselineRegistration]) -> tuple[str, ...]:
    """Baselines that must produce a result row in every primary stratum.

    Minimum-required baselines always count (Requirement 11.1); conditional ones
    count once their conformance check passed (Requirement 11.3).
    """
    required = []
    for baseline_id, registration in registry.items():
        if registration.requirement is BaselineRequirement.MINIMUM_REQUIRED:
            required.append(baseline_id)
        elif registration.status is ExecutionStatus.COMPLETED:
            required.append(baseline_id)
    return tuple(sorted(required))


def not_run_baselines(registry: Mapping[str, BaselineRegistration]) -> tuple[tuple[str, str], ...]:
    """Preserved ``not_run`` baselines and their recorded reasons (Requirement 11.4)."""
    return tuple(
        (baseline_id, registration.status_reason)
        for baseline_id, registration in sorted(registry.items())
        if registration.status is ExecutionStatus.NOT_RUN
    )


# ---------------------------------------------------------------------------
# Result rows and primary matrix validation (Requirements 11.7, 11.8, 11.10)
# ---------------------------------------------------------------------------


class ComparisonOutcome(str, Enum):
    """How the proposed policy fared against a baseline on one stratum.

    Requirement 11.8 forbids filtering: inferior, tied and undetermined rows are
    first-class members of the matrix, not omissions.
    """

    PROPOSED_SUPERIOR = "proposed_superior"
    PROPOSED_INFERIOR = "proposed_inferior"
    TIE = "tie"
    UNDETERMINED = "undetermined"


NON_WINNING_OUTCOMES = frozenset({
    ComparisonOutcome.PROPOSED_INFERIOR,
    ComparisonOutcome.TIE,
    ComparisonOutcome.UNDETERMINED,
})


@dataclass(frozen=True, slots=True)
class EvaluationStratum:
    """One primary evaluation stratum: a map scenario crossed with a data split."""

    scenario: MapScenario
    split: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario", MapScenario(self.scenario))
        if not isinstance(self.split, str) or not self.split.strip():
            raise ResearchValidationError("MISSING_REQUIRED_FIELD", "split must be non-empty", path="split")

    @property
    def stratum_id(self) -> str:
        return f"{self.scenario.value}/{self.split}"


@dataclass(frozen=True, slots=True)
class EpisodeCaseRef:
    """The identifying fields of the shared evaluation case (Requirement 11.7).

    Every policy in a stratum is scored on the same case.  The fields are the
    generic identity of an episode -- map, scenario, placement and seed -- so the
    paired-evaluation ``EpisodeCase`` built later can reference this row without
    the registry needing to change.
    """

    map_hash: str
    scenario: MapScenario
    placement_hash: str
    seed: int

    def __post_init__(self) -> None:
        for name in ("map_hash", "placement_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ResearchValidationError("MISSING_REQUIRED_FIELD", f"{name} must be non-empty", path=name)
        object.__setattr__(self, "scenario", MapScenario(self.scenario))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ResearchValidationError(
                "INVALID_EPISODE_CASE", "seed must be a nonnegative integer", path="seed", actual=self.seed
            )

    @property
    def case_id(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class MetricSchemaRefs:
    """References to the metric artifacts backing one result row.

    All of :data:`REQUIRED_METRIC_FIELDS` must be present and non-empty, mirroring
    the mandatory-reference pattern of ``RewardAblationResult``: a comparison can
    never be reported without the behavioral evidence behind it.  Extra keys are
    allowed, but :func:`validate_primary_matrix` requires every row in a stratum
    to expose the *same* key set (Requirement 11.7).
    """

    refs: Mapping[str, str]

    def __post_init__(self) -> None:
        refs = {str(key): value for key, value in self.refs.items()}
        for key, value in refs.items():
            if not isinstance(value, str) or not value.strip():
                raise ResearchValidationError(
                    "MISSING_METRIC_REFERENCE", f"metric reference {key!r} must be non-empty", path=f"refs.{key}"
                )
        missing = sorted(set(REQUIRED_METRIC_FIELDS) - set(refs))
        if missing:
            raise ResearchValidationError(
                "MISSING_METRIC_REFERENCE",
                "result rows must carry the full metric contract",
                path="refs", expected=list(REQUIRED_METRIC_FIELDS), actual=missing,
            )
        object.__setattr__(self, "refs", MappingProxyType(refs))

    @property
    def schema_key(self) -> tuple[str, ...]:
        return tuple(sorted(self.refs))


@dataclass(frozen=True, slots=True)
class BaselineResultRow:
    """One baseline's preserved result in one primary evaluation stratum."""

    stratum: EvaluationStratum
    baseline_id: str
    episode_case: EpisodeCaseRef
    metrics: MetricSchemaRefs
    outcome_vs_proposed: ComparisonOutcome
    schema_version: str = BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_id, str) or not self.baseline_id.strip():
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "baseline_id must be non-empty", path="baseline_id"
            )
        object.__setattr__(self, "outcome_vs_proposed", ComparisonOutcome(self.outcome_vs_proposed))
        if self.episode_case.scenario is not self.stratum.scenario:
            raise ResearchValidationError(
                "EPISODE_CASE_SCENARIO_MISMATCH",
                "a result row's episode case must belong to its own stratum scenario",
                path="episode_case", expected=self.stratum.scenario.value, actual=self.episode_case.scenario.value,
            )

    @property
    def config_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class PrimaryMatrixReport:
    """The accepted, unfiltered result matrix for one stratum."""

    stratum: EvaluationStratum
    retained_rows: tuple[BaselineResultRow, ...]
    not_run: tuple[tuple[str, str], ...]
    budget_hash: str | None
    episode_case_id: str

    @property
    def outcome_counts(self) -> Mapping[ComparisonOutcome, int]:
        counts = {outcome: 0 for outcome in ComparisonOutcome}
        for row in self.retained_rows:
            counts[row.outcome_vs_proposed] += 1
        return MappingProxyType(counts)

    @property
    def non_winning_rows(self) -> tuple[BaselineResultRow, ...]:
        """Inferior/tied/undetermined rows, retained verbatim (Requirement 11.8)."""
        return tuple(row for row in self.retained_rows if row.outcome_vs_proposed in NON_WINNING_OUTCOMES)


def _validate_stratum(
    stratum: EvaluationStratum,
    rows: Sequence[BaselineResultRow],
    registry: Mapping[str, BaselineRegistration],
) -> PrimaryMatrixReport:
    by_baseline: dict[str, BaselineResultRow] = {}
    for row in rows:
        if row.baseline_id in by_baseline:
            raise ResearchValidationError(
                "DUPLICATE_BASELINE_ROW",
                f"{row.baseline_id!r} reports more than one row in stratum {stratum.stratum_id!r}",
                path="rows", actual=row.baseline_id,
            )
        by_baseline[row.baseline_id] = row

    unregistered = sorted(set(by_baseline) - set(registry))
    if unregistered:
        raise ResearchValidationError(
            "UNREGISTERED_BASELINE_ROW",
            f"stratum {stratum.stratum_id!r} reports baselines that are not registered",
            path="rows", expected=sorted(registry), actual=unregistered,
        )

    required = required_baseline_ids(registry)
    missing = sorted(set(required) - set(by_baseline))
    if missing:
        raise ResearchValidationError(
            "MISSING_MINIMUM_BASELINE",
            f"stratum {stratum.stratum_id!r} is missing required baseline results",
            path="rows", expected=list(required), actual=missing,
        )

    for baseline_id in sorted(by_baseline):
        registration = registry[baseline_id]
        if registration.status is ExecutionStatus.NOT_RUN:
            raise ResearchValidationError(
                "NOT_RUN_BASELINE_HAS_RESULT",
                f"{baseline_id!r} is registered not_run but reports a result row",
                path="rows", actual=baseline_id,
            )
        unrecorded = registration.cost.unrecorded_fields
        if unrecorded:
            raise ResearchValidationError(
                "INCOMPLETE_COST_ACCOUNTING",
                f"{baseline_id!r} reports results with unrecorded cost fields",
                path="cost", actual=list(unrecorded),
            )

    reference_row = by_baseline[sorted(by_baseline)[0]]
    for baseline_id, row in sorted(by_baseline.items()):
        if row.episode_case != reference_row.episode_case:
            raise ResearchValidationError(
                "EPISODE_CASE_MISMATCH",
                f"{baseline_id!r} was scored on a different episode case than the rest of the stratum",
                path="episode_case",
                expected=reference_row.episode_case.case_id, actual=row.episode_case.case_id,
            )
        if row.metrics.schema_key != reference_row.metrics.schema_key:
            raise ResearchValidationError(
                "METRIC_SCHEMA_MISMATCH",
                f"{baseline_id!r} reports a different metric schema than the rest of the stratum",
                path="metrics",
                expected=list(reference_row.metrics.schema_key), actual=list(row.metrics.schema_key),
            )

    budgets = {
        registry[baseline_id].training_budget.budget_hash: baseline_id
        for baseline_id in by_baseline
        if registry[baseline_id].policy_kind is PolicyKind.LEARNED
    }
    if len(budgets) > 1:
        raise ResearchValidationError(
            "BUDGET_MISMATCH",
            f"learned policies in stratum {stratum.stratum_id!r} were not trained under a matched budget",
            path="training_budget", actual=sorted(budgets.values()),
        )

    return PrimaryMatrixReport(
        stratum=stratum,
        retained_rows=tuple(rows),
        not_run=not_run_baselines(registry),
        budget_hash=next(iter(budgets), None),
        episode_case_id=reference_row.episode_case.case_id,
    )


def validate_primary_matrix(
    rows: Sequence[BaselineResultRow],
    registry: Mapping[str, BaselineRegistration],
    *,
    strata: Sequence[EvaluationStratum],
) -> tuple[PrimaryMatrixReport, ...]:
    """Validate baseline coverage and fairness for every primary stratum.

    Raises :class:`ResearchValidationError` on a missing minimum baseline, a
    budget mismatch between compared learned policies, a metric-schema or
    episode-case difference, an unregistered or duplicated row, a row for a
    ``not_run`` baseline, or unrecorded costs.  Nothing is ever filtered: the
    returned reports retain every supplied row, including the ones where the
    proposed policy loses or ties (Requirement 11.8).
    """
    if not strata:
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD", "at least one primary stratum must be declared", path="strata"
        )
    grouped: dict[str, list[BaselineResultRow]] = {}
    for row in rows:
        grouped.setdefault(row.stratum.stratum_id, []).append(row)

    declared = {stratum.stratum_id: stratum for stratum in strata}
    extra = sorted(set(grouped) - set(declared))
    if extra:
        raise ResearchValidationError(
            "UNDECLARED_STRATUM",
            "result rows reference strata that are not declared primary strata",
            path="strata", expected=sorted(declared), actual=extra,
        )
    reports = []
    for stratum_id, stratum in sorted(declared.items()):
        stratum_rows = grouped.get(stratum_id)
        if not stratum_rows:
            raise ResearchValidationError(
                "MISSING_STRATUM",
                f"primary stratum {stratum_id!r} has no baseline results",
                path="strata", actual=stratum_id,
            )
        reports.append(_validate_stratum(stratum, stratum_rows, registry))
    return tuple(reports)


__all__ = (
    "ALLOWED_SELECTION_SPLITS",
    "BASELINE_SCHEMA_VERSION",
    "CONDITIONAL_BASELINE_IDS",
    "COST_FIELD_NAMES",
    "DEFAULT_ACTOR_HIDDEN_DIMS",
    "DIRECTED_SHORTEST_PATH_ID",
    "ENCIRCLEMENT_ID",
    "FIXED_HEURISTIC_SELECTION",
    "GREEDY_INTERCEPT_ID",
    "GREEDY_INTERCEPT_PROFILE_ID",
    "LEGACY_SHARED_PPO_ID",
    "MINIMUM_REQUIRED_BASELINE_IDS",
    "NON_WINNING_OUTCOMES",
    "POLICY_INTERFACE_CONTRACT",
    "PROPOSED_POLICY_ID",
    "REQUIRED_METRIC_FIELDS",
    "BaselineRegistration",
    "BaselineRequirement",
    "BaselineResultRow",
    "ComparisonOutcome",
    "CostAvailability",
    "CostField",
    "EncirclementParameters",
    "EncirclementPolice",
    "EpisodeCaseRef",
    "EvaluationStratum",
    "GoalAssignment",
    "GoalAssignmentPlan",
    "GoalEvader",
    "GoalEvaderParameters",
    "GreedyInterceptPolice",
    "HyperparameterSelection",
    "InterfaceCheck",
    "MetricSchemaRefs",
    "PolicyCostSchema",
    "PolicyInterfaceReport",
    "PolicyKind",
    "PrimaryMatrixReport",
    "RuntimeMeasurement",
    "SmartEvader",
    "TrainingBudget",
    "check_policy_interface_conformance",
    "decision_intersection_id",
    "directed_shortest_path_policy",
    "greedy_intercept_policy",
    "make_interior_network",
    "measured_cost",
    "not_applicable_cost",
    "not_measured_cost",
    "not_run_baselines",
    "numeric_baseline_registry",
    "required_baseline_ids",
    "validate_primary_matrix",
    "_Graph",
)
