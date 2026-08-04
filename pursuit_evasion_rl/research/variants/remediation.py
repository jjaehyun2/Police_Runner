"""Remediated reward Condition for the S1/S2 training symptoms (2026-08-05).

Registered as a NEW Condition (protocol C4: coefficients are never changed
in place; the audited default in ``variants/rewards.py`` stays byte-identical).
This module bundles the reward-side fixes:

  A. distances measured on the DIRECTED ROAD GRAPH (meters), not euclidean --
     road-optimal moves can no longer be charged as "retreat" just because a
     detour transiently increases straight-line distance;
  B. symmetric regress (multiplier 1.0) via the existing symmetric arm --
     removes the concave kink that made exploratory movement strictly worse
     than STAY in expectation for far officers;
  C. the own term holds the fugitive at its POST-step position on both
     endpoints -- an officer's own reward no longer carries +/- fugitive-motion
     noise that its action cannot influence;
  G. per-officer credit assignment -- the team min-distance delta is credited
     only to the officer defining the new minimum, and the capture bonus pays
     the capturer in full while other officers receive
     ``noncapturer_capture_share`` of it (free-riding no longer pays full).

Trainer-side fixes D (timeout truncation bootstrap), E (arrival-intersection
decisions), and F (u-turn suppression ON + stored logit-bias replay) are
opt-in ``ResearchTrainer`` axes; :func:`remediated_trainer_kwargs` returns the
full bundle for launchers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import ModelNetwork, POLICE_COUNT
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.policies.baselines import _Graph
from pursuit_evasion_rl.research.variants.rewards import (
    RewardComponentSet,
    symmetric_retreat_components,
)
from pursuit_evasion_rl.research.variants.stabilization import StabilizationCondition

REMEDIATED_CONDITION_ID = "real_scale_daejeon_remediated_v1"

#: Capture-bonus share paid to every officer that did not make the capture.
#: 1.0 reproduces the audited broadcast; 0.0 pays the capturer alone.  0.25
#: keeps a cooperative signal while making contribution clearly dominant.
#: This is the primary open design knob of the remediated Condition.
NONCAPTURER_CAPTURE_SHARE = 0.25

#: Per-step distance deltas are physically bounded by the two speeds; clamp at
#: one distance scale (50 m) so a reachability flip in the road graph can
#: never inject a sentinel-sized reward impulse.
MAX_ABS_DELTA_M = 50.0


def _placement_source(graph: _Graph, placement: Any) -> tuple[int, float]:
    """The node an officer can next act from, plus the arc offset to reach it."""
    if placement.segment_id is not None:
        segment = graph.seg[placement.segment_id]
        remaining = (1.0 - float(placement.progress)) * float(segment.length_m)
        return int(segment.end_id), max(0.0, remaining)
    return int(placement.intersection_id), 0.0


def _placement_target(graph: _Graph, placement: Any) -> tuple[int, float]:
    """The node from which a mid-segment target point is entered, plus offset."""
    if placement.segment_id is not None:
        segment = graph.seg[placement.segment_id]
        offset = float(placement.progress) * float(segment.length_m)
        return int(segment.start_id), max(0.0, offset)
    return int(placement.intersection_id), 0.0


class RoadDistance:
    """Directed road-graph distance between two vehicle placements, cached.

    Falls back to euclidean distance when the target is unreachable in the
    directed graph (a lower bound, so the delta stays smooth instead of
    jumping to a sentinel).
    """

    def __init__(self, network: ModelNetwork, graph: _Graph | None = None) -> None:
        self.network = network
        self.graph = graph or _Graph(network)

    def between(self, source_placement: Any, target_placement: Any) -> float:
        graph = self.graph
        # Same-segment shortcut: the officer is behind the target on one road.
        if (
            source_placement.segment_id is not None
            and source_placement.segment_id == target_placement.segment_id
            and float(source_placement.progress) <= float(target_placement.progress)
        ):
            segment = graph.seg[source_placement.segment_id]
            gap = float(target_placement.progress) - float(source_placement.progress)
            return gap * float(segment.length_m)
        source_node, source_offset = _placement_source(graph, source_placement)
        target_node, target_offset = _placement_target(graph, target_placement)
        network_distance = graph.dist_to(target_node).get(source_node)
        if network_distance is None:
            return float(
                math.dist(
                    placement_position(self.network, source_placement),
                    placement_position(self.network, target_placement),
                )
            )
        return source_offset + float(network_distance) + target_offset


@dataclass
class RemediatedStepReward:
    """``ResearchTrainer.step_reward_fn`` implementing fixes A, B, C, and G.

    One instance may serve several networks (train + validation): road-graph
    caches are built lazily per network identity.
    """

    components: RewardComponentSet = field(default_factory=symmetric_retreat_components)
    noncapturer_capture_share: float = NONCAPTURER_CAPTURE_SHARE
    max_abs_delta_m: float = MAX_ABS_DELTA_M

    def __post_init__(self) -> None:
        share = float(self.noncapturer_capture_share)
        if not math.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ResearchValidationError(
                "INVALID_REWARD_COEFFICIENT",
                "noncapturer_capture_share must be in [0, 1]",
                actual=share,
            )
        self._distances: dict[int, RoadDistance] = {}

    def _road(self, network: ModelNetwork) -> RoadDistance:
        cached = self._distances.get(id(network))
        if cached is None:
            cached = RoadDistance(network)
            self._distances[id(network)] = cached
        return cached

    def _clamped_delta(self, before_m: float, after_m: float) -> float:
        raw = before_m - after_m
        return max(-self.max_abs_delta_m, min(self.max_abs_delta_m, raw))

    def __call__(self, network: ModelNetwork, before: Any, after: Any, captured: bool) -> tuple[float, ...]:
        road = self._road(network)
        c = self.components
        scale = c.distance_scale_m
        # C: own endpoints share the post-step fugitive position.
        own_before = [road.between(placement, after.fugitive) for placement in before.police]
        own_after = [road.between(placement, after.fugitive) for placement in after.police]
        # Team endpoints stay on the joint states (fugitive progress matters).
        joint_before = [road.between(placement, before.fugitive) for placement in before.police]
        joint_after = own_after  # identical measurements (fugitive after)
        team_delta = self._clamped_delta(min(joint_before), min(joint_after)) / scale
        argmin_officer = min(range(POLICE_COUNT), key=lambda index: joint_after[index])
        rewards: list[float] = []
        for index in range(POLICE_COUNT):
            delta = self._clamped_delta(own_before[index], own_after[index]) / scale
            # Same kink formula as the audited registry: with the symmetric
            # arm (multiplier 1.0, fix B) this is the identity on retreats.
            own_delta = delta if delta >= 0.0 else c.regress_multiplier * delta
            value = c.own_coefficient * own_delta
            if index == argmin_officer:
                value += c.team_coefficient * team_delta
            value -= c.time_penalty
            if captured:
                share = 1.0 if index == argmin_officer else self.noncapturer_capture_share
                value += c.capture_bonus * share
            rewards.append(value)
        return tuple(rewards)


def remediated_stabilization() -> StabilizationCondition:
    """U-turn suppression ON (fix F); hysteresis stays off (see ablations.py:
    distant-goal holding is incompatible with neighbor-only decisions)."""
    return StabilizationCondition(u_turn_suppression=True, hysteresis=False)


def remediated_trainer_kwargs(
    *,
    components: RewardComponentSet | None = None,
    noncapturer_capture_share: float = NONCAPTURER_CAPTURE_SHARE,
) -> dict[str, Any]:
    """The full remediation bundle as ``ResearchTrainer`` keyword arguments."""
    reward_components = components or symmetric_retreat_components()
    return {
        "reward_components": reward_components,
        "step_reward_fn": RemediatedStepReward(
            components=reward_components,
            noncapturer_capture_share=noncapturer_capture_share,
        ),
        "stabilization_condition": remediated_stabilization(),
        "timeout_bootstrap": True,
        "arrival_decisions": True,
    }


__all__ = (
    "MAX_ABS_DELTA_M",
    "NONCAPTURER_CAPTURE_SHARE",
    "REMEDIATED_CONDITION_ID",
    "RemediatedStepReward",
    "RoadDistance",
    "remediated_stabilization",
    "remediated_trainer_kwargs",
)
