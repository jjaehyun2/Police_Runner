"""Focused unit tests for fixed baseline parameter and assignment schemas."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import DomainValidationError, VehiclePlacement
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.policies.baselines import (
    EncirclementParameters,
    EncirclementPolice,
    GoalAssignmentPlan,
    GoalEvaderParameters,
    _Graph,
)

pytestmark = pytest.mark.offline


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def test_fixed_parameter_schemas_are_frozen_hashable_and_complete():
    evader = GoalEvaderParameters()
    police = EncirclementParameters()

    assert evader.content_hash == content_hash(evader)
    assert police.content_hash == content_hash(police)
    assert police.direct_pursuit_count == 2
    assert police.cordon_ring_min_m < police.cordon_ring_max_m
    with pytest.raises(FrozenInstanceError):
        police.close_pursuit_m = 1.0


def test_parameter_schemas_reject_invalid_values():
    with pytest.raises(DomainValidationError, match="positive"):
        GoalEvaderParameters(vision_range_m=0.0)
    with pytest.raises(DomainValidationError, match="below maximum"):
        EncirclementParameters(cordon_ring_min_m=400.0, cordon_ring_max_m=300.0)


def test_encirclement_emits_a_complete_goal_assignment_schema():
    network = _network()
    graph = _Graph(network)
    policy = EncirclementPolice(network, graph=graph)
    nodes = [item.id for item in network.intersections]
    police = tuple(VehiclePlacement(intersection_id=nodes[index]) for index in range(6))
    fugitive = VehiclePlacement(intersection_id=nodes[-1])

    recommendations = policy.recommend(
        police=police,
        fugitive=fugitive,
        step=0,
        max_steps=10,
        incoming_headings=[None] * 6,
    )

    plan = policy.last_goal_assignment
    assert isinstance(plan, GoalAssignmentPlan)
    assert len(recommendations) == len(plan.assignments) == 6
    assert {item.officer_id for item in plan.assignments} == set(range(6))
    assert {item.proximity_rank for item in plan.assignments} == set(range(6))
    assert plan.content_hash == content_hash(plan)
