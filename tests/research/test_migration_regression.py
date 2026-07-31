"""End-to-end regressions for the root-to-research-package migration.

The comparisons deliberately use the real deterministic episode runner and
checkpoint fingerprinting.  They prove compatibility, not scientific
performance or new experimental evidence.

**Validates: Requirements 1.1, 9.2, 11.3, 19.4**
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import (
    EpisodeConfig,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.runner import run_single_episode
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.migration import (
    EVALUATE_CLI_DEFAULTS,
    FORBIDDEN_ROOT_MODULES,
    TRAIN_CLI_DEFAULTS,
    assert_research_import_boundary,
    convert_legacy_checkpoint,
    convert_legacy_cli_arguments,
    find_forbidden_research_imports,
    load_legacy_checkpoint,
)
from pursuit_evasion_rl.research.policies.baselines import (
    EncirclementPolice,
    GoalEvader,
    _Graph,
)
from pursuit_evasion_rl.research.variants.observations import Observation28DAdapter

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/migration_golden.json").read_text(encoding="utf-8")
)

def _episode_network() -> ModelNetwork:
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _episode_config() -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=14.0,
        fugitive_speed_mps=10.0,
        capture_radius_m=20.0,
        max_steps=12,
    )


def _load_legacy_baseline_module():
    sys.modules.pop("demo_pursuit", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("demo_pursuit")
    assert any(item.category is DeprecationWarning for item in caught)
    return module


def _episode_snapshot(policy_type, graph_type):
    network = _episode_network()
    graph = graph_type(network)
    policy = policy_type(network, graph=graph)

    def fugitive_factory(rng):
        return GoalEvader(network, rng=rng, graph=graph)

    record = run_single_episode(
        network,
        _episode_config(),
        policy,
        run_id="package-migration-regression",
        run_seed=417,
        episode_index=2,
        fugitive_factory=fugitive_factory,
    )
    state = {
        "initial": record.initial_state,
        "trace": tuple(transition.state for transition in record.transitions),
    }
    actions = tuple(transition.actions for transition in record.transitions)
    outcome = {
        "outcome": record.outcome,
        "terminal_priority": record.terminal_priority,
    }
    return {
        "state": state,
        "actions": actions,
        "outcome": outcome,
        "hashes": {
            "state": content_hash(state),
            "actions": content_hash(actions),
            "outcome": content_hash(outcome),
        },
        "record_hashes": dict(record.hashes),
        "parameter_hash": policy.parameter_hash,
        "assignment_hash": policy.last_goal_assignment.content_hash,
    }


def test_baseline_shim_preserves_state_action_outcome_and_hashes():
    legacy = _load_legacy_baseline_module()

    before = _episode_snapshot(legacy.EncirclementPolice, legacy._Graph)
    after = _episode_snapshot(EncirclementPolice, _Graph)

    assert legacy.EncirclementPolice is EncirclementPolice
    assert legacy.GoalEvader is GoalEvader
    assert legacy._Graph is _Graph
    assert before["state"] == after["state"]
    assert before["actions"] == after["actions"]
    assert before["outcome"] == after["outcome"]
    assert before["hashes"] == after["hashes"]
    assert before["record_hashes"] == after["record_hashes"]
    assert before["parameter_hash"] == after["parameter_hash"]
    assert before["assignment_hash"] == after["assignment_hash"]

def _observation_network() -> ModelNetwork:
    positions = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    segments = tuple(
        Segment(
            id=index,
            start_id=index,
            end_id=(index + 1) % 4,
            length_m=100.0,
            geometry_xy=(positions[index], positions[(index + 1) % 4]),
        )
        for index in range(4)
    )
    intersections = tuple(
        Intersection(
            id=index,
            position_xy=position,
            source_signature=f"migration-fixture-{index}",
            outgoing_segment_ids=(index,),
            incoming_segment_ids=((index - 1) % 4,),
        )
        for index, position in enumerate(positions)
    )
    return ModelNetwork(intersections, segments)


def _observation_vectors(adapter, police, fugitive) -> tuple[np.ndarray, ...]:
    return tuple(
        adapter.observe(
            police_index=index,
            police=police,
            fugitive=fugitive,
            step=7,
            max_steps=40,
            incoming_heading=0.25,
        )
        for index in range(6)
    )


def test_observation_shim_preserves_input_state_vectors_and_hashes():
    legacy = importlib.import_module("osm_obs_aug")
    network = _observation_network()
    police = tuple(
        VehiclePlacement(intersection_id=index) for index in (0, 1, 2, 3, 0, 2)
    )
    fugitive = VehiclePlacement(intersection_id=1)
    input_state = {"police": police, "fugitive": fugitive, "step": 7, "max_steps": 40}
    input_hash_before = content_hash(input_state)

    shim_vectors = _observation_vectors(legacy.AugmentedObs(network), police, fugitive)
    package_vectors = _observation_vectors(Observation28DAdapter(network), police, fugitive)

    assert content_hash(input_state) == input_hash_before
    assert legacy.AugmentedObs is Observation28DAdapter
    assert len(shim_vectors) == len(package_vectors) == 6
    for shim_vector, package_vector in zip(shim_vectors, package_vectors):
        assert shim_vector.dtype == package_vector.dtype == np.float32
        assert np.array_equal(shim_vector, package_vector)