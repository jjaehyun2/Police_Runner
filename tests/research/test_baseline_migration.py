"""Regression coverage for the root-to-package baseline migration."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import importlib
import pickle
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig
from pursuit_evasion_rl.osm_demo.runner import run_single_episode
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.policies.baselines import (
    EncirclementParameters,
    EncirclementPolice,
    GoalAssignmentPlan,
    GoalEvader,
    GoalEvaderParameters,
    _Graph,
)

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[2]


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _config() -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=14.0,
        fugitive_speed_mps=10.0,
        capture_radius_m=20.0,
        max_steps=12,
    )


def _run(policy_type, graph_type):
    network = _network()
    graph = graph_type(network)
    policy = policy_type(network, graph=graph)
    evaders = []

    def fugitive_factory(rng):
        evader = GoalEvader(network, rng=rng, graph=graph)
        evaders.append(evader)
        return evader

    record = run_single_episode(
        network,
        _config(),
        policy,
        run_id="migration",
        run_seed=417,
        episode_index=2,
        fugitive_factory=fugitive_factory,
    )
    hashes = {
        "action": content_hash(tuple(transition.actions for transition in record.transitions)),
        "state": content_hash(
            {
                "initial": record.initial_state,
                "states": tuple(transition.state for transition in record.transitions),
            }
        ),
        "outcome": content_hash(
            {"outcome": record.outcome, "terminal_priority": record.terminal_priority}
        ),
    }
    return hashes, policy, evaders[0]


def _load_legacy_module():
    sys.modules.pop("demo_pursuit", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("demo_pursuit")
    assert any(
        item.category is DeprecationWarning and "deprecated" in str(item.message)
        for item in caught
    )
    return module


def test_legacy_and_package_paths_have_identical_action_state_outcome_hashes():
    """Same fixture/seed preserves the canonical migration contract."""
    legacy = _load_legacy_module()

    legacy_hashes, legacy_policy, legacy_evader = _run(
        legacy.EncirclementPolice, legacy._Graph
    )
    package_hashes, package_policy, package_evader = _run(
        EncirclementPolice, _Graph
    )

    assert legacy_hashes == package_hashes
    assert legacy_policy.parameter_hash == package_policy.parameter_hash
    assert legacy_evader.parameter_hash == package_evader.parameter_hash
    assert isinstance(package_policy.last_goal_assignment, GoalAssignmentPlan)
    assert (
        legacy_policy.last_goal_assignment.content_hash
        == package_policy.last_goal_assignment.content_hash
    )


def test_legacy_import_and_pickle_globals_resolve_to_package_classes():
    legacy = _load_legacy_module()

    assert legacy.GoalEvader is GoalEvader
    assert legacy.EncirclementPolice is EncirclementPolice
    assert legacy._Graph is _Graph
    assert pickle.loads(b"cdemo_pursuit\nGoalEvader\n.") is GoalEvader
    assert pickle.loads(b"cdemo_pursuit\nEncirclementPolice\n.") is EncirclementPolice
    assert pickle.loads(b"cdemo_pursuit\n_Graph\n.") is _Graph


def test_new_package_has_no_root_module_import():
    tree = ast.parse(
        (ROOT / "pursuit_evasion_rl/research/policies/baselines.py").read_text(
            encoding="utf-8"
        )
    )
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "demo_pursuit" not in imported


def test_public_parameter_schema_hashes_are_stable():
    assert GoalEvaderParameters().content_hash == content_hash(GoalEvaderParameters())
    assert EncirclementParameters().content_hash == content_hash(EncirclementParameters())
