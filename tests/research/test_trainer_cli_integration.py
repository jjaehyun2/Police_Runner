"""Focused integration checks for task 4.3 trainer and CLI migration."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.cli.evaluate import LearnedPolice
from pursuit_evasion_rl.research.maps import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.policies.baselines import _Graph
from pursuit_evasion_rl.research.training.trainer import ResearchTrainer, TrainerConfig

pytestmark = pytest.mark.offline
ROOT = Path(__file__).resolve().parents[2]


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "fixture-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "fixture-validation"),),
    )


def test_tiny_two_update_training_is_legal_validation_selected_and_run_scoped(tmp_path):
    output_root = tmp_path / "artifacts" / "research" / "checkpoints"
    trainer = ResearchTrainer(
        train_network=_network(),
        validation_network=_network(),
        tuning_data=_tuning_view(),
        config=TrainerConfig(
            updates=2,
            episodes_per_update=1,
            validation_episodes=1,
            max_steps=3,
            hidden_dims=(8,),
            ppo_epochs=1,
        ),
        condition_id="tiny-fixture",
        training_seed=73,
        output_root=output_root,
        run_id="two-update-run",
    )

    result = trainer.train()

    expected_run = output_root / "tiny-fixture" / "73" / "two-update-run"
    assert result.run_directory == expected_run
    assert len(result.history) == 2
    assert [item.update_index for item in result.history] == [1, 2]
    assert all(item.transition_count > 0 for item in result.history)
    assert result.all_actions_legal
    assert result.checkpoint_path.parent == expected_run
    assert list(tmp_path.rglob("*.pt")) == [result.checkpoint_path]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["state"] == "sealed"
    assert manifest["selection_source"] == "validation"
    assert manifest["checkpoint"]["selection_source"] == "validation"
    assert set(manifest["data_capabilities"]) == {"train", "validation"}
    capabilities = json.dumps(manifest["data_capabilities"]).lower()
    assert '"test"' not in capabilities
    assert "cross_city" not in capabilities

    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["selection_source"] == "validation"
    assert payload["update_index"] in {1, 2}


def test_checkpoint_loads_through_package_evaluator_and_emits_legal_recommendations(tmp_path):
    network = _network()
    trainer = ResearchTrainer(
        train_network=network,
        validation_network=network,
        tuning_data=_tuning_view(),
        config=TrainerConfig(updates=1, max_steps=2, hidden_dims=(8,)),
        condition_id="evaluation-fixture",
        training_seed=11,
        output_root=tmp_path / "artifacts" / "research" / "checkpoints",
        run_id="evaluation-load",
    )
    result = trainer.train()
    policy = LearnedPolice(network, _Graph(network), result.checkpoint_path)

    from pursuit_evasion_rl.osm_demo.environment import OSMRoadPursuitEnv
    from pursuit_evasion_rl.osm_demo.models import EpisodeConfig

    env = OSMRoadPursuitEnv(network, EpisodeConfig(1.0, 16.0, 9.0, 25.0, 2))
    env.reset(seed=9)
    state = env.episode_state()
    recommendations = policy.recommend(
        police=state.police,
        fugitive=state.fugitive,
        step=state.step,
        max_steps=2,
        incoming_headings=[state.incoming_headings.get(f"police_{index}") for index in range(6)],
    )
    masks = env.action_masks()
    assert len(recommendations) == 6
    assert all(masks[f"police_{index}"][recommendation.action_index] for index, recommendation in enumerate(recommendations))


def test_trainer_surface_excludes_held_out_handles_and_legacy_ppo_imports():
    parameters = inspect.signature(ResearchTrainer.__init__).parameters
    assert "test" not in parameters
    assert "test_data" not in parameters
    assert "cross_city" not in parameters

    trainer_path = ROOT / "pursuit_evasion_rl" / "research" / "training" / "trainer.py"
    source = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "pursuit_evasion_rl.training.algorithms" not in imported
    assert "_ppo_update" not in source


def test_all_four_root_scripts_are_package_cli_wrappers():
    expected = {
        "train_osm_pursuit.py": "pursuit_evasion_rl.research",
        "eval_trained.py": "pursuit_evasion_rl.research",
        "diag_episode.py": "pursuit_evasion_rl.research.cli.evaluate",
        "render_zoom.py": "pursuit_evasion_rl.research.cli.evaluate",
    }
    for filename, package in expected.items():
        source = (ROOT / filename).read_text(encoding="utf-8")
        assert package in source
        assert "pursuit_evasion_rl.training.algorithms" not in source
        assert "from demo_pursuit" not in source
        assert "from osm_obs_aug" not in source
