"""Regression test for a real bug found during the real-scale run: checkpoint
selection used strict improvement (`>`), so once validation_capture_rate (a
single-episode 0/1 score by default) first hit its ceiling, every later
update tied it and none of them counted as "better" -- the saved checkpoint
froze at whichever update first got lucky, discarding the rest of training.
The fix treats a tie as an improvement (`>=`) so the checkpoint keeps moving
forward with training instead of pinning near initialization.
"""
from __future__ import annotations

import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.training.trainer import ResearchTrainer, TrainerConfig

import pytest

pytestmark = pytest.mark.offline


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "checkpoint-selection-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "checkpoint-selection-validation"),),
    )


def test_a_validation_score_that_saturates_immediately_still_saves_the_last_update(tmp_path, monkeypatch) -> None:
    network = _network()
    config = TrainerConfig(updates=6, episodes_per_update=1, max_steps=12, hidden_dims=(4,))
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=config,
        condition_id="cond-checkpoint-selection", training_seed=1, output_root=tmp_path,
    )
    # Every update ties at the ceiling from the very first one -- exactly the
    # degenerate single-episode-validation condition that triggered the bug.
    monkeypatch.setattr(trainer, "_validation_score", lambda update_index: 1.0)

    result = trainer.train()
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)

    assert payload["update_index"] == config.updates, (
        "a validation score tied at the ceiling for every update must still save "
        "the most recent update, not freeze at the first one"
    )
    assert payload["validation_score"] == 1.0


def test_a_validation_score_that_only_ever_improves_still_saves_the_last_update(tmp_path, monkeypatch) -> None:
    network = _network()
    config = TrainerConfig(updates=5, episodes_per_update=1, max_steps=12, hidden_dims=(4,))
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=config,
        condition_id="cond-checkpoint-selection-monotone", training_seed=2, output_root=tmp_path,
    )
    scores = iter([0.0, 0.2, 0.4, 0.6, 0.8])
    monkeypatch.setattr(trainer, "_validation_score", lambda update_index: next(scores))

    result = trainer.train()
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)

    assert payload["update_index"] == config.updates
    assert payload["validation_score"] == pytest.approx(0.8)


def test_a_validation_score_that_regresses_after_an_early_peak_keeps_the_peak(tmp_path, monkeypatch) -> None:
    network = _network()
    config = TrainerConfig(updates=4, episodes_per_update=1, max_steps=12, hidden_dims=(4,))
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=config,
        condition_id="cond-checkpoint-selection-regress", training_seed=3, output_root=tmp_path,
    )
    # Peaks at update 2, then strictly regresses -- the saved checkpoint must
    # stay pinned at update 2 rather than following the score downward.
    scores = iter([0.5, 1.0, 0.3, 0.1])
    monkeypatch.setattr(trainer, "_validation_score", lambda update_index: next(scores))

    result = trainer.train()
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)

    assert payload["update_index"] == 2
    assert payload["validation_score"] == 1.0
