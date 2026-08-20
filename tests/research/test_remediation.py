"""Remediated Condition (2026-08-05): road-distance credit-assigned rewards,
timeout truncation bootstrap, arrival-intersection decisions, and stored
logit-bias replay all thread through the real trainer without breaking the
audited defaults (which tests/research/test_reward_variants.py etc. pin)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.environment import OSMRoadPursuitEnv
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, POLICE_COUNT
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.policies.baselines import _Graph
from pursuit_evasion_rl.research.policies.masked_mappo import MaskedMAPPOBatch, ResearchMaskedMAPPO, StoredActionMask
from pursuit_evasion_rl.research.smdp import AsyncDecisionTransitionBuffer, DecisionEpoch
from pursuit_evasion_rl.research.training.trainer import ResearchTrainer, TrainerConfig
from pursuit_evasion_rl.research.variants.remediation import (
    RemediatedStepReward,
    RoadDistance,
    remediated_trainer_kwargs,
)

pytestmark = pytest.mark.offline


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "remediation-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "remediation-validation"),),
    )


def _config(**overrides) -> TrainerConfig:
    fields = dict(updates=1, episodes_per_update=1, max_steps=12, hidden_dims=(4,))
    fields.update(overrides)
    return TrainerConfig(**fields)


def _episode_state(network):
    env = OSMRoadPursuitEnv(network, EpisodeConfig(dt_s=1.0, police_speed_mps=16.0, fugitive_speed_mps=9.0, capture_radius_m=25.0, max_steps=12))
    env.reset(seed=7)
    return env.episode_state()


def test_road_distance_between_parked_placements_matches_directed_dijkstra() -> None:
    network = _network()
    state = _episode_state(network)
    road = RoadDistance(network)
    graph = _Graph(network)
    source = state.police[0]
    target = state.fugitive
    expected = graph.dist_to(int(target.intersection_id)).get(int(source.intersection_id))
    if expected is None:
        pytest.skip("fixture placement pair unreachable; fallback path covered elsewhere")
    assert road.between(source, target) == pytest.approx(expected)


def test_road_distance_is_directed_not_euclidean() -> None:
    network = _network()
    road = RoadDistance(network)
    state = _episode_state(network)
    graph = _Graph(network)
    source, target = state.police[0], state.fugitive
    euclid = math.dist(
        graph.pos[int(source.intersection_id)], graph.pos[int(target.intersection_id)]
    )
    assert road.between(source, target) >= euclid - 1e-6  # road can never beat straight line


def test_static_step_pays_exactly_the_time_penalty_to_everyone() -> None:
    network = _network()
    state = _episode_state(network)
    reward_fn = RemediatedStepReward()
    rewards = reward_fn(network, state, state, False)
    assert len(rewards) == POLICE_COUNT
    for value in rewards:
        assert value == pytest.approx(-reward_fn.components.time_penalty)


def test_capture_bonus_is_split_between_capturer_and_supporters() -> None:
    network = _network()
    state = _episode_state(network)
    reward_fn = RemediatedStepReward()
    rewards = reward_fn(network, state, state, True)
    bonus = reward_fn.components.capture_bonus
    time_penalty = reward_fn.components.time_penalty
    full = pytest.approx(bonus - time_penalty)
    share = pytest.approx(bonus * reward_fn.noncapturer_capture_share - time_penalty)
    capturer_payouts = [value for value in rewards if value == full]
    supporter_payouts = [value for value in rewards if value == share]
    assert len(capturer_payouts) == 1
    assert len(supporter_payouts) == POLICE_COUNT - 1


def test_reward_deltas_are_clamped_against_reachability_flips() -> None:
    reward_fn = RemediatedStepReward()
    assert reward_fn._clamped_delta(10_000.0, 0.0) == reward_fn.max_abs_delta_m
    assert reward_fn._clamped_delta(0.0, 10_000.0) == -reward_fn.max_abs_delta_m


def _tiny_buffer_with_one_step() -> AsyncDecisionTransitionBuffer:
    buffer = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=4)
    epochs = {
        officer: DecisionEpoch.create(
            actor_obs=(0.0,), action=0, action_mask=(True, True),
            old_log_prob=-0.5, critic_context=(0.0,),
        )
        for officer in range(POLICE_COUNT)
    }
    buffer.start_decisions(epochs)
    buffer.record_physical_step((1.0,) * POLICE_COUNT)
    return buffer


def test_close_terminal_default_still_drops_the_bootstrap() -> None:
    transitions = _tiny_buffer_with_one_step().close_terminal(((0.0,),) * POLICE_COUNT)
    assert all(item.terminal for item in transitions)
    assert transitions[0].bootstrap_target(123.0) == pytest.approx(1.0)


def test_close_terminal_truncation_keeps_the_bootstrap() -> None:
    transitions = _tiny_buffer_with_one_step().close_terminal(
        ((0.0,),) * POLICE_COUNT, terminal=False
    )
    assert not any(item.terminal for item in transitions)
    expected = 1.0 + (0.9 ** 1) * 123.0
    assert transitions[0].bootstrap_target(123.0) == pytest.approx(expected)


def test_logit_bias_round_trips_from_epoch_to_transition() -> None:
    buffer = AsyncDecisionTransitionBuffer(gamma=0.9, max_virtual_hops=4)
    bias = (0.0, -1.0)
    epochs = {
        officer: DecisionEpoch.create(
            actor_obs=(0.0,), action=0, action_mask=(True, True),
            old_log_prob=-0.5, critic_context=(0.0,),
            logit_bias=bias if officer == 0 else None,
        )
        for officer in range(POLICE_COUNT)
    }
    buffer.start_decisions(epochs)
    buffer.record_physical_step((0.0,) * POLICE_COUNT)
    transitions = buffer.close_terminal(((0.0,),) * POLICE_COUNT)
    by_officer = {item.officer_id: item for item in transitions}
    assert by_officer[0].logit_bias == bias
    assert by_officer[1].logit_bias is None


def test_forward_batch_reapplies_the_sampling_bias() -> None:
    torch.manual_seed(3)
    policy = ResearchMaskedMAPPO(
        actor_obs_dim=4, critic_context_dim=8, action_dim=3, num_officers=POLICE_COUNT,
        hidden_dims=(8,), learning_rate=1e-3, entropy_coefficient=0.0, device="cpu",
    )
    observation = torch.randn(4)
    mask = policy.seal_mask(np.array([True, True, True]))
    bias = torch.tensor([0.0, -5.0, 0.0])
    rng = torch.Generator().manual_seed(11)
    sample = policy.sample(observation, mask, rng, officer_id=0, logit_bias=bias)

    def _batch(with_bias: bool) -> MaskedMAPPOBatch:
        return MaskedMAPPOBatch(
            actor_obs=observation.unsqueeze(0),
            officer_ids=torch.tensor([0], dtype=torch.long),
            actions=torch.tensor([sample.action], dtype=torch.long),
            old_log_probs=torch.tensor([sample.log_prob]),
            advantages=torch.zeros(1),
            returns=torch.zeros(1),
            critic_context=torch.zeros(1, 8),
            stored_masks=(StoredActionMask(sample.stored_mask_bytes, sample.stored_mask_hash, 3),),
            sampling_mask_bytes=(sample.stored_mask_bytes,),
            sampling_mask_hashes=(sample.stored_mask_hash,),
            logit_biases=bias.unsqueeze(0) if with_bias else None,
        )

    replayed = float(policy.recompute_log_probs(_batch(True))[0].item())
    dropped = float(policy.recompute_log_probs(_batch(False))[0].item())
    assert replayed == pytest.approx(sample.log_prob, abs=1e-5)
    assert abs(dropped - sample.log_prob) > 1e-4  # the old bug: bias silently dropped


def test_trainer_trains_with_the_full_remediation_bundle(tmp_path) -> None:
    network = _network()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(),
        config=_config(), condition_id="cond-remediated-bundle", training_seed=5,
        output_root=tmp_path, **remediated_trainer_kwargs(),
    )
    assert trainer._timeout_bootstrap and trainer._arrival_decisions
    result = trainer.train()
    assert result.checkpoint_path.is_file()
    assert result.all_actions_legal


def test_arrival_decisions_axis_alone_trains(tmp_path) -> None:
    network = _network()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(),
        config=_config(), condition_id="cond-arrival-only", training_seed=6,
        output_root=tmp_path, arrival_decisions=True,
    )
    result = trainer.train()
    assert result.checkpoint_path.is_file()
    assert result.all_actions_legal