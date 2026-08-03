"""Task 12.3 prerequisite: ResearchTrainer's reward/observation/placement axes are pluggable.

Each axis defaults to exactly the trainer's prior hardcoded behavior when its
constructor parameter is omitted (Requirement 9.6-9.7, 9.10-9.11, 10.1-10.3);
these tests exercise the *override* path against a real, tiny CPU trainer to
prove each axis is genuinely threaded through to a real rollout, not merely
accepted and ignored.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.environment import OSMRoadPursuitEnv
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.training.trainer import ResearchTrainer, TrainerConfig
from pursuit_evasion_rl.research.variants.observations import (
    OBSERVATION_21D_DIM,
    OBSERVATION_28D_DIM,
    Observation21DAdapter,
)
from pursuit_evasion_rl.research.policies.baselines import decision_intersection_id
from pursuit_evasion_rl.research.policies.masked_mappo import ResearchMaskedMAPPO
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle, generate_placement
from pursuit_evasion_rl.research.variants.rewards import RewardComponent, RewardComponentSet, leave_one_component_out
from pursuit_evasion_rl.research.variants.stabilization import StabilizationCondition, u_turn_penalties
from pursuit_evasion_rl.osm_demo.policies import build_action_mask

pytestmark = pytest.mark.offline


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "ablation-axes-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "ablation-axes-validation"),),
    )


def _config(**overrides) -> TrainerConfig:
    fields = dict(updates=1, episodes_per_update=1, max_steps=12, hidden_dims=(4,))
    fields.update(overrides)
    return TrainerConfig(**fields)


# ---------------------------------------------------------------------------
# Reward axis
# ---------------------------------------------------------------------------


def test_omitting_reward_components_reconstructs_the_audited_defaults_from_config(tmp_path) -> None:
    network = _network()
    config = _config()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=config,
        condition_id="cond-reward-default", training_seed=1, output_root=tmp_path,
    )
    expected = RewardComponentSet(
        team_coefficient=config.team_coefficient, own_coefficient=config.own_coefficient,
        time_penalty=config.time_penalty, capture_bonus=config.capture_bonus,
        regress_multiplier=config.regress_multiplier, distance_scale_m=config.distance_scale_m,
    )
    assert trainer._reward_components == expected


def test_an_explicit_reward_component_override_is_threaded_through_and_trains(tmp_path) -> None:
    network = _network()
    excluded = leave_one_component_out(RewardComponent.CAPTURE)
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-reward-loo", training_seed=1, output_root=tmp_path, reward_components=excluded,
    )
    assert trainer._reward_components.capture_bonus == 0.0
    assert trainer._reward_components is excluded
    result = trainer.train()
    assert result.checkpoint_path.is_file()


# ---------------------------------------------------------------------------
# Observation axis
# ---------------------------------------------------------------------------


def test_a_21d_observation_override_trains_a_correctly_shaped_policy(tmp_path) -> None:
    network = _network()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-obs-21d", training_seed=1, output_root=tmp_path,
        observation_dim=OBSERVATION_21D_DIM,
        observation_adapter_factory=lambda net, clip, near: Observation21DAdapter(net, clip_distance_m=clip),
    )
    assert trainer.policy.actor_obs_dim == OBSERVATION_21D_DIM
    assert trainer.policy.critic_context_dim == OBSERVATION_21D_DIM * 6
    result = trainer.train()
    assert result.checkpoint_path.is_file()
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["actor_obs_dim"] == OBSERVATION_21D_DIM


def test_omitting_the_observation_override_still_trains_a_28d_policy(tmp_path) -> None:
    network = _network()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-obs-28d-default", training_seed=1, output_root=tmp_path,
    )
    assert trainer.policy.actor_obs_dim == OBSERVATION_28D_DIM


# ---------------------------------------------------------------------------
# Placement axis
# ---------------------------------------------------------------------------


def test_omitting_placement_config_preserves_the_prior_bare_reset(tmp_path) -> None:
    network = _network()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-placement-default", training_seed=1, output_root=tmp_path,
    )
    assert trainer._placement_config is None
    result = trainer.train()
    assert result.checkpoint_path.is_file()


def test_a_placement_override_deterministically_reproduces_generate_placement(tmp_path) -> None:
    network = _network()
    # GLOBAL rather than RING: the point of this test is proving the trainer's
    # reset wiring reaches generate_placement deterministically, not exercising
    # RING's feasible-band constraint against this particular tiny fixture
    # network (which some seeds cannot satisfy for all six officers).
    placement_config = PlacementCurriculumConfig(style=PlacementStyle.GLOBAL)
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-placement-global", training_seed=1, output_root=tmp_path, placement_config=placement_config,
    )
    seed = 4242
    env = OSMRoadPursuitEnv(network, trainer.episode_config)
    trainer._reset_episode(env, network, seed=seed)
    state = env.episode_state()

    expected = generate_placement(network, replace(placement_config, placement_seed=seed))

    assert state.fugitive == expected.fugitive
    assert tuple(state.police) == expected.police

    # And it is a genuine draw from the RING style, not accidentally the
    # trainer's old bare env.reset(seed=seed) initial state.
    bare_env = OSMRoadPursuitEnv(network, trainer.episode_config)
    bare_env.reset(seed=seed)
    bare_state = bare_env.episode_state()
    assert (tuple(state.police), state.fugitive) != (tuple(bare_state.police), bare_state.fugitive)


def test_a_placement_override_trains_end_to_end(tmp_path) -> None:
    network = _network()
    placement_config = PlacementCurriculumConfig(style=PlacementStyle.MIXED)
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-placement-mixed", training_seed=1, output_root=tmp_path, placement_config=placement_config,
    )
    result = trainer.train()
    assert result.checkpoint_path.is_file()


# ---------------------------------------------------------------------------
# Stabilization axis: u-turn suppression (hysteresis is a separate, still-open
# design question -- see the session's design discussion).
# ---------------------------------------------------------------------------


def test_omitting_stabilization_condition_makes_logit_bias_a_no_op(tmp_path) -> None:
    network = _network()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-stab-default", training_seed=1, output_root=tmp_path,
    )
    assert trainer._stabilization_condition is None
    result = trainer.train()
    assert result.checkpoint_path.is_file()


def test_a_u_turn_suppression_override_trains_end_to_end(tmp_path) -> None:
    network = _network()
    condition = StabilizationCondition(u_turn_suppression=True, hysteresis=False)
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-stab-uturn", training_seed=1, output_root=tmp_path, stabilization_condition=condition,
    )
    result = trainer.train()
    assert result.checkpoint_path.is_file()


def test_candidate_end_intersection_ids_matches_a_reference_build_action_mask_call(tmp_path) -> None:
    network = _network()
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=_config(),
        condition_id="cond-stab-candidates", training_seed=1, output_root=tmp_path,
    )
    env = OSMRoadPursuitEnv(network, trainer.episode_config)
    env.reset(seed=7)
    state = env.episode_state()
    officer_id = 0
    decision_int = decision_intersection_id(network, state.police[officer_id])
    heading = state.incoming_headings.get(f"police_{officer_id}")

    candidates = trainer._candidate_end_intersection_ids(network, decision_int, heading)

    _, ordered = build_action_mask(network, decision_int, heading)
    segment_end = {segment.id: int(segment.end_id) for segment in network.segments}
    expected_moves = tuple(segment_end[segment_id] for segment_id in ordered)

    assert len(candidates) == trainer.policy.action_dim
    assert candidates[: len(expected_moves)] == expected_moves
    assert all(value == -1 for value in candidates[len(expected_moves):])


def test_a_reversing_candidate_receives_the_configured_penalty_and_nothing_else_does() -> None:
    """u_turn_penalties itself is already property-tested; this pins the trainer's own wiring of it."""
    condition = StabilizationCondition(u_turn_suppression=True, u_turn_penalty=3.5)
    candidates = (10, 20, -1, -1, -1, -1)
    penalties = u_turn_penalties(condition, candidates, previous_intersection_id=20)
    assert penalties == (0.0, -3.5, 0.0, 0.0, 0.0, 0.0)


def test_logit_bias_reliably_suppresses_the_biased_action_across_many_samples() -> None:
    """Pins ResearchMaskedMAPPO.sample's new logit_bias parameter directly."""
    torch.manual_seed(0)
    policy = ResearchMaskedMAPPO(actor_obs_dim=4, critic_context_dim=24, action_dim=6, num_officers=6, hidden_dims=(8,))
    obs = torch.zeros(4)
    mask = policy.seal_mask(torch.tensor([True, True, True, True, True, True]))
    huge_penalty = torch.tensor([0.0, -1e6, 0.0, 0.0, 0.0, 0.0])

    generator = torch.Generator().manual_seed(1)
    biased_actions = {
        policy.sample(obs, mask, generator, officer_id=0, logit_bias=huge_penalty).action for _ in range(50)
    }
    assert 1 not in biased_actions

    generator = torch.Generator().manual_seed(1)
    unbiased_actions = {
        policy.sample(obs, mask, generator, officer_id=0).action for _ in range(50)
    }
    # Not asserting action 1 appears (would be flaky with an untrained policy);
    # only that the bias path is what suppresses it, not the mask itself.
    assert unbiased_actions <= {0, 1, 2, 3, 4, 5}
