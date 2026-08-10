"""Road-dynamics condition tests (mentoring T1): sealed-attribute speeds,
side-car per-episode randomization, travel-time distances, trainer axis."""

from __future__ import annotations

import math

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.environment import OSMRoadPursuitEnv, STAY_ACTION
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, POLICE_COUNT
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.policies.baselines import _Graph
from pursuit_evasion_rl.research.training.trainer import ResearchTrainer, TrainerConfig
from pursuit_evasion_rl.research.traffic.its_client import LinkSpeed, fetch_bbox_speeds, map_speeds_to_segments
from pursuit_evasion_rl.research.variants.remediation import RemediatedStepReward, RoadDistance, remediated_trainer_kwargs
from pursuit_evasion_rl.research.variants.road_dynamics import (
    RoadDynamicsConfig,
    base_segment_speed_mps,
    sample_segment_speeds,
    travel_time_weight,
)

pytestmark = pytest.mark.offline


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _episode_config():
    return EpisodeConfig(
        dt_s=1.0, police_speed_mps=16.0, fugitive_speed_mps=9.0,
        capture_radius_m=25.0, max_steps=12,
    )


def test_base_speeds_positive_and_bounded_for_every_real_segment() -> None:
    network = _network()
    for segment in network.segments:
        if segment.virtual:
            continue
        speed = base_segment_speed_mps(segment)
        assert 0.0 < speed <= 120.0 / 3.6


def test_sampling_is_deterministic_per_seed_and_respects_floor() -> None:
    network = _network()
    config = RoadDynamicsConfig(min_speed_factor=0.5, max_speed_factor=1.0, speed_floor_mps=2.0)
    first = sample_segment_speeds(network, config, seed=11)
    second = sample_segment_speeds(network, config, seed=11)
    other = sample_segment_speeds(network, config, seed=12)
    assert first == second
    assert first != other
    assert all(value >= 2.0 for value in first.values())


def test_its_overlay_wins_over_sampled_speed() -> None:
    network = _network()
    target = next(s.id for s in network.segments if not s.virtual)
    config = RoadDynamicsConfig(its_overlay={target: 3.21})
    speeds = sample_segment_speeds(network, config, seed=1)
    assert speeds[target] == pytest.approx(3.21)


def test_env_segment_speeds_slow_the_police_down() -> None:
    network = _network()
    slow = {s.id: 1.0 for s in network.segments if not s.virtual}
    displacements = {}
    for label, options_speeds in (("fast", None), ("slow", slow)):
        env = OSMRoadPursuitEnv(network, _episode_config())
        options = {"segment_speeds": options_speeds} if options_speeds else None
        env.reset(seed=3, options=options)
        start = [placement_position(network, p) for p in env.episode_state().police]
        actions = {f"police_{i}": 0 for i in range(POLICE_COUNT)}
        actions["fugitive"] = STAY_ACTION
        for _ in range(3):
            if env.outcome is not None:
                break
            env.step(dict(actions))
        end = [placement_position(network, p) for p in env.episode_state().police]
        displacements[label] = sum(math.dist(a, b) for a, b in zip(start, end))
    assert displacements["slow"] < displacements["fast"]


def test_env_rejects_unknown_segment_or_bad_speed() -> None:
    network = _network()
    env = OSMRoadPursuitEnv(network, _episode_config())
    with pytest.raises(Exception):
        env.reset(seed=1, options={"segment_speeds": {999999: 5.0}})
    real = next(s.id for s in network.segments if not s.virtual)
    with pytest.raises(Exception):
        env.reset(seed=1, options={"segment_speeds": {real: 0.0}})


def test_graph_weight_function_changes_distances() -> None:
    network = _network()
    metres = _Graph(network)
    halves = _Graph(network, weight=lambda s: float(s.length_m) / 2.0)
    target = network.intersections[0].id
    base = metres.dist_to(target)
    halved = halves.dist_to(target)
    for node, value in base.items():
        assert halved[node] == pytest.approx(value / 2.0)


def test_road_distance_travel_time_mode_returns_seconds() -> None:
    network = _network()
    state_env = OSMRoadPursuitEnv(network, _episode_config())
    state_env.reset(seed=7)
    state = state_env.episode_state()
    metres = RoadDistance(network)
    seconds = RoadDistance(network, segment_speeds={}, speed_cap_mps=16.0)
    d_m = metres.between(state.police[0], state.fugitive)
    d_s = seconds.between(state.police[0], state.fugitive)
    assert d_s > 0.0
    # traveling d_m metres takes at least d_m / 16 seconds
    assert d_s >= d_m / 16.0 - 1e-6


def test_remediated_reward_with_dynamics_static_step_is_time_penalty() -> None:
    network = _network()
    env = OSMRoadPursuitEnv(network, _episode_config())
    env.reset(seed=7)
    state = env.episode_state()
    reward_fn = RemediatedStepReward(dynamics=RoadDynamicsConfig())
    reward_fn.on_episode(network, sample_segment_speeds(network, RoadDynamicsConfig(), seed=7))
    rewards = reward_fn(network, state, state, False)
    assert len(rewards) == POLICE_COUNT
    for value in rewards:
        assert value == pytest.approx(-reward_fn.components.time_penalty)


def test_trainer_trains_with_dynamics_bundle(tmp_path) -> None:
    network = _network()
    tuning = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "dyn-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "dyn-validation"),),
    )
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=tuning,
        config=TrainerConfig(updates=1, episodes_per_update=1, max_steps=12, hidden_dims=(4,)),
        condition_id="cond-dynamic-bundle", training_seed=9, output_root=tmp_path,
        **remediated_trainer_kwargs(dynamics=RoadDynamicsConfig()),
    )
    assert trainer._road_dynamics_config is not None
    result = trainer.train()
    assert result.checkpoint_path.is_file()
    assert result.all_actions_legal


def test_its_fetch_without_key_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("ITS_API_KEY", raising=False)
    assert fetch_bbox_speeds(min_x=127.0, max_x=127.1, min_y=36.0, max_y=36.1) is None


def test_its_link_matching_maps_to_nearest_segment() -> None:
    network = _network()
    segment = next(s for s in network.segments if not s.virtual and s.geometry_xy)
    xs = [p[0] for p in segment.geometry_xy]
    ys = [p[1] for p in segment.geometry_xy]
    mid = (sum(xs) / len(xs), sum(ys) / len(ys))
    links = [LinkSpeed("L1", 36.0, mid[0], mid[1])]
    matched = map_speeds_to_segments(network, links, to_xy=lambda x, y: (x, y), max_match_m=5.0)
    assert matched.get(segment.id) == pytest.approx(10.0)