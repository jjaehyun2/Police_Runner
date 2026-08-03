"""Team-level numeric baselines (GreedyInterceptPolice, directed shortest-path)
replay correctly through replay_episode()'s team_policy bridge.

Prior to this, only single-officer FrozenPolicy objects (act(observation) -> int)
could be replayed; team-level baselines expose recommend(*, police, fugitive,
step, max_steps, incoming_headings=None) instead, which needs every officer's
raw VehiclePlacement at once -- info a single OfficerObservation cannot
losslessly reconstruct. This is Requirement 11.2's conditional-baseline path
actually wired into evaluation, not just declared in the registry.
"""
from __future__ import annotations

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.environment import OSMRoadPursuitEnv
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig
from pursuit_evasion_rl.research.domain import DataKind, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.evaluation.paired import (
    EpisodeCase,
    FugitiveRNGSpec,
    TerminationConfig,
    replay_episode,
    scenario_outcome_domain,
    seal_episode_case,
)
from pursuit_evasion_rl.research.policies.baselines import (
    GreedyInterceptPolice,
    check_policy_interface_conformance,
    directed_shortest_path_policy,
    greedy_intercept_policy,
)

pytestmark = pytest.mark.offline

TIMEOUT_STEPS = 8


@pytest.fixture(scope="module")
def network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


@pytest.fixture(scope="module")
def episode_config() -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=1.0, police_speed_mps=16.0, fugitive_speed_mps=9.0,
        capture_radius_m=25.0, max_steps=TIMEOUT_STEPS,
    )


@pytest.fixture(scope="module")
def case(network, episode_config) -> EpisodeCase:
    env = OSMRoadPursuitEnv(network, episode_config)
    env.reset(seed=7)
    state = env.episode_state()
    return seal_episode_case(
        network=network, data_kind=DataKind.ACTUAL_OSM_MAP, scenario=MapScenario.BOUNDARY_ESCAPE,
        police=state.police, fugitive=state.fugitive,
        fugitive_rng=FugitiveRNGSpec(stream_id="team-baseline-replay", seed=7),
        environment=episode_config,
        termination=TerminationConfig(
            outcome_domain=scenario_outcome_domain(MapScenario.BOUNDARY_ESCAPE),
            timeout_step_limit=TIMEOUT_STEPS,
        ),
    )


def test_greedy_intercept_conforms_to_the_common_policy_interface(network, case):
    police, fugitive = case.police, case.fugitive
    report = check_policy_interface_conformance(
        "greedy_intercept", greedy_intercept_policy(network),
        network=network, police=police, fugitive=fugitive, max_steps=TIMEOUT_STEPS,
    )
    assert report.conforms, report.failure_reason


def test_directed_shortest_path_conforms_to_the_common_policy_interface(network, case):
    police, fugitive = case.police, case.fugitive
    report = check_policy_interface_conformance(
        "directed_shortest_path", directed_shortest_path_policy(network),
        network=network, police=police, fugitive=fugitive, max_steps=TIMEOUT_STEPS,
    )
    assert report.conforms, report.failure_reason


def test_greedy_intercept_replays_through_the_team_policy_bridge(network, case):
    replay = replay_episode(case, network=network, team_policy=greedy_intercept_policy(network))
    assert replay.policy_id == GreedyInterceptPolice(network).profile
    assert replay.physical_steps >= 1
    assert replay.outcome in case.termination.outcome_domain


def test_directed_shortest_path_replays_through_the_team_policy_bridge(network, case):
    baseline = directed_shortest_path_policy(network)
    replay = replay_episode(case, network=network, team_policy=baseline)
    assert replay.policy_id == baseline.profile
    assert replay.physical_steps >= 1
    assert replay.outcome in case.termination.outcome_domain


def test_team_policy_replay_is_deterministic(network, case):
    first = replay_episode(case, network=network, team_policy=greedy_intercept_policy(network))
    second = replay_episode(case, network=network, team_policy=greedy_intercept_policy(network))
    assert first.outcome == second.outcome
    assert first.physical_steps == second.physical_steps
    assert [step.officer_actions for step in first.trajectory] == [
        step.officer_actions for step in second.trajectory
    ]


def test_replay_episode_requires_exactly_one_of_policy_or_team_policy(network, case):
    with pytest.raises(ResearchValidationError) as excinfo:
        replay_episode(case, network=network)
    assert excinfo.value.code == "INVALID_POLICY_ARGUMENT"

    class _StubPolicy:
        policy_id = "stub"

        def act(self, observation) -> int:
            return 0

    with pytest.raises(ResearchValidationError) as excinfo:
        replay_episode(
            case, _StubPolicy(), network=network, team_policy=greedy_intercept_policy(network),
        )
    assert excinfo.value.code == "INVALID_POLICY_ARGUMENT"


def test_team_policy_replay_rejects_a_recommendation_of_the_wrong_cardinality(network, case):
    class _BadTeamPolicy:
        profile = "bad-team-policy"
        experimental = False

        def recommend(self, **kwargs):
            return ()

    with pytest.raises(ResearchValidationError) as excinfo:
        replay_episode(case, network=network, team_policy=_BadTeamPolicy())
    assert excinfo.value.code == "INVALID_TEAM_RECOMMENDATION"
