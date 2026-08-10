"""SUMO pursuit environment tests.

These exercise the real toolchain (netconvert + a live TraCI connection)
against the Daejeon snapshot this repo ships, because the failures worth
catching here -- an unreachable dispatch, a vehicle SUMO silently retires,
a fugitive spawned next to the map boundary -- only appear in a real
simulation.  They are skipped rather than failed when SUMO is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

sumolib = pytest.importorskip("sumolib")
traci = pytest.importorskip("traci")

from pursuit_evasion_rl.sumo_env.environment import (  # noqa: E402
    ACTION_SIZE, FUGITIVE_ID, POLICE_COUNT, POLICE_IDS, STAY_ACTION,
    EpisodeOutcome, SumoEpisodeConfig, SumoPursuitEnv,
)
from pursuit_evasion_rl.sumo_env.net_builder import (  # noqa: E402
    SumoToolchainError, build_network, largest_cached_snapshot, overpass_to_osm_xml,
)
from pursuit_evasion_rl.sumo_env.observations import (  # noqa: E402
    OBSERVATION_DIM, SumoObservationAdapter,
)
from pursuit_evasion_rl.sumo_env.policies import EncirclementPolicy  # noqa: E402
from pursuit_evasion_rl.sumo_env.traffic import (  # noqa: E402
    TrafficConfig, dispersed_spawn_edges, usable_edges, write_background_routes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.offline


@pytest.fixture(scope="module")
def network():
    try:
        source = largest_cached_snapshot(REPO_ROOT / "cache")
    except SumoToolchainError as error:
        pytest.skip(str(error))
    try:
        return build_network(source, output_root=REPO_ROOT / "cache/sumo")
    except SumoToolchainError as error:
        pytest.skip(str(error))


def _config(**overrides) -> SumoEpisodeConfig:
    fields = dict(
        max_steps=25, seed=7,
        traffic=TrafficConfig(vehicle_count=40, warmup_steps=5, seed=7),
    )
    fields.update(overrides)
    return SumoEpisodeConfig(**fields)


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------


def test_daejeon_snapshot_builds_a_usable_sumo_network(network) -> None:
    assert network.net_path.is_file()
    assert network.edge_count > 500
    assert network.node_count > 200
    # --tls.guess is on, so a real city extract must yield signalised junctions.
    assert network.traffic_light_count > 0


def test_network_build_is_cached_by_content(network) -> None:
    again = build_network(network.source_path, output_root=REPO_ROOT / "cache/sumo")
    assert again.net_path == network.net_path
    assert again.cache_key == network.cache_key


def test_overpass_conversion_emits_well_formed_osm(tmp_path, network) -> None:
    import xml.etree.ElementTree as ET

    destination = tmp_path / "converted.osm.xml"
    nodes, ways = overpass_to_osm_xml(network.source_path, destination)
    assert nodes > 0 and ways > 0
    root = ET.parse(destination).getroot()
    assert root.tag == "osm"
    assert len(root.findall("node")) == nodes
    assert len(root.findall("way")) == ways


# ---------------------------------------------------------------------------
# Background traffic
# ---------------------------------------------------------------------------


def test_background_routes_are_deterministic_per_seed(tmp_path, network) -> None:
    net = sumolib.net.readNet(str(network.net_path))
    config = TrafficConfig(vehicle_count=25, seed=3)
    first, _ = write_background_routes(net, tmp_path / "a.rou.xml", config)
    second, _ = write_background_routes(net, tmp_path / "b.rou.xml", config)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_dispersed_spawn_edges_are_distinct_and_separated(network) -> None:
    net = sumolib.net.readNet(str(network.net_path))
    edges = dispersed_spawn_edges(net, count=POLICE_COUNT, seed=11, min_separation_m=300.0)
    assert len(edges) == POLICE_COUNT
    assert len(set(edges)) == POLICE_COUNT
    assert set(edges).issubset({edge.getID() for edge in usable_edges(net)})


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_reset_places_every_vehicle_in_live_traffic(network) -> None:
    env = SumoPursuitEnv(network, _config(), work_dir=REPO_ROOT / "cache/sumo/runs")
    try:
        state = env.reset(seed=7)
        assert len(state.police) == POLICE_COUNT
        assert state.fugitive is not None
        # Warm-up ran, so ordinary traffic is already on the network.
        assert state.background_count > 0
        assert state.outcome is None
    finally:
        env.close()


def test_fugitive_starts_away_from_the_map_boundary(network) -> None:
    """A boundary spawn would score an escape before any pursuit happened."""
    env = SumoPursuitEnv(network, _config(), work_dir=REPO_ROOT / "cache/sumo/runs")
    try:
        state = env.reset(seed=7)
        edge = env.sumolib_net.getEdge(state.fugitive.edge_id)
        assert edge.getOutgoing(), "fugitive spawned on a dead-end boundary edge"
    finally:
        env.close()


def test_action_mask_always_allows_stay_and_matches_targets(network) -> None:
    env = SumoPursuitEnv(network, _config(), work_dir=REPO_ROOT / "cache/sumo/runs")
    try:
        env.reset(seed=7)
        for officer in POLICE_IDS:
            mask = env.action_mask(officer)
            assert len(mask) == ACTION_SIZE
            assert mask[STAY_ACTION] is True
            assert sum(1 for slot in mask[:-1] if slot) == len(env.legal_targets(officer))
    finally:
        env.close()


def test_set_target_refuses_unreachable_edges(network) -> None:
    env = SumoPursuitEnv(network, _config(), work_dir=REPO_ROOT / "cache/sumo/runs")
    try:
        state = env.reset(seed=7)
        officer = state.police[0]
        assert env.set_target(officer.vehicle_id, "definitely-not-an-edge") is False
        assert env.current_target(officer.vehicle_id) != "definitely-not-an-edge"
    finally:
        env.close()


def test_tracked_vehicles_survive_a_full_episode(network) -> None:
    """Route keep-alive must stop SUMO retiring an agent as 'arrived'."""
    env = SumoPursuitEnv(
        network, _config(max_steps=60), work_dir=REPO_ROOT / "cache/sumo/runs"
    )
    try:
        state = env.reset(seed=13)
        policy = EncirclementPolicy(env)
        while state.outcome is None:
            state = env.step_targets(policy.dispatch(state))
        assert state.outcome != EpisodeOutcome.VOID, "a tracked vehicle disappeared"
    finally:
        env.close()


def test_episode_terminates_with_a_declared_outcome(network) -> None:
    env = SumoPursuitEnv(
        network, _config(max_steps=40), work_dir=REPO_ROOT / "cache/sumo/runs"
    )
    try:
        state = env.reset(seed=9)
        policy = EncirclementPolicy(env)
        while state.outcome is None:
            state = env.step_targets(policy.dispatch(state))
        assert state.outcome in {
            EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE,
            EpisodeOutcome.TIMEOUT, EpisodeOutcome.VOID,
        }
        # Stepping a finished episode must not advance it.
        frozen = env.step_targets({})
        assert frozen.step == state.step
    finally:
        env.close()


def test_capture_requires_the_officer_to_be_within_the_radius(network) -> None:
    env = SumoPursuitEnv(
        network, _config(max_steps=350), work_dir=REPO_ROOT / "cache/sumo/runs"
    )
    try:
        state = env.reset(seed=10)
        policy = EncirclementPolicy(env)
        while state.outcome is None:
            state = env.step_targets(policy.dispatch(state))
        if state.outcome == EpisodeOutcome.CAPTURE:
            assert env.min_separation_m() <= env.config.capture_radius_m
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def test_observation_has_fixed_width_and_finite_values(network) -> None:
    env = SumoPursuitEnv(network, _config(), work_dir=REPO_ROOT / "cache/sumo/runs")
    try:
        env.reset(seed=7)
        adapter = SumoObservationAdapter(env)
        for officer in POLICE_IDS:
            vector = adapter.observe(officer)
            assert vector.shape == (OBSERVATION_DIM,)
            assert all(math.isfinite(float(value)) for value in vector)
    finally:
        env.close()


def test_exit_traffic_reports_vehicle_classes_separately(network) -> None:
    """The class split is the whole point: background traffic must be visible."""
    env = SumoPursuitEnv(
        network, _config(traffic=TrafficConfig(vehicle_count=250, warmup_steps=40, seed=7)),
        work_dir=REPO_ROOT / "cache/sumo/runs",
    )
    try:
        state = env.reset(seed=7)
        adapter = SumoObservationAdapter(env)
        for _ in range(10):
            for officer in POLICE_IDS:
                for exit_info in adapter.exit_traffic(officer):
                    assert exit_info.background >= 0
                    assert 0 <= exit_info.police <= POLICE_COUNT
                    assert 0.0 <= exit_info.occupancy <= 1.0
                    assert 0.0 <= exit_info.mean_speed_ratio <= 1.0
            state = env.step_targets({})
            if state.outcome:
                break

        # Officers' own exits may all be quiet on a 2600-edge map, so assert
        # the counter itself against the busiest edge in the simulation: if
        # background vehicles exist anywhere, the class split must see them.
        connection = env._connection
        busiest, busiest_count = None, 0
        for edge in env.sumolib_net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(":"):
                continue
            try:
                count = connection.edge.getLastStepVehicleNumber(edge_id)
            except Exception:
                continue
            if count > busiest_count:
                busiest, busiest_count = edge_id, count
        assert busiest_count > 0, "no background traffic anywhere in the simulation"
        background, police, fugitive_present = adapter._vehicle_class_counts(busiest)
        assert background + police + (1 if fugitive_present else 0) == busiest_count
        assert background > 0
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_dispatches_reachable_targets_only(network) -> None:
    env = SumoPursuitEnv(network, _config(), work_dir=REPO_ROOT / "cache/sumo/runs")
    try:
        state = env.reset(seed=7)
        policy = EncirclementPolicy(env)
        for _ in range(10):
            targets = policy.dispatch(state)
            for vehicle_id, edge_id in targets.items():
                if not edge_id:
                    continue
                view = next(
                    (item for item in (*state.police, state.fugitive)
                     if item and item.vehicle_id == vehicle_id), None
                )
                if view is None or view.edge_id.startswith(":"):
                    continue
                assert env.reachable(view.edge_id, edge_id)
            state = env.step_targets(targets)
            if state.outcome:
                break
    finally:
        env.close()


def test_policy_is_deterministic_for_a_fixed_seed(network) -> None:
    outcomes = []
    for _ in range(2):
        env = SumoPursuitEnv(
            network, _config(max_steps=40), work_dir=REPO_ROOT / "cache/sumo/runs"
        )
        try:
            state = env.reset(seed=21)
            policy = EncirclementPolicy(env)
            while state.outcome is None:
                state = env.step_targets(policy.dispatch(state))
            outcomes.append((state.outcome, state.step))
        finally:
            env.close()
    assert outcomes[0] == outcomes[1]