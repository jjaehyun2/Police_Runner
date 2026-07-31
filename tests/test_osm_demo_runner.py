"""Focused offline tests for the deterministic episode runners.

Covers single-run determinism, batch stream independence and paired
learned-vs-baseline comparison sharing placement and fugitive randomness
(design section 7; Requirements 9.1-9.9, 14.2-14.4).  All networks are built in
process from committed offline fixtures; no external OSM access is performed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    EpisodeConfig,
    EpisodeOutcome,
    EpisodeRecord,
)
from pursuit_evasion_rl.osm_demo.policies import (
    ACTION_DIM,
    STAY_ACTION,
    ActionRecommendation,
    BaselinePolicePolicy,
    PolicePolicy,
    build_action_mask,
    decode_recommendation,
)
from pursuit_evasion_rl.osm_demo.runner import (
    PairedEpisodeRecords,
    derive_stream_seed,
    run_batch,
    run_paired_batch,
    run_single_episode,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
def _grid_network():
    """A small bidirectional Daejeon-shaped grid (no degree splitting needed)."""
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _config(**overrides) -> EpisodeConfig:
    defaults = dict(
        dt_s=1.0,
        police_speed_mps=15.0,
        fugitive_speed_mps=10.0,
        capture_radius_m=20.0,
        max_steps=12,
    )
    defaults.update(overrides)
    return EpisodeConfig(**defaults)


class _StayPolicy:
    """A trivial deterministic policy where every officer stays.

    It satisfies the same :class:`PolicePolicy` protocol as the learned and
    baseline recommenders, so the paired runner can compare two genuinely
    different policies on identical placements and fugitive randomness.
    """

    profile = "stay_test_v0"
    experimental = False

    def __init__(self, network) -> None:
        self.network = network
        self._segments = {segment.id: segment for segment in network.segments}

    def _decision(self, placement) -> int:
        if placement.segment_id is not None:
            return int(self._segments[placement.segment_id].end_id)
        return int(placement.intersection_id)

    def recommend(self, *, police, fugitive, step, max_steps, incoming_headings=None):
        recommendations = []
        for index in range(POLICE_COUNT):
            heading = None if incoming_headings is None else incoming_headings[index]
            decision_id = self._decision(police[index])
            mask, ordered = build_action_mask(self.network, decision_id, heading)
            probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
            probabilities[STAY_ACTION] = 1.0
            recommendations.append(
                decode_recommendation(
                    agent_id=f"police_{index}",
                    network=self.network,
                    decision_intersection_id=decision_id,
                    mask=mask,
                    ordered_segment_ids=ordered,
                    probabilities=probabilities,
                    profile=self.profile,
                    compatibility_report_id=self.profile,
                    action_index=STAY_ACTION,
                )
            )
        return tuple(recommendations)


def _core(record: EpisodeRecord):
    """Policy-independent core of a record for paired comparison."""
    return (record.initial_state, record.transitions, record.outcome, record.terminal_priority)


# ---------------------------------------------------------------------------
# Independent random streams
# ---------------------------------------------------------------------------
def test_derived_stream_seeds_are_independent_per_purpose():
    a = derive_stream_seed(7, 3, "placement")
    b = derive_stream_seed(7, 3, "fugitive")
    c = derive_stream_seed(7, 3, "policy")
    # Distinct purposes yield distinct streams, so adding a new stream cannot
    # perturb an existing one.
    assert len({a, b, c}) == 3
    # Deterministic for the same identity inputs.
    assert derive_stream_seed(7, 3, "placement") == a
    # Different episode index changes the stream.
    assert derive_stream_seed(7, 4, "placement") != a


# ---------------------------------------------------------------------------
# Single-run determinism (Requirements 9.1-9.2, 9.8)
# ---------------------------------------------------------------------------
def test_single_episode_is_reproducible_for_the_same_seed():
    network = _grid_network()
    config = _config()
    policy = BaselinePolicePolicy(network)

    first = run_single_episode(network, config, policy, run_id="run", run_seed=2024, episode_index=0)
    second = run_single_episode(network, config, policy, run_id="run", run_seed=2024, episode_index=0)

    assert first == second  # identical placement, actions, states, events, outcome
    assert first.outcome in tuple(EpisodeOutcome)
    assert len(first.initial_state.police) == POLICE_COUNT
    # Input hashes and priority metadata are recorded on the record.
    assert first.hashes["network"] and first.hashes["config"]
    assert first.hashes["policy_profile"] == policy.profile
    assert first.terminal_priority


def test_single_episode_records_actions_states_and_events():
    network = _grid_network()
    config = _config()
    policy = BaselinePolicePolicy(network)

    record = run_single_episode(network, config, policy, run_id="run", run_seed=11, episode_index=0)

    assert record.transitions, "expected at least one recorded transition"
    for transition in record.transitions:
        # Six police actions plus one fugitive action recorded per transition.
        assert len(transition.actions) == POLICE_COUNT + 1
        assert len(transition.state.police) == POLICE_COUNT
    # Steps are contiguous starting at 1.
    assert [t.step for t in record.transitions] == list(range(1, len(record.transitions) + 1))


# ---------------------------------------------------------------------------
# Batch stream independence (Requirements 9.1-9.2, 9.8)
# ---------------------------------------------------------------------------
def test_batch_episode_matches_the_same_episode_run_alone():
    network = _grid_network()
    config = _config()
    policy = BaselinePolicePolicy(network)

    batch = run_batch(network, config, policy, run_id="run", run_seed=99, episode_count=4)
    assert len(batch) == 4

    # Each episode's independent streams make it identical whether run alone or
    # inside the batch: adding earlier episodes cannot perturb a later one.
    for index in range(4):
        alone = run_single_episode(
            network, config, policy, run_id="run", run_seed=99, episode_index=index
        )
        assert batch[index] == alone

    # The batch is reproducible as a whole.
    again = run_batch(network, config, policy, run_id="run", run_seed=99, episode_count=4)
    assert batch == again


# ---------------------------------------------------------------------------
# Paired comparison (Requirements 14.3-14.4)
# ---------------------------------------------------------------------------
def test_paired_batch_shares_placement_across_policies():
    network = _grid_network()
    config = _config()
    learned = BaselinePolicePolicy(network)
    baseline = _StayPolicy(network)

    pairs = run_paired_batch(
        network, config, learned, baseline, run_id="paired", run_seed=555, episode_count=3
    )
    assert len(pairs) == 3
    for pair in pairs:
        assert isinstance(pair, PairedEpisodeRecords)
        # Both policies see the identical seven-vehicle initial placement.
        assert pair.learned.initial_state == pair.baseline.initial_state
        assert pair.learned.hashes["initial_state"] == pair.baseline.hashes["initial_state"]
        # The two runs are distinguished by policy profile.
        assert pair.learned.hashes["policy_profile"] == learned.profile
        assert pair.baseline.hashes["policy_profile"] == baseline.profile


def test_paired_batch_with_same_policy_is_byte_identical():
    # Using the same policy on both sides proves the placement and fugitive
    # random streams are shared and independent of which side is running.
    network = _grid_network()
    config = _config()
    policy = BaselinePolicePolicy(network)

    pairs = run_paired_batch(
        network, config, policy, BaselinePolicePolicy(network),
        run_id="paired", run_seed=321, episode_count=3,
    )
    for pair in pairs:
        assert _core(pair.learned) == _core(pair.baseline)


def test_paired_batch_is_reproducible():
    network = _grid_network()
    config = _config()
    learned = BaselinePolicePolicy(network)
    baseline = _StayPolicy(network)

    first = run_paired_batch(
        network, config, learned, baseline, run_id="paired", run_seed=7, episode_count=2
    )
    second = run_paired_batch(
        network, config, learned, baseline, run_id="paired", run_seed=7, episode_count=2
    )
    assert first == second


def test_baseline_policy_satisfies_protocol_used_by_runner():
    network = _grid_network()
    assert isinstance(BaselinePolicePolicy(network), PolicePolicy)
    assert isinstance(_StayPolicy(network), PolicePolicy)
