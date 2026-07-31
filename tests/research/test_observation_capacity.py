"""Task 5.1 capacity accounting regressions for the 21D/28D observation variants."""
from __future__ import annotations

import pytest

from pursuit_evasion_rl.osm_demo.models import DomainValidationError
from pursuit_evasion_rl.research.variants.observations import (
    OBSERVATION_21D_DIM,
    OBSERVATION_28D_DIM,
    actor_capacity,
    compare_observation_capacities,
    find_capacity_matched_hidden_width,
    mlp_flop_estimate,
    mlp_parameter_count,
)

pytestmark = pytest.mark.offline


def test_mlp_parameter_and_flop_counts_match_hand_computation():
    # A single Linear(3, 4) layer: (3 + 1 bias) * 4 = 16 parameters.
    assert mlp_parameter_count(3, 4, (5,)) == (3 + 1) * 5 + (5 + 1) * 4
    assert mlp_flop_estimate(3, 4, (5,)) == 2 * 3 * 5 + 2 * 5 * 4


def test_21d_and_28d_actors_report_the_same_computation_rule_and_differ_only_by_input_width():
    base = actor_capacity(OBSERVATION_21D_DIM, action_dim=6, num_officers=6, hidden_dims=(128, 128))
    aug = actor_capacity(OBSERVATION_28D_DIM, action_dim=6, num_officers=6, hidden_dims=(128, 128))
    assert aug.parameter_count > base.parameter_count
    # Only the first-layer input width differs (7 extra observation features).
    assert aug.parameter_count - base.parameter_count == 7 * 128


def test_default_hidden_width_is_not_capacity_matched_but_search_finds_a_matched_width():
    default_comparison = compare_observation_capacities(
        base_observation_dim=OBSERVATION_21D_DIM,
        matched_observation_dim=OBSERVATION_28D_DIM,
        action_dim=6,
        num_officers=6,
        hidden_dims=(128, 128),
    )
    assert not default_comparison.is_capacity_matched
    assert default_comparison.relative_parameter_difference > 0.01

    matched, widths = find_capacity_matched_hidden_width(
        base_observation_dim=OBSERVATION_21D_DIM,
        base_hidden_dims=(128, 128),
        target_observation_dim=OBSERVATION_28D_DIM,
        action_dim=6,
        num_officers=6,
    )
    assert matched.is_capacity_matched
    assert widths == (125, 125)


def test_search_raises_when_no_candidate_width_reaches_tolerance():
    with pytest.raises(DomainValidationError) as excinfo:
        find_capacity_matched_hidden_width(
            base_observation_dim=OBSERVATION_21D_DIM,
            base_hidden_dims=(128, 128),
            target_observation_dim=OBSERVATION_28D_DIM,
            action_dim=6,
            num_officers=6,
            search_widths=(1,),
        )
    assert excinfo.value.code == "CAPACITY_MATCH_NOT_FOUND"


def test_invalid_dimensions_are_rejected():
    with pytest.raises(DomainValidationError):
        mlp_parameter_count(0, 4, (5,))
    with pytest.raises(DomainValidationError):
        actor_capacity(21, action_dim=6, num_officers=0, hidden_dims=(128,))
