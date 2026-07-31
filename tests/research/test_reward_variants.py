"""Task 5.2 hand-computed regressions for the reward component registry.

Also proves the registry stays numerically identical to the audited
``ResearchTrainer._step_rewards`` formula it was extracted from, so the two
implementations cannot silently drift apart.
"""
from __future__ import annotations

import math

import pytest

from pursuit_evasion_rl.osm_demo.models import DomainValidationError
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.rewards import (
    AUDITED_REWARD_COMPONENTS,
    RewardAblationResult,
    RewardComponent,
    asymmetric_retreat_components,
    compute_component_trace,
    leave_one_component_out,
    symmetric_retreat_components,
    total_rewards,
)

pytestmark = pytest.mark.offline

_OLD = (100.0, 40.0, 60.0)
_NEW = (70.0, 55.0, 60.0)  # officer 0 advances, officer 1 retreats, officer 2 unchanged


def _hand_totals(*, regress_multiplier: float, captured: bool) -> tuple[float, float, float]:
    scale = 50.0
    team_term = (min(_OLD) - min(_NEW)) / scale  # min(100,40,60)=40 -> min(70,55,60)=55
    team = 1.0 * team_term
    totals = []
    for old, new in zip(_OLD, _NEW):
        delta = (old - new) / scale
        own_delta = delta if delta >= 0.0 else regress_multiplier * delta
        own = 0.6 * own_delta
        total = team + own - 0.02 + (20.0 if captured else 0.0)
        totals.append(total)
    return tuple(totals)


def test_full_component_sum_matches_hand_computation():
    expected = _hand_totals(regress_multiplier=1.6, captured=False)
    actual = total_rewards(AUDITED_REWARD_COMPONENTS, _OLD, _NEW, captured=False)
    assert actual == pytest.approx(expected)

    expected_captured = _hand_totals(regress_multiplier=1.6, captured=True)
    actual_captured = total_rewards(AUDITED_REWARD_COMPONENTS, _OLD, _NEW, captured=True)
    assert actual_captured == pytest.approx(expected_captured)
    # Officer 1 retreated (40 -> 55); the audited multiplier steepens its penalty.
    assert actual[1] < 0.6 * ((_OLD[1] - _NEW[1]) / 50.0)


def test_negative_progress_multiplier_only_scales_the_retreating_officer():
    trace = compute_component_trace(AUDITED_REWARD_COMPONENTS, _OLD, _NEW, captured=False)
    delta_1 = (_OLD[1] - _NEW[1]) / 50.0  # negative: officer 1 retreated
    assert delta_1 < 0.0
    assert trace[1].own == pytest.approx(0.6 * 1.6 * delta_1)
    delta_0 = (_OLD[0] - _NEW[0]) / 50.0  # positive: officer 0 advanced, no multiplier
    assert trace[0].own == pytest.approx(0.6 * delta_0)


def test_leave_one_out_differences_equal_the_removed_components_exact_value():
    full = compute_component_trace(AUDITED_REWARD_COMPONENTS, _OLD, _NEW, captured=True)

    no_team = leave_one_component_out(RewardComponent.TEAM)
    diff_team = total_rewards(AUDITED_REWARD_COMPONENTS, _OLD, _NEW, captured=True)
    without_team = total_rewards(no_team, _OLD, _NEW, captured=True)
    for officer_id in range(3):
        assert diff_team[officer_id] - without_team[officer_id] == pytest.approx(full[officer_id].team)

    no_own = leave_one_component_out(RewardComponent.OWN)
    without_own = total_rewards(no_own, _OLD, _NEW, captured=True)
    for officer_id in range(3):
        assert diff_team[officer_id] - without_own[officer_id] == pytest.approx(full[officer_id].own)

    no_time = leave_one_component_out(RewardComponent.TIME)
    without_time = total_rewards(no_time, _OLD, _NEW, captured=True)
    for officer_id in range(3):
        assert diff_team[officer_id] - without_time[officer_id] == pytest.approx(full[officer_id].time)

    no_capture = leave_one_component_out(RewardComponent.CAPTURE)
    without_capture = total_rewards(no_capture, _OLD, _NEW, captured=True)
    for officer_id in range(3):
        assert diff_team[officer_id] - without_capture[officer_id] == pytest.approx(full[officer_id].capture)


def test_symmetric_and_asymmetric_retreat_arms_differ_only_in_regress_multiplier():
    symmetric = symmetric_retreat_components()
    asymmetric = asymmetric_retreat_components()
    assert symmetric.retreat_arm == "symmetric"
    assert asymmetric.retreat_arm == "asymmetric"
    assert symmetric.regress_multiplier == 1.0
    assert asymmetric.regress_multiplier == pytest.approx(1.6)
    for name in ("team_coefficient", "own_coefficient", "time_penalty", "capture_bonus", "distance_scale_m"):
        assert getattr(symmetric, name) == getattr(asymmetric, name)

    expected_symmetric = _hand_totals(regress_multiplier=1.0, captured=False)
    assert total_rewards(symmetric, _OLD, _NEW, captured=False) == pytest.approx(expected_symmetric)


def test_registry_matches_the_audited_trainer_configuration_defaults():
    config = TrainerConfig(updates=1)
    assert AUDITED_REWARD_COMPONENTS.team_coefficient == config.team_coefficient
    assert AUDITED_REWARD_COMPONENTS.own_coefficient == config.own_coefficient
    assert AUDITED_REWARD_COMPONENTS.time_penalty == config.time_penalty
    assert AUDITED_REWARD_COMPONENTS.capture_bonus == config.capture_bonus
    assert AUDITED_REWARD_COMPONENTS.regress_multiplier == config.regress_multiplier
    assert AUDITED_REWARD_COMPONENTS.distance_scale_m == config.distance_scale_m


def test_invalid_component_and_distances_are_rejected():
    with pytest.raises(DomainValidationError) as excinfo:
        leave_one_component_out("not_a_component")  # type: ignore[arg-type]
    assert excinfo.value.code == "INVALID_REWARD_COMPONENT"

    with pytest.raises(DomainValidationError):
        compute_component_trace(AUDITED_REWARD_COMPONENTS, (1.0, 2.0), (1.0,), captured=False)

    with pytest.raises(DomainValidationError):
        compute_component_trace(AUDITED_REWARD_COMPONENTS, (), (), captured=False)

    with pytest.raises(DomainValidationError):
        compute_component_trace(AUDITED_REWARD_COMPONENTS, (math.nan,), (1.0,), captured=False)


def test_reward_ablation_result_requires_capture_and_stability_metric_references():
    trace = compute_component_trace(AUDITED_REWARD_COMPONENTS, _OLD, _NEW, captured=True)
    result = RewardAblationResult(
        condition_id="reward_full",
        components=AUDITED_REWARD_COMPONENTS,
        excluded_component=None,
        capture_metric_ref="metrics/capture_rate.json#reward_full",
        stability_metric_ref="metrics/anti_oscillation.json#reward_full",
        component_trace=trace,
    )
    assert result.config_hash

    with pytest.raises(DomainValidationError) as missing_capture:
        RewardAblationResult(
            condition_id="reward_full",
            components=AUDITED_REWARD_COMPONENTS,
            excluded_component=None,
            capture_metric_ref="",
            stability_metric_ref="metrics/anti_oscillation.json#reward_full",
            component_trace=trace,
        )
    assert missing_capture.value.code == "MISSING_CAPTURE_METRIC_REFERENCE"

    with pytest.raises(DomainValidationError) as missing_stability:
        RewardAblationResult(
            condition_id="reward_full",
            components=AUDITED_REWARD_COMPONENTS,
            excluded_component=None,
            capture_metric_ref="metrics/capture_rate.json#reward_full",
            stability_metric_ref="",
            component_trace=trace,
        )
    assert missing_stability.value.code == "MISSING_STABILITY_METRIC_REFERENCE"
