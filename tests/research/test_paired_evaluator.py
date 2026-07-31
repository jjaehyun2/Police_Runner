"""Task 7.1 sealed EpisodeCase, stratum separation and frozen policy replay."""
from __future__ import annotations

from typing import Any

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.environment import OSMRoadPursuitEnv
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, POLICE_COUNT, VehiclePlacement
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.domain import DataKind, EpisodeOutcome, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.evaluation import paired as paired_module
from pursuit_evasion_rl.research.evaluation.paired import (
    CASE_CONTRACT_FIELDS,
    EpisodeCase,
    EpisodeReplayRecord,
    EvaluationMapRef,
    EvaluationProvenance,
    EvaluationStratumLabel,
    FugitiveRNGSpec,
    MapProvenanceIndex,
    PairedContractError,
    PolicyEpisodeRecord,
    SealedCaseBook,
    TerminationConfig,
    assign_stratum,
    compare_cases,
    episode_handle,
    paired_status,
    replay_episode,
    replay_to_episode_result,
    require_paired,
    scenario_outcome_domain,
    seal_episode_case,
)
from pursuit_evasion_rl.research.maps.splits import (
    DataHandle,
    HandleKind,
    SplitScope,
    TuningDataView,
)

pytestmark = pytest.mark.offline

TIMEOUT_STEPS = 6
FROZEN_POLICY_HASH = content_hash("frozen-policy-under-evaluation")


class LowestLegalActionPolicy:
    """Deterministic stand-in: always take the lowest legal action index."""

    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


class HighestLegalActionPolicy:
    """A second frozen policy, so replay hashes are shown to depend on behavior."""

    policy_id = "highest-legal-action-v1"

    def act(self, observation) -> int:
        return max(index for index, legal in enumerate(observation.action_mask) if legal)


@pytest.fixture(scope="module")
def network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


@pytest.fixture(scope="module")
def episode_config() -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=16.0,
        fugitive_speed_mps=9.0,
        capture_radius_m=25.0,
        max_steps=8,
    )


@pytest.fixture(scope="module")
def default_placements(network, episode_config):
    env = OSMRoadPursuitEnv(network, episode_config)
    env.reset(seed=11)
    state = env.episode_state()
    return state.police, state.fugitive


@pytest.fixture(scope="module")
def case(network, episode_config, default_placements) -> EpisodeCase:
    police, fugitive = default_placements
    return seal_episode_case(
        network=network,
        data_kind=DataKind.ACTUAL_OSM_MAP,
        scenario=MapScenario.BOUNDARY_ESCAPE,
        police=police,
        fugitive=fugitive,
        fugitive_rng=FugitiveRNGSpec(stream_id="goal_evader/pcg64", seed=11),
        environment=episode_config,
        termination=TerminationConfig(
            outcome_domain=scenario_outcome_domain(MapScenario.BOUNDARY_ESCAPE),
            timeout_step_limit=TIMEOUT_STEPS,
        ),
    )


def _spare_segment_placement(network, used: tuple[VehiclePlacement, ...]) -> VehiclePlacement:
    taken = {placement.identity for placement in used}
    for segment in network.segments:
        if segment.virtual:
            continue
        candidate = VehiclePlacement(segment_id=segment.id, progress=0.5)
        if candidate.identity not in taken:
            return candidate
    raise AssertionError("fixture network has no spare physical segment placement")


def _perturbed(case: EpisodeCase, network, field: str) -> EpisodeCase:
    """Rebuild the case with exactly one sealed contract field changed."""
    fields: dict[str, Any] = {
        "map_hash": case.map_hash,
        "data_kind": case.data_kind,
        "scenario": case.scenario,
        "police": case.police,
        "fugitive": case.fugitive,
        "fugitive_rng": case.fugitive_rng,
        "environment": case.environment,
        "termination": case.termination,
    }
    spare = _spare_segment_placement(network, case.police + (case.fugitive,))
    if field == "map_hash":
        fields["map_hash"] = content_hash("a different registered map network")
    elif field.startswith("police_"):
        index = int(field.rsplit("_", 1)[1])
        police = list(case.police)
        police[index] = spare
        fields["police"] = tuple(police)
    elif field == "fugitive":
        fields["fugitive"] = spare
    elif field == "fugitive_rng":
        fields["fugitive_rng"] = FugitiveRNGSpec(
            stream_id=case.fugitive_rng.stream_id, seed=case.fugitive_rng.seed + 1
        )
    elif field == "environment":
        fields["environment"] = EpisodeConfig(
            dt_s=case.environment.dt_s,
            police_speed_mps=case.environment.police_speed_mps,
            fugitive_speed_mps=case.environment.fugitive_speed_mps,
            capture_radius_m=case.environment.capture_radius_m + 5.0,
            max_steps=case.environment.max_steps,
        )
    elif field == "termination":
        fields["termination"] = TerminationConfig(
            outcome_domain=case.termination.outcome_domain,
            timeout_step_limit=case.termination.timeout_step_limit - 1,
        )
    else:  # pragma: no cover - guards the parametrization against typos
        raise AssertionError(f"unhandled contract field {field!r}")
    return EpisodeCase(**fields)


def _record(case: EpisodeCase, network, policy, *, training_seed: int = 3) -> PolicyEpisodeRecord:
    return PolicyEpisodeRecord(
        policy_id=policy.policy_id,
        training_seed=training_seed,
        case=case,
        replay=replay_episode(case, policy, network=network),
    )


def _synthetic_case(
    *,
    map_hash: str,
    scenario: MapScenario = MapScenario.BOUNDARY_ESCAPE,
    data_kind: DataKind = DataKind.ACTUAL_OSM_MAP,
    seed: int = 1,
) -> EpisodeCase:
    """A contract-only case; strata never depend on replayability."""
    config = EpisodeConfig(
        dt_s=1.0, police_speed_mps=16.0, fugitive_speed_mps=9.0,
        capture_radius_m=25.0, max_steps=8,
    )
    return EpisodeCase(
        map_hash=map_hash,
        data_kind=data_kind,
        scenario=scenario,
        police=tuple(VehiclePlacement(intersection_id=index) for index in range(POLICE_COUNT)),
        fugitive=VehiclePlacement(intersection_id=POLICE_COUNT),
        fugitive_rng=FugitiveRNGSpec(stream_id="goal_evader/pcg64", seed=seed),
        environment=config,
        termination=TerminationConfig(
            outcome_domain=scenario_outcome_domain(scenario), timeout_step_limit=TIMEOUT_STEPS
        ),
    )


IN_REGION_HASH = content_hash("train-partition-network")
VALIDATION_HASH = content_hash("validation-partition-network")
HELD_OUT_HASH = content_hash("test-partition-network")
CROSS_CITY_HASH = content_hash("zero-shot-city-network")


@pytest.fixture
def provenance_index() -> MapProvenanceIndex:
    return MapProvenanceIndex((
        EvaluationMapRef(IN_REGION_HASH, SplitScope.TRAIN, "daejeon"),
        EvaluationMapRef(VALIDATION_HASH, SplitScope.VALIDATION, "daejeon"),
        EvaluationMapRef(HELD_OUT_HASH, SplitScope.TEST, "daejeon"),
        EvaluationMapRef(CROSS_CITY_HASH, SplitScope.CROSS_CITY, "sejong"),
    ))


# ---------------------------------------------------------------------------
# 1. Replay determinism
# ---------------------------------------------------------------------------


def test_replaying_one_case_twice_produces_identical_replay_hashes(case, network):
    first = replay_episode(case, LowestLegalActionPolicy(), network=network)
    second = replay_episode(case, LowestLegalActionPolicy(), network=network)

    assert first.physical_steps == TIMEOUT_STEPS
    assert len(first.trajectory) == TIMEOUT_STEPS
    assert all(len(step.officer_actions) == POLICE_COUNT for step in first.trajectory)
    assert first.replay_hash == second.replay_hash
    assert first.trajectory == second.trajectory
    assert first.outcome in case.termination.outcome_domain


def test_replay_hash_tracks_policy_behavior_rather_than_being_constant(case, network):
    lowest = replay_episode(case, LowestLegalActionPolicy(), network=network)
    highest = replay_episode(case, HighestLegalActionPolicy(), network=network)

    assert lowest.replay_hash != highest.replay_hash
    assert lowest.trajectory != highest.trajectory


def test_replay_refuses_a_network_that_is_not_the_sealed_map(case, network):
    foreign = EpisodeCase(
        map_hash=content_hash("some other map"),
        data_kind=case.data_kind,
        scenario=case.scenario,
        police=case.police,
        fugitive=case.fugitive,
        fugitive_rng=case.fugitive_rng,
        environment=case.environment,
        termination=case.termination,
    )

    with pytest.raises(ResearchValidationError) as excinfo:
        replay_episode(foreign, LowestLegalActionPolicy(), network=network)

    assert excinfo.value.code == "REPLAY_MAP_MISMATCH"


def test_replay_produces_a_registry_episode_result_in_the_case_stratum(case, network):
    replay = replay_episode(case, LowestLegalActionPolicy(), network=network)

    result = replay_to_episode_result(case, replay, episode_id="ep-1", map_id="daejeon-train")

    assert result.stratum == case.map_stratum
    assert result.outcome is replay.outcome
    assert result.map_hash == case.map_hash


# ---------------------------------------------------------------------------
# 2. Named mismatch for every sealed contract field
# ---------------------------------------------------------------------------


def test_the_contract_field_list_covers_every_sealed_case_field():
    assert CASE_CONTRACT_FIELDS == (
        "map_hash", "data_kind", "scenario",
        "police_0", "police_1", "police_2", "police_3", "police_4", "police_5",
        "fugitive", "fugitive_rng", "environment", "termination",
    )


def test_identical_records_are_paired_and_expose_the_shared_case_hash(case, network):
    left = _record(case, network, LowestLegalActionPolicy())
    right = _record(case, network, HighestLegalActionPolicy())

    status = paired_status(left, right)

    assert status.paired
    assert status.mismatches == ()
    assert status.case_hash == case.case_hash
    assert require_paired(left, right) == case.case_hash


@pytest.mark.parametrize(
    "field",
    [
        "map_hash",
        "police_0", "police_1", "police_2", "police_3", "police_4", "police_5",
        "fugitive", "fugitive_rng", "environment", "termination",
    ],
)
def test_each_perturbed_contract_field_produces_its_own_named_mismatch(case, network, field):
    perturbed = _perturbed(case, network, field)
    assert perturbed.case_hash != case.case_hash

    status = compare_cases(case, perturbed)

    assert not status.paired
    assert status.mismatched_fields == (field,)
    assert status.case_hash is None
    mismatch = status.mismatches[0]
    assert mismatch.left != mismatch.right


@pytest.mark.parametrize(
    "field",
    ["map_hash", "police_3", "fugitive", "fugitive_rng", "environment", "termination"],
)
def test_a_perturbed_field_blocks_paired_marking_and_names_itself(case, network, field):
    left = _record(case, network, LowestLegalActionPolicy())
    perturbed = _perturbed(case, network, field)
    baseline = replay_episode(case, HighestLegalActionPolicy(), network=network)
    right = PolicyEpisodeRecord(
        policy_id=HighestLegalActionPolicy.policy_id,
        training_seed=left.training_seed,
        case=perturbed,
        replay=EpisodeReplayRecord(
            case_hash=perturbed.case_hash,
            policy_id=HighestLegalActionPolicy.policy_id,
            outcome=baseline.outcome,
            physical_steps=baseline.physical_steps,
            trajectory=baseline.trajectory,
        ),
    )

    with pytest.raises(PairedContractError) as excinfo:
        require_paired(left, right)

    assert excinfo.value.code == "PAIRED_CONTRACT_MISMATCH"
    assert excinfo.value.actual == [field]
    assert not excinfo.value.status.paired


def test_a_broken_training_seed_correspondence_is_a_named_mismatch(case, network):
    left = _record(case, network, LowestLegalActionPolicy(), training_seed=3)
    right = _record(case, network, HighestLegalActionPolicy(), training_seed=4)

    status = paired_status(left, right)

    assert not status.paired
    assert status.mismatched_fields == ("training_seed",)


def test_an_incomplete_contract_field_list_fails_closed_instead_of_reporting_a_pair(
    case, network, monkeypatch
):
    monkeypatch.setattr(paired_module, "CASE_CONTRACT_FIELDS", ("map_hash", "data_kind", "scenario"))
    perturbed = _perturbed(case, network, "fugitive_rng")

    with pytest.raises(ResearchValidationError) as excinfo:
        compare_cases(case, perturbed)

    assert excinfo.value.code == "INCOMPLETE_CONTRACT_COMPARISON"


def test_paired_status_is_equivalent_to_complete_case_equality(case, network):
    for field in CASE_CONTRACT_FIELDS:
        if field in {"data_kind", "scenario"}:
            continue
        perturbed = _perturbed(case, network, field)
        assert (perturbed == case) is compare_cases(case, perturbed).paired


# ---------------------------------------------------------------------------
# 3. Interior/Boundary x in-region/held-out/cross-city stratum assignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("map_hash", "expected"),
    [
        (IN_REGION_HASH, EvaluationProvenance.IN_REGION),
        (VALIDATION_HASH, EvaluationProvenance.IN_REGION),
        (HELD_OUT_HASH, EvaluationProvenance.HELD_OUT),
        (CROSS_CITY_HASH, EvaluationProvenance.CROSS_CITY),
    ],
)
@pytest.mark.parametrize(
    "scenario", [MapScenario.INTERIOR_CONTAINED, MapScenario.BOUNDARY_ESCAPE]
)
def test_a_case_is_assigned_to_exactly_one_stratum(provenance_index, map_hash, expected, scenario):
    subject = _synthetic_case(map_hash=map_hash, scenario=scenario)

    label = assign_stratum(subject, provenance_index)

    assert label.provenance is expected
    assert label.scenario is scenario
    assert label.map_stratum == subject.map_stratum
    assert label.stratum_id == f"{subject.data_kind.value}/{scenario.value}/{expected.value}"


def test_interior_and_boundary_cases_never_share_a_stratum(provenance_index):
    interior = _synthetic_case(map_hash=HELD_OUT_HASH, scenario=MapScenario.INTERIOR_CONTAINED)
    boundary = _synthetic_case(map_hash=HELD_OUT_HASH, scenario=MapScenario.BOUNDARY_ESCAPE)

    assert assign_stratum(interior, provenance_index) != assign_stratum(boundary, provenance_index)


def test_every_case_lands_in_exactly_one_stratum_of_the_book(provenance_index):
    cases = (
        _synthetic_case(map_hash=IN_REGION_HASH, seed=1),
        _synthetic_case(map_hash=HELD_OUT_HASH, seed=2),
        _synthetic_case(map_hash=HELD_OUT_HASH, scenario=MapScenario.INTERIOR_CONTAINED, seed=3),
        _synthetic_case(map_hash=CROSS_CITY_HASH, seed=4),
    )
    book = SealedCaseBook(cases=cases, provenance=provenance_index)

    assigned = [item for label in book.strata for item in book.cases_for(label)]

    assert len(book.strata) == 4
    assert sorted(item.case_hash for item in assigned) == sorted(item.case_hash for item in cases)


def test_a_case_with_missing_map_provenance_is_rejected(provenance_index):
    orphan = _synthetic_case(map_hash=content_hash("an unregistered network"))

    with pytest.raises(ResearchValidationError) as excinfo:
        assign_stratum(orphan, provenance_index)

    assert excinfo.value.code == "UNKNOWN_MAP_PROVENANCE"


def test_a_map_claimed_by_two_split_scopes_is_rejected():
    with pytest.raises(ResearchValidationError) as excinfo:
        MapProvenanceIndex((
            EvaluationMapRef(HELD_OUT_HASH, SplitScope.TEST, "daejeon"),
            EvaluationMapRef(HELD_OUT_HASH, SplitScope.TRAIN, "daejeon"),
        ))

    assert excinfo.value.code == "AMBIGUOUS_MAP_PROVENANCE"


def test_a_duplicate_case_cannot_be_counted_twice(provenance_index):
    subject = _synthetic_case(map_hash=IN_REGION_HASH)

    with pytest.raises(ResearchValidationError) as excinfo:
        SealedCaseBook(cases=(subject, subject), provenance=provenance_index)

    assert excinfo.value.code == "DUPLICATE_EPISODE_CASE"


def test_cross_city_strata_reject_synthetic_fixtures():
    with pytest.raises(ResearchValidationError) as excinfo:
        EvaluationStratumLabel(
            data_kind=DataKind.SYNTHETIC_FIXTURE,
            scenario=MapScenario.BOUNDARY_ESCAPE,
            provenance=EvaluationProvenance.CROSS_CITY,
        )

    assert excinfo.value.code == "ZERO_SHOT_REQUIRES_ACTUAL_OSM"


# ---------------------------------------------------------------------------
# 4. Held-out and cross-city cases stay out of the tuning lane
# ---------------------------------------------------------------------------


@pytest.fixture
def book(provenance_index) -> SealedCaseBook:
    return SealedCaseBook(
        cases=(
            _synthetic_case(map_hash=IN_REGION_HASH, seed=1),
            _synthetic_case(map_hash=VALIDATION_HASH, seed=2),
            _synthetic_case(map_hash=HELD_OUT_HASH, seed=3),
            _synthetic_case(map_hash=CROSS_CITY_HASH, seed=4),
        ),
        provenance=provenance_index,
    )


def test_the_tuning_lane_returns_only_in_region_cases(book, provenance_index):
    tuning = book.tuning_cases()

    assert len(tuning) == 2
    scopes = {provenance_index.scope_for(item.map_hash) for item in tuning}
    assert scopes == {SplitScope.TRAIN, SplitScope.VALIDATION}
    assert not scopes & {SplitScope.TEST, SplitScope.CROSS_CITY}


def test_the_tuning_data_view_carries_only_train_and_validation_handles(book):
    view = book.tuning_data_view()

    assert len(view.train) == 1 and len(view.validation) == 1
    assert all(handle.kind is HandleKind.EPISODE for handle in view.train + view.validation)


def test_a_held_out_case_handle_cannot_enter_a_tuning_data_view(book, provenance_index):
    held_out = book.frozen_evaluation_cases(
        EvaluationProvenance.HELD_OUT, frozen_policy_hash=FROZEN_POLICY_HASH
    )
    handle = episode_handle(held_out[0], provenance_index)
    assert handle.scope is SplitScope.TEST

    with pytest.raises(ResearchValidationError) as excinfo:
        TuningDataView(train=(handle,), validation=())

    assert excinfo.value.code == "TUNING_DATA_LEAKAGE"


def test_a_cross_city_case_handle_cannot_enter_a_tuning_data_view(book, provenance_index):
    cross_city = book.frozen_evaluation_cases(
        EvaluationProvenance.CROSS_CITY, frozen_policy_hash=FROZEN_POLICY_HASH
    )
    handle = episode_handle(cross_city[0], provenance_index)

    with pytest.raises(ResearchValidationError) as excinfo:
        TuningDataView(train=(), validation=(handle,))

    assert excinfo.value.code == "TUNING_DATA_LEAKAGE"


def test_the_frozen_evaluation_lane_refuses_in_region_cases(book):
    with pytest.raises(ResearchValidationError) as excinfo:
        book.frozen_evaluation_cases(
            EvaluationProvenance.IN_REGION, frozen_policy_hash=FROZEN_POLICY_HASH
        )

    assert excinfo.value.code == "EVALUATION_LANE_MISMATCH"


def test_the_frozen_evaluation_lane_requires_a_frozen_policy_hash(book):
    with pytest.raises(ResearchValidationError) as excinfo:
        book.frozen_evaluation_cases(EvaluationProvenance.HELD_OUT, frozen_policy_hash="not-a-hash")

    assert excinfo.value.code == "INVALID_CONTENT_HASH"


def test_the_split_scope_survives_into_the_episode_handle(book, provenance_index):
    handles = {
        episode_handle(case, provenance_index).scope
        for label in book.strata
        for case in book.cases_for(label)
    }

    assert handles == {
        SplitScope.TRAIN, SplitScope.VALIDATION, SplitScope.TEST, SplitScope.CROSS_CITY
    }
    assert all(
        isinstance(episode_handle(case, provenance_index), DataHandle)
        for label in book.strata
        for case in book.cases_for(label)
    )
