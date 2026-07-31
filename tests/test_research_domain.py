"""Task 1.1 examples for the research domain and canonical foundation."""

import json

import pytest

from pursuit_evasion_rl.research.canonical import (
    CanonicalSerializationError,
    canonical_json,
    canonical_loads,
    content_hash,
)
from pursuit_evasion_rl.research.domain import (
    AnalysisClassification,
    ClaimRecord,
    ClaimScope,
    ClaimStatus,
    Condition,
    DataKind,
    EvidenceRecord,
    EvidenceType,
    ExecutionStatus,
    MapRecord,
    MapScenario,
    ProtocolState,
    ResearchProtocol,
    ResearchQuestion,
    RunRecord,
    ScreeningDecision,
    exactly_one,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError


def test_canonical_round_trip_preserves_failed_not_run_and_null():
    value = {"execution": "failed", "baseline": "not_run", "effect": None}
    encoded = canonical_json(value)
    assert canonical_json(canonical_loads(encoded)) == encoded
    assert json.loads(encoded) == value


def test_hash_is_independent_of_mapping_key_order():
    assert content_hash({"b": 2, "a": {"y": None, "x": 1}}) == content_hash(
        {"a": {"x": 1, "y": None}, "b": 2}
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected_with_structured_error(value):
    with pytest.raises(CanonicalSerializationError) as raised:
        canonical_json({"nested": [value]})
    assert raised.value.code == "CANONICAL_SERIALIZATION_ERROR"
    assert raised.value.path == "$.nested[0]"
    assert len(raised.value.record.content_hash) == 64


def test_exactly_one_rejects_missing_duplicate_and_unknown_classification():
    for invalid in (None, ["test_result", "paper_claim"], "not-an-evidence-type"):
        with pytest.raises(ResearchValidationError) as raised:
            exactly_one(invalid, EvidenceType, path="evidence_type")
        assert raised.value.code in {
            "MISSING_CLASSIFICATION",
            "DUPLICATE_CLASSIFICATION",
            "INVALID_CLASSIFICATION",
        }
    assert exactly_one("included", ScreeningDecision, path="screening") is ScreeningDecision.INCLUDED


def test_persisted_evidence_has_verified_content_hash_and_is_tamper_evident():
    evidence = EvidenceRecord(
        record_id="ev-1",
        evidence_type=EvidenceType.TEST_RESULT,
        producer="pytest",
        created_at_utc="2026-01-01T00:00:00Z",
        method="unit test",
        source="tests/test_research_domain.py",
        extracted_value={"status": "failed", "estimate": None},
        verification="verified",
        limitations=("fixture only",),
    )
    assert len(evidence.content_hash) == 64
    payload = evidence.as_dict()
    assert payload["extracted_value"] == {"status": "failed", "estimate": None}
    with pytest.raises(ResearchValidationError, match="content_hash"):
        EvidenceRecord(**{**payload, "content_hash": "0" * 64})


def _question() -> ResearchQuestion:
    return ResearchQuestion(
        question_id="RQ1",
        analysis_class=AnalysisClassification.CONFIRMATORY,
        primary_outcome="capture risk difference",
        directional_inequality="proposed - baseline > 0",
        practical_threshold=0.05,
        threshold_unit="risk difference",
        decision_rule="CI and adjusted p-value must pass",
    )


def test_representative_models_are_versioned_hashed_and_enum_serialized():
    protocol = ResearchProtocol(
        protocol_id="protocol-1",
        state=ProtocolState.SEALED,
        questions=(_question(),),
        specifications={"missing": None},
        frozen_at_utc="2026-01-01T00:00:00Z",
        signer="researcher",
    )
    condition = Condition(
        condition_id="condition-1",
        policy="mappo",
        observation="osm_topology_v1",
        reward="full",
        placement="mixed",
        uturn="off",
        hysteresis="off",
        scenario=MapScenario.INTERIOR_CONTAINED,
        evader="goal_evader",
        budget={"steps": 100},
    )
    registered_map = MapRecord(
        map_id="map-1",
        data_kind=DataKind.SYNTHETIC_FIXTURE,
        scenario=MapScenario.INTERIOR_CONTAINED,
        network_hash="a" * 64,
        provenance={"generator": "test", "seed": 1},
    )
    run = RunRecord(
        run_id="run-1",
        protocol_hash=protocol.content_hash,
        condition_hash=condition.content_hash,
        execution_status=ExecutionStatus.NOT_RUN,
        status_reason="resource ceiling",
        result=None,
    )
    for model in (protocol, condition, registered_map, run):
        payload = model.as_dict()
        assert payload["schema_version"] == "1.0"
        assert len(payload["content_hash"]) == 64
    assert run.as_dict()["execution_status"] == "not_run"
    assert run.as_dict()["result"] is None


def test_novelty_claim_rejects_priority_language_and_requires_evidence_axis():
    with pytest.raises(ResearchValidationError, match="priority"):
        ClaimRecord(
            claim_id="claim-1",
            text="최초의 OSM pursuit system",
            scope=ClaimScope.NOVELTY,
            status=ClaimStatus.SUPPORTED,
            evidence_ids=("ev-1",),
            comparison_axes=("OSM provenance",),
        )


def test_nested_artifact_hash_is_part_of_parent_identity():
    first = content_hash({"artifact": {"content_hash": "a" * 64}})
    second = content_hash({"artifact": {"content_hash": "b" * 64}})
    assert first != second


def test_duplicate_map_classification_is_rejected_at_model_boundary():
    with pytest.raises(ResearchValidationError) as raised:
        MapRecord(
            map_id="bad-map",
            data_kind=[DataKind.ACTUAL_OSM_MAP, DataKind.SYNTHETIC_FIXTURE],
            scenario=MapScenario.INTERIOR_CONTAINED,
            network_hash="a" * 64,
            provenance={"source": "ambiguous"},
        )
    assert raised.value.code == "DUPLICATE_CLASSIFICATION"
