"""Task 1.2 unit tests for legacy asset and prior-result auditing."""

from pathlib import Path

import pytest

from pursuit_evasion_rl.research.audit import (
    AssetAuditor,
    AssetCategory,
    AssetSpec,
    ClaimKind,
    ClaimRegister,
    ClaimRegisterEntry,
    EvidenceRegistry,
    KnownDefect,
    PriorResultInput,
    TestExecution as AuditTestExecution,
    audit_repository,
    build_prior_result,
    prior_from_training_log,
)
from pursuit_evasion_rl.research.canonical import sha256_bytes
from pursuit_evasion_rl.research.domain import (
    DataKind,
    EvidenceRecord,
    EvidenceType,
    PriorResultStatus,
    TestStatus as AuditTestStatus,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError

ROOT = Path(__file__).resolve().parents[1]


def _prior(**changes) -> PriorResultInput:
    values = {
        "result_id": "prior-1",
        "generator_path": "train.py",
        "code_revision": "abc+clean",
        "data_kind": DataKind.SYNTHETIC_FIXTURE,
        "map_hash": "a" * 64,
        "condition": "b" * 64,
        "training_seed": 7,
        "checkpoint_time": "episode_1000",
        "episodes": 100,
        "created_at_utc": "2026-01-01T00:00:00Z",
        "method": "registered evaluation",
        "source": "results.json",
        "observed_value": {"capture_rate": 0.5},
        "independent_training_seeds": 3,
        "evaluation_episodes_per_condition": 100,
        "has_ci95": True,
        "has_spatially_disjoint_split": True,
    }
    values.update(changes)
    return PriorResultInput(**values)


def _evidence(record_id: str, evidence_type: EvidenceType) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        evidence_type=evidence_type,
        producer="pytest",
        created_at_utc="2026-01-01T00:00:00Z",
        method="unit test",
        source="tests/test_research_audit.py",
        extracted_value={"present": True},
        verification="verified",
    )


def test_inventory_records_passed_failed_and_not_tested_with_reasons(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    for relative in ("src/environment.py", "src/policy.py", "src/metrics.py",
                     "tests/test_environment.py", "tests/test_policy.py"):
        (tmp_path / relative).write_text("# fixture\n", encoding="utf-8")
    assets = (
        AssetSpec(AssetCategory.ENVIRONMENT, "src/environment.py", ("tests/test_environment.py",)),
        AssetSpec(AssetCategory.POLICY, "src/policy.py", ("tests/test_policy.py",)),
        AssetSpec(AssetCategory.METRICS, "src/metrics.py"),
    )
    results = {
        "src/environment.py": AuditTestExecution(
            command=("pytest", "tests/test_environment.py"), return_code=0,
            output_hash=sha256_bytes(b"passed"),
        ),
        "src/policy.py": AuditTestExecution(
            command=("pytest", "tests/test_policy.py"), return_code=1,
            output_hash=sha256_bytes(b"failed"),
        ),
    }
    records = AssetAuditor(tmp_path, assets).inventory("revision+dirty", results)
    assert [item.test_status for item in records] == [
        AuditTestStatus.PASSED, AuditTestStatus.FAILED, AuditTestStatus.NOT_TESTED
    ]
    assert all(item.reason for item in records)
    assert all(item.code_revision == "revision+dirty" for item in records)


def test_real_training_log_peak_099_is_fixed_preliminary():
    raw = prior_from_training_log(ROOT)
    record = build_prior_result(raw)
    assert raw.observed_value["value"] == 0.99
    assert raw.observed_value["window_episodes"] == 100
    assert record.status is PriorResultStatus.PRELIMINARY
    assert record.same_training_distribution_moving_window
    assert any("same training distribution" in reason for reason in record.classification_reasons)
    assert record.training_seed == "unknown"
    assert record.unknown_reasons["training_seed"]


def test_escape_bug_always_invalid_even_when_flag_is_false_and_gates_pass():
    defect = KnownDefect(
        code="escape-detection-bug",
        description="escape was checked after progress reset",
        result_changing=False,
    )
    record = build_prior_result(_prior(known_defects=(defect,), force_preliminary=True))
    assert record.status is PriorResultStatus.INVALID
    assert record.classification_reasons[0].startswith("known result-changing defect")


def test_any_declared_result_changing_defect_has_invalid_precedence():
    defect = KnownDefect(
        code="reward-sign-error", description="changed the reported outcome", result_changing=True
    )
    assert build_prior_result(_prior(known_defects=(defect,))).status is PriorResultStatus.INVALID


def test_unknown_provenance_without_field_specific_reason_is_rejected():
    with pytest.raises(ResearchValidationError) as raised:
        build_prior_result(_prior(training_seed="unknown"))
    assert raised.value.code == "UNKNOWN_REASON_REQUIRED"
    assert raised.value.path == "unknown_reasons.training_seed"


def test_incomplete_evidence_record_cannot_be_registered():
    record = EvidenceRecord(
        record_id="empty-observation",
        evidence_type=EvidenceType.EXPERIMENT_OBSERVATION,
        producer="pytest",
        created_at_utc="2026-01-01T00:00:00Z",
        method="unit test",
        source="empty.json",
        extracted_value={},
        verification="unverified",
    )
    with pytest.raises(ResearchValidationError) as raised:
        EvidenceRegistry().register(record)
    assert raised.value.code == "INCOMPLETE_EVIDENCE_RECORD"


@pytest.mark.parametrize(
    ("claim_kind", "required_type", "substitute_type", "status"),
    [
        (ClaimKind.IMPLEMENTATION, EvidenceType.IMPLEMENTATION_EXISTENCE,
         EvidenceType.TEST_RESULT, "exists"),
        (ClaimKind.TEST, EvidenceType.TEST_RESULT,
         EvidenceType.EXPERIMENT_OBSERVATION, "passed"),
        (ClaimKind.PRIOR_RESULT, EvidenceType.EXPERIMENT_OBSERVATION,
         EvidenceType.PAPER_CLAIM, PriorResultStatus.PRELIMINARY),
        (ClaimKind.PAPER, EvidenceType.PAPER_CLAIM,
         EvidenceType.IMPLEMENTATION_EXISTENCE, "supported"),
    ],
)
def test_evidence_types_cannot_substitute_for_claim_kind(
    claim_kind, required_type, substitute_type, status
):
    evidence = EvidenceRegistry()
    evidence.register(_evidence("required", required_type))
    evidence.register(_evidence("substitute", substitute_type))
    register = ClaimRegister(evidence)
    register.register(ClaimRegisterEntry(
        claim_id="valid", claim_kind=claim_kind, statement="typed claim",
        status=status, evidence_ids=("required",),
    ))
    with pytest.raises(ResearchValidationError) as raised:
        register.register(ClaimRegisterEntry(
            claim_id="invalid", claim_kind=claim_kind, statement="wrong evidence",
            status=status, evidence_ids=("substitute",),
        ))
    assert raised.value.code == "EVIDENCE_TYPE_SUBSTITUTION"


def test_repository_audit_skip_tests_keeps_boundaries_and_completes():
    report = audit_repository(ROOT, execute_tests=False)
    assert {item.category for item in report.inventory} == set(AssetCategory)
    assert all(item.test_status is AuditTestStatus.NOT_TESTED for item in report.inventory)
    assert all(item.reason for item in report.inventory)
    assert report.prior_results[0].status is PriorResultStatus.PRELIMINARY
    evidence_types = {item.evidence_type for item in report.evidence_records}
    assert EvidenceType.IMPLEMENTATION_EXISTENCE in evidence_types
    assert EvidenceType.EXPERIMENT_OBSERVATION in evidence_types
    assert report.completeness_passed
