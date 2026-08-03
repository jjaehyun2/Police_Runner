"""Property 1 tests for fail-closed evidence and prior-result classifications."""

from __future__ import annotations

from dataclasses import dataclass, replace
from string import ascii_lowercase, digits
from typing import Any

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.audit import (
    ClaimKind,
    ClaimRegister,
    ClaimRegisterEntry,
    EvidenceRegistry,
    KnownDefect,
    PriorResultInput,
    build_prior_result,
)
from pursuit_evasion_rl.research.domain import (
    ClaimRecord,
    ClaimScope,
    ClaimStatus,
    DataKind,
    EvidenceRecord,
    EvidenceType,
    PriorResultStatus,
    exactly_one,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError

# **Property 1: Evidence classifications are exclusive, precedence-safe, and type-safe**
# **Validates: Requirements 1.1–1.7, 1.9, 4.6–4.7, 19.1**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)
_TOKEN = st.text(ascii_lowercase + digits, min_size=1, max_size=12)


@dataclass(frozen=True)
class PriorGateCase:
    seeds: int
    episodes: int
    has_ci95: bool
    has_spatial_split: bool
    force_preliminary: bool
    moving_window: bool
    defect: KnownDefect | None

@st.composite
def prior_gate_cases(draw: st.DrawFn) -> PriorGateCase:
    defect_kind = draw(st.sampled_from(("none", "escape", "result_changing")))
    if defect_kind == "escape":
        defect = KnownDefect(
            code=draw(st.sampled_from(("escape-bug", "escape_detection_bug", "pre-fix escape bug"))),
            description="escape outcome could change",
            result_changing=False,
        )
    elif defect_kind == "result_changing":
        defect = KnownDefect(
            code=f"generated-{draw(_TOKEN)}",
            description="generated result-changing defect",
            result_changing=True,
        )
    else:
        defect = None
    return PriorGateCase(
        seeds=draw(st.integers(min_value=0, max_value=6)),
        episodes=draw(st.integers(min_value=0, max_value=200)),
        has_ci95=draw(st.booleans()),
        has_spatial_split=draw(st.booleans()),
        force_preliminary=draw(st.booleans()),
        moving_window=draw(st.booleans()),
        defect=defect,
    )


def _prior(case: PriorGateCase, token: str = "record") -> PriorResultInput:
    return PriorResultInput(
        result_id=f"prior-{token}",
        generator_path="train.py",
        code_revision="abc123+clean",
        data_kind=DataKind.SYNTHETIC_FIXTURE,
        map_hash="a" * 64,
        condition="b" * 64,
        training_seed=7,
        checkpoint_time="episode_1000",
        episodes=case.episodes,
        created_at_utc="2026-01-01T00:00:00Z",
        method="registered paired evaluation",
        source="results.json",
        observed_value={"capture_rate": 0.5},
        known_defects=(() if case.defect is None else (case.defect,)),
        independent_training_seeds=case.seeds,
        evaluation_episodes_per_condition=case.episodes,
        has_ci95=case.has_ci95,
        has_spatially_disjoint_split=case.has_spatial_split,
        same_training_distribution_moving_window=case.moving_window,
        force_preliminary=case.force_preliminary,
    )


def _expected_status(case: PriorGateCase) -> PriorResultStatus:
    if case.defect is not None and case.defect.invalidates_result:
        return PriorResultStatus.INVALID
    if (
        case.seeds < 3
        or case.episodes < 100
        or not case.has_ci95
        or not case.has_spatial_split
        or case.force_preliminary
        or case.moving_window
    ):
        return PriorResultStatus.PRELIMINARY
    return PriorResultStatus.VERIFIED


@_PBT_SETTINGS
@given(case=prior_gate_cases(), token=_TOKEN)
def test_prior_result_status_is_exclusive_and_matches_all_precedence_gates(
    case: PriorGateCase, token: str
) -> None:
    record = build_prior_result(_prior(case, token))
    expected = _expected_status(case)

    assert record.status is expected
    assert sum(record.status is status for status in PriorResultStatus) == 1


@_PBT_SETTINGS
@given(
    defect=st.one_of(
        st.sampled_from(("escape-bug", "escape_detection_bug", "pre-fix escape bug")).map(
            lambda code: KnownDefect(code, "escape defect", False)
        ),
        _TOKEN.map(lambda code: KnownDefect(f"result-{code}", "changed result", True)),
    ),
    seeds=st.integers(min_value=0, max_value=10),
    episodes=st.integers(min_value=0, max_value=1_000),
    ci=st.booleans(),
    split=st.booleans(),
)
def test_escape_and_result_changing_defects_always_take_invalid_precedence(
    defect: KnownDefect, seeds: int, episodes: int, ci: bool, split: bool
) -> None:
    case = PriorGateCase(seeds, episodes, ci, split, True, True, defect)
    record = build_prior_result(_prior(case))

    assert record.status is PriorResultStatus.INVALID
    assert "takes precedence" in record.classification_reasons[0]


_STATUS_DEFECTS = st.sampled_from(("missing", "duplicate", "mismatch"))


@_PBT_SETTINGS
@given(case=prior_gate_cases(), defect=_STATUS_DEFECTS)
def test_mutated_prior_status_record_is_rejected_at_the_classification_gate(
    case: PriorGateCase, defect: str
) -> None:
    record = build_prior_result(_prior(case))
    if defect == "missing":
        bad_status: Any = None
        expected_code = "MISSING_CLASSIFICATION"
    elif defect == "duplicate":
        bad_status = (record.status, record.status)
        expected_code = "DUPLICATE_CLASSIFICATION"
    else:
        bad_status = next(status for status in PriorResultStatus if status is not record.status)
        expected_code = "PRIOR_RESULT_STATUS_MISMATCH"

    with pytest.raises(ResearchValidationError) as raised:
        replace(record, status=bad_status, content_hash=None)
    assert raised.value.code == expected_code


def _evidence(record_id: str, evidence_type: Any) -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        evidence_type=evidence_type,
        producer="hypothesis",
        created_at_utc="2026-01-01T00:00:00Z",
        method="generated property case",
        source=__file__,
        extracted_value={"present": True},
        verification="verified",
    )


_EVIDENCE_CLASSIFICATION_DEFECTS = st.sampled_from(
    ("missing", "duplicate", "invalid", "provenance_mismatch")
)


@_PBT_SETTINGS
@given(evidence_type=st.sampled_from(tuple(EvidenceType)), defect=_EVIDENCE_CLASSIFICATION_DEFECTS)
def test_evidence_type_gate_rejects_each_single_exactly_one_defect(
    evidence_type: EvidenceType, defect: str
) -> None:
    expected_codes = {
        "missing": "MISSING_CLASSIFICATION",
        "duplicate": "DUPLICATE_CLASSIFICATION",
        "invalid": "INVALID_CLASSIFICATION",
        "provenance_mismatch": "PROVENANCE_MISMATCH",
    }
    with pytest.raises(ResearchValidationError) as raised:
        if defect == "provenance_mismatch":
            exactly_one(
                evidence_type,
                EvidenceType,
                path="evidence_type",
                provenance_consistent=False,
            )
        else:
            bad_value: Any = {
                "missing": None,
                "duplicate": (evidence_type, evidence_type),
                "invalid": "not-an-evidence-type",
            }[defect]
            _evidence("classification-defect", bad_value)
    assert raised.value.code == expected_codes[defect]


_EXPECTED_EVIDENCE = {
    ClaimKind.IMPLEMENTATION: EvidenceType.IMPLEMENTATION_EXISTENCE,
    ClaimKind.TEST: EvidenceType.TEST_RESULT,
    ClaimKind.PRIOR_RESULT: EvidenceType.EXPERIMENT_OBSERVATION,
    ClaimKind.PAPER: EvidenceType.PAPER_CLAIM,
}


def _claim_status(kind: ClaimKind) -> str | PriorResultStatus:
    return PriorResultStatus.PRELIMINARY if kind is ClaimKind.PRIOR_RESULT else "recorded"


@st.composite
def claim_type_cases(draw: st.DrawFn) -> tuple[ClaimKind, EvidenceType, EvidenceType]:
    kind = draw(st.sampled_from(tuple(ClaimKind)))
    required = _EXPECTED_EVIDENCE[kind]
    substitute = draw(st.sampled_from(tuple(item for item in EvidenceType if item is not required)))
    return kind, required, substitute


@_PBT_SETTINGS
@given(case=claim_type_cases())
def test_evidence_types_never_substitute_across_claim_boundaries(
    case: tuple[ClaimKind, EvidenceType, EvidenceType]
) -> None:
    kind, required, substitute = case
    evidence = EvidenceRegistry()
    evidence.register(_evidence("required", required))
    evidence.register(_evidence("substitute", substitute))
    claims = ClaimRegister(evidence)
    claims.register(
        ClaimRegisterEntry("valid", kind, "typed claim", _claim_status(kind), ("required",))
    )

    with pytest.raises(ResearchValidationError) as raised:
        claims.register(
            ClaimRegisterEntry(
                "invalid", kind, "substituted claim", _claim_status(kind), ("substitute",)
            )
        )
    assert raised.value.code == "EVIDENCE_TYPE_SUBSTITUTION"


_RECORD_LINK_DEFECTS = st.sampled_from(("duplicate_record", "missing_record", "wrong_type"))


@_PBT_SETTINGS
@given(token=_TOKEN, defect=_RECORD_LINK_DEFECTS)
def test_duplicate_missing_and_inconsistent_evidence_records_fail_closed(
    token: str, defect: str
) -> None:
    evidence = EvidenceRegistry()
    record = _evidence(f"evidence-{token}", EvidenceType.TEST_RESULT)
    evidence.register(record)

    if defect == "duplicate_record":
        with pytest.raises(ResearchValidationError) as raised:
            evidence.register(record)
        assert raised.value.code == "DUPLICATE_IDENTIFIER"
        return

    evidence_id = "absent" if defect == "missing_record" else record.record_id
    kind = ClaimKind.TEST if defect == "missing_record" else ClaimKind.IMPLEMENTATION
    with pytest.raises(ResearchValidationError) as raised:
        ClaimRegister(evidence).register(
            ClaimRegisterEntry("claim", kind, "generated claim", "recorded", (evidence_id,))
        )
    assert raised.value.code == (
        "MISSING_EVIDENCE" if defect == "missing_record" else "EVIDENCE_TYPE_SUBSTITUTION"
    )


_NOVELTY_DEFECTS = st.sampled_from(("missing_evidence", "missing_axis", "priority_language"))


@_PBT_SETTINGS
@given(token=_TOKEN, defect=_NOVELTY_DEFECTS)
def test_novelty_claim_requires_evidence_axis_and_nonpriority_language(
    token: str, defect: str
) -> None:
    values = {
        "claim_id": f"novelty-{token}",
        "text": "Bounded comparison on a registered research axis",
        "scope": ClaimScope.NOVELTY,
        "status": ClaimStatus.SUPPORTED,
        "evidence_ids": (f"evidence-{token}",),
        "comparison_axes": (f"axis-{token}",),
    }
    expected_code = "INCOMPLETE_NOVELTY_EVIDENCE"
    if defect == "missing_evidence":
        values["evidence_ids"] = ()
    elif defect == "missing_axis":
        values["comparison_axes"] = ()
    else:
        values["text"] = "The first and only pursuit system"
        expected_code = "FORBIDDEN_PRIORITY_CLAIM"

    with pytest.raises(ResearchValidationError) as raised:
        ClaimRecord(**values)
    assert raised.value.code == expected_code
