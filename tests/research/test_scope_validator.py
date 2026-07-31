"""Task 8.6 regressions: reality limits, ethics risks, and Future_Work_System scope."""
from __future__ import annotations

import dataclasses

import pytest

from pursuit_evasion_rl.research.domain import ClaimRecord, ClaimScope, ClaimStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.paper.scope import (
    FUTURE_WORK_DEMO_LABEL,
    REQUIRED_PRE_FIELD_VALIDATION_CATEGORIES,
    REQUIRED_RISK_CATEGORIES,
    FutureWorkArtifact,
    FutureWorkRegistry,
    FutureWorkSystem,
    NonSubstitutionStatement,
    PreFieldValidationCategory,
    PreFieldValidationEntry,
    PreFieldValidationReport,
    PreFieldValidationStatus,
    ProhibitedConnection,
    RiskCategory,
    RiskDisclosure,
    RiskStatement,
    ScopeEthicsBundle,
    ScopeSection,
    ScopeSectionKind,
    SimulationLimitationDisclosure,
    assert_no_cross_category_substitution,
    assert_no_field_readiness_derivation,
    assert_no_future_work_dependency,
)

pytestmark = pytest.mark.offline

PRIMARY_ROOT = "artifacts/primary"
FUTURE_ROOT = "artifacts/future_work"


def _limitations() -> SimulationLimitationDisclosure:
    return SimulationLimitationDisclosure(
        traffic_and_vehicle_dynamics="차량 동역학과 주변 교통류는 모델링되지 않았다.",
        sensor_error="센서 위치 오차와 결측은 0으로 가정한다.",
        communication_latency_and_loss="통신 지연과 패킷 손실은 모델링되지 않았다.",
        human_behavior="운전자와 지휘관의 인간 행동은 모델링되지 않았다.",
        legal_and_operational_constraints="법적·운영 제약은 모델링되지 않았다.",
    )


def _risks(*, omit: RiskCategory | None = None) -> RiskDisclosure:
    return RiskDisclosure(
        statements=tuple(
            RiskStatement(category=category, description=f"{category.value} 위험 서술")
            for category in RiskCategory
            if category is not omit
        )
    )


def _validation_report(
    *,
    omit: PreFieldValidationCategory | None = None,
    statuses: dict[PreFieldValidationCategory, PreFieldValidationStatus] | None = None,
) -> PreFieldValidationReport:
    statuses = statuses or {}
    return PreFieldValidationReport(
        entries=tuple(
            PreFieldValidationEntry(
                category=category,
                validation_id=f"pfv-{category.value}",
                status=statuses.get(category, PreFieldValidationStatus.NOT_PERFORMED),
            )
            for category in PreFieldValidationCategory
            if category is not omit
        )
    )


def _cctv_demo() -> FutureWorkArtifact:
    return FutureWorkArtifact(
        artifact_id="fw-cctv-demo-001",
        system=FutureWorkSystem.CCTV,
        description="CCTV 연동 개념 demo (미구현, 향후 과제)",
        artifact_path=f"{FUTURE_ROOT}/cctv/demo.json",
    )


def _registry(*artifacts: FutureWorkArtifact) -> FutureWorkRegistry:
    return FutureWorkRegistry(
        primary_result_root=PRIMARY_ROOT,
        future_work_root=FUTURE_ROOT,
        artifacts=artifacts or (_cctv_demo(),),
    )


def _sections() -> tuple[ScopeSection, ...]:
    return (
        ScopeSection(
            section_id="sec-scope-current",
            kind=ScopeSectionKind.CURRENT_RESEARCH_SCOPE,
            title="현재 연구 범위",
            body="도로 그래프 위 다중 경찰 추격 정책의 시뮬레이션 평가에 한정한다.",
        ),
        ScopeSection(
            section_id="sec-scope-future",
            kind=ScopeSectionKind.FUTURE_WORK,
            title="향후 과제",
            body="아래 시스템은 본 연구에서 구현하지 않았으며 향후 과제로 남긴다.",
            systems=tuple(FutureWorkSystem),
        ),
    )


def _bundle(*, registry: FutureWorkRegistry | None = None) -> ScopeEthicsBundle:
    return ScopeEthicsBundle(
        limitations=_limitations(),
        non_substitution=NonSubstitutionStatement(),
        risks=_risks(),
        pre_field_validation=_validation_report(),
        future_work=registry or _registry(),
        sections=_sections(),
    )


# --- 완료 검증 1: field-readiness 파생 문구는 핵심 claim에 붙을 수 없다 (AC 17.4) ---


@pytest.mark.parametrize(
    "text",
    [
        "제안 정책은 임계치를 초과했으므로 현장 안전성 향상을 기대할 수 있다.",
        "결과는 관할 구역의 범죄 감소로 이어진다.",
        "시뮬레이션 포획률은 실제 검거율 향상을 의미한다.",
        "The policy yields a real-world capture rate improvement.",
        "These results demonstrate field safety gains and crime reduction.",
    ],
)
def test_field_readiness_derived_phrasing_is_rejected(text: str) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_field_readiness_derivation(text)
    assert excinfo.value.code == "FIELD_READINESS_DERIVATION_BLOCKED"
    assert excinfo.value.actual, "the error must name which banned phrase matched"


def test_threshold_exceeding_result_stated_without_derivation_is_allowed() -> None:
    assert_no_field_readiness_derivation(
        "제안 정책의 시뮬레이션 포획률은 기준선 대비 0.12 높으며 practical threshold를 초과한다."
    )


def test_field_readiness_claim_rejects_derived_text() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        _bundle().field_readiness_claim(
            claim_id="claim-fr-bad",
            text="시뮬레이션 결과는 현장 안전성을 입증한다.",
        )
    assert excinfo.value.code == "FIELD_READINESS_DERIVATION_BLOCKED"


# --- 완료 검증 2: future-work node를 핵심 claim에 연결하면 차단된다 (AC 18.5, 18.8) ---


def test_future_work_evidence_blocks_novelty_claim() -> None:
    registry = _registry()
    novelty = ClaimRecord(
        claim_id="claim-novelty-001",
        text="제안 구성은 비교 축에서 기존 방법과 다른 결과를 보인다.",
        scope=ClaimScope.NOVELTY,
        status=ClaimStatus.PRELIMINARY,
        evidence_ids=("fw-cctv-demo-001",),
        comparison_axes=("observation_scope",),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_future_work_dependency(novelty, registry)
    assert excinfo.value.code == "FUTURE_WORK_DEPENDENCY_BLOCKED"
    assert excinfo.value.actual == ["fw-cctv-demo-001"]
    assert excinfo.value.as_dict()["details"]["connection"] == "novelty_evidence"


@pytest.mark.parametrize(
    "connection",
    [
        ProhibitedConnection.NOVELTY_EVIDENCE,
        ProhibitedConnection.SUCCESS_CRITERION,
        ProhibitedConnection.FIELD_READINESS_GROUND,
    ],
)
def test_future_work_blocked_for_each_prohibited_connection(
    connection: ProhibitedConnection,
) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_future_work_dependency(
            ("run-primary-007", "fw-cctv-demo-001"), _registry(), connection=connection
        )
    assert excinfo.value.code == "FUTURE_WORK_DEPENDENCY_BLOCKED"
    assert excinfo.value.as_dict()["details"]["connection"] == connection.value
    assert excinfo.value.as_dict()["details"]["systems"] == ["cctv"]


@pytest.mark.parametrize(
    "scope",
    [ClaimScope.NOVELTY, ClaimScope.RESULT, ClaimScope.FIELD_READINESS],
)
def test_future_work_blocked_for_every_core_claim_scope(scope: ClaimScope) -> None:
    claim = ClaimRecord(
        claim_id=f"claim-{scope.value}",
        text="제안 구성은 비교 축에서 기준선과 구분되는 결과를 보인다.",
        scope=scope,
        status=(
            ClaimStatus.UNSUPPORTED
            if scope is ClaimScope.FIELD_READINESS
            else ClaimStatus.PRELIMINARY
        ),
        evidence_ids=("fw-cctv-demo-001",),
        comparison_axes=("observation_scope",),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_future_work_dependency(claim, _registry())
    assert excinfo.value.code == "FUTURE_WORK_DEPENDENCY_BLOCKED"


def test_primary_evidence_passes_the_dependency_check() -> None:
    claim = ClaimRecord(
        claim_id="claim-novelty-clean",
        text="제안 구성은 비교 축에서 기준선과 구분되는 결과를 보인다.",
        scope=ClaimScope.NOVELTY,
        status=ClaimStatus.PRELIMINARY,
        evidence_ids=("run-primary-007",),
        comparison_axes=("observation_scope",),
    )
    assert_no_future_work_dependency(claim, _registry())


def test_future_work_demo_must_carry_the_label_and_a_separate_path() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        FutureWorkArtifact(
            artifact_id="fw-anpr-001",
            system=FutureWorkSystem.ANPR,
            description="ANPR 향후 과제",
            artifact_path=f"{FUTURE_ROOT}/anpr/demo.json",
            label="primary_result",
        )
    assert excinfo.value.code == "INVALID_FUTURE_WORK_LABEL"

    inside_primary = FutureWorkArtifact(
        artifact_id="fw-anpr-002",
        system=FutureWorkSystem.ANPR,
        description="ANPR 향후 과제",
        artifact_path=f"{PRIMARY_ROOT}/anpr/demo.json",
    )
    assert inside_primary.label == FUTURE_WORK_DEMO_LABEL
    with pytest.raises(ResearchValidationError) as excinfo:
        _registry(inside_primary)
    assert excinfo.value.code == "FUTURE_WORK_PATH_NOT_SEPARATED"


def test_current_scope_section_may_not_carry_future_work_systems() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        ScopeSection(
            section_id="sec-scope-current",
            kind=ScopeSectionKind.CURRENT_RESEARCH_SCOPE,
            title="현재 연구 범위",
            body="본 연구는 CCTV 연동을 포함한다.",
            systems=(FutureWorkSystem.CCTV,),
        )
    assert excinfo.value.code == "FUTURE_WORK_SYSTEM_IN_CURRENT_SCOPE"


def test_bundle_requires_one_section_of_each_kind() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        ScopeEthicsBundle(
            limitations=_limitations(),
            non_substitution=NonSubstitutionStatement(),
            risks=_risks(),
            pre_field_validation=_validation_report(),
            future_work=_registry(),
            sections=_sections()[:1],
        )
    assert excinfo.value.code == "SCOPE_SECTION_NOT_SEPARATED"


# --- 완료 검증 3: 5개 risk / 6개 validation category 중 하나라도 빠지면 실패 ---


@pytest.mark.parametrize("omitted", list(RiskCategory))
def test_missing_any_risk_category_fails(omitted: RiskCategory) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        _risks(omit=omitted)
    assert excinfo.value.code == "MISSING_RISK_CATEGORY"
    assert excinfo.value.actual == [omitted.value]


def test_risk_categories_are_exactly_the_five_named_ones() -> None:
    assert len(REQUIRED_RISK_CATEGORIES) == 5
    assert {item.value for item in RiskCategory} == {
        "excessive_pursuit_inducement",
        "regional_bias",
        "surveillance_expansion",
        "automation_bias",
        "policy_misuse",
    }


def test_empty_risk_description_fails() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        RiskStatement(category=RiskCategory.AUTOMATION_BIAS, description="   ")
    assert excinfo.value.code == "MISSING_RISK_DESCRIPTION"


@pytest.mark.parametrize("omitted", list(PreFieldValidationCategory))
def test_missing_any_pre_field_validation_category_fails(
    omitted: PreFieldValidationCategory,
) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        _validation_report(omit=omitted)
    assert excinfo.value.code == "MISSING_PRE_FIELD_VALIDATION_CATEGORY"
    assert excinfo.value.actual == [omitted.value]


def test_pre_field_validation_categories_are_exactly_the_six_named_ones() -> None:
    assert len(REQUIRED_PRE_FIELD_VALIDATION_CATEGORIES) == 6
    assert {item.value for item in PreFieldValidationCategory} == {
        "traffic_microsimulation",
        "sensor_uncertainty_assessment",
        "human_in_the_loop_evaluation",
        "safety_verification",
        "legal_review",
        "controlled_pilot",
    }


def test_validation_status_domain_is_exactly_three_values() -> None:
    assert {item.value for item in PreFieldValidationStatus} == {
        "not_performed",
        "failed",
        "passed",
    }


def test_each_validation_category_requires_its_own_identifier() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        PreFieldValidationReport(
            entries=tuple(
                PreFieldValidationEntry(
                    category=category,
                    validation_id="pfv-shared",
                    status=PreFieldValidationStatus.NOT_PERFORMED,
                )
                for category in PreFieldValidationCategory
            )
        )
    assert excinfo.value.code == "DUPLICATE_IDENTIFIER"


@pytest.mark.parametrize("status", ["", None, "pending", ["passed", "failed"]])
def test_validation_entry_requires_exactly_one_known_status(status: object) -> None:
    with pytest.raises(ResearchValidationError):
        PreFieldValidationEntry(
            category=PreFieldValidationCategory.LEGAL_REVIEW,
            validation_id="pfv-legal_review",
            status=status,  # type: ignore[arg-type]
        )


# --- 완료 검증 4: 한 category의 통과가 다른 category를 대신할 수 없다 (AC 17.8) ---


def test_one_passed_category_never_substitutes_for_another() -> None:
    report = _validation_report(
        statuses={PreFieldValidationCategory.SAFETY_VERIFICATION: PreFieldValidationStatus.PASSED}
    )
    assert (
        report.status_of(PreFieldValidationCategory.SAFETY_VERIFICATION)
        is PreFieldValidationStatus.PASSED
    )
    assert (
        report.status_of(PreFieldValidationCategory.LEGAL_REVIEW)
        is PreFieldValidationStatus.NOT_PERFORMED
    )

    safety_id = report.entry_of(PreFieldValidationCategory.SAFETY_VERIFICATION).validation_id
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_cross_category_substitution(
            report,
            category=PreFieldValidationCategory.LEGAL_REVIEW,
            cited_validation_ids=(safety_id,),
        )
    assert excinfo.value.code == "CROSS_CATEGORY_VALIDATION_SUBSTITUTION"

    own_id = report.entry_of(PreFieldValidationCategory.LEGAL_REVIEW).validation_id
    assert_no_cross_category_substitution(
        report,
        category=PreFieldValidationCategory.LEGAL_REVIEW,
        cited_validation_ids=(own_id,),
    )


def test_report_exposes_no_aggregate_readiness_shortcut() -> None:
    report = _validation_report(
        statuses={category: PreFieldValidationStatus.PASSED for category in PreFieldValidationCategory}
    )
    for shortcut in ("all_passed", "is_ready", "passed_count", "ready", "summary_status"):
        assert not hasattr(report, shortcut), f"{shortcut} would collapse six independent reports"
    public = {name for name in dir(report) if not name.startswith("_")}
    assert public == {"entries", "entry_of", "schema_version", "status_of"}


# --- 완료 검증 5: 완전한 bundle은 구성되고, Field_Readiness는 항상 unsupported ---


def test_complete_bundle_constructs_and_is_content_addressed() -> None:
    bundle = _bundle()
    assert len(bundle.risks.statements) == 5
    assert len(bundle.pre_field_validation.entries) == 6
    assert bundle.limitations.sensor_error
    assert "대체하지 않는다" in bundle.non_substitution.statement
    assert bundle.bundle_hash == _bundle().bundle_hash
    assert len(bundle.bundle_hash) == 64


@pytest.mark.parametrize("capture_rate", [0.99, 0.01, 1.0, 0.0])
def test_field_readiness_claim_is_unsupported_for_any_performance(capture_rate: float) -> None:
    claim = _bundle().field_readiness_claim(
        claim_id="claim-field-readiness",
        text=f"시뮬레이션 포획률은 {capture_rate}이며, 본 결과는 현장 적용 근거가 아니다.",
        evidence_ids=("run-primary-007",),
        observed_capture_rate=capture_rate,
    )
    assert claim.scope is ClaimScope.FIELD_READINESS
    assert claim.status is ClaimStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "status",
    [
        ClaimStatus.SUPPORTED,
        ClaimStatus.PRELIMINARY,
        ClaimStatus.INCONCLUSIVE,
        ClaimStatus.FALSIFIED,
        ClaimStatus.INVALID,
    ],
)
def test_domain_layer_still_refuses_any_supported_field_readiness_claim(
    status: ClaimStatus,
) -> None:
    """The bundle's convenience constructor agrees with domain.ClaimRecord's own rule."""
    with pytest.raises(ResearchValidationError) as excinfo:
        ClaimRecord(
            claim_id="claim-field-readiness",
            text="본 결과는 현장 적용 근거가 아니다.",
            scope=ClaimScope.FIELD_READINESS,
            status=status,
            evidence_ids=("run-primary-007",),
        )
    assert excinfo.value.code == "FIELD_READINESS_UNSUPPORTED"

    claim = _bundle().field_readiness_claim(
        claim_id="claim-field-readiness",
        text="본 결과는 현장 적용 근거가 아니다.",
        evidence_ids=("run-primary-007",),
    )
    with pytest.raises(ResearchValidationError) as replaced:
        dataclasses.replace(claim, status=status)
    assert replaced.value.code == "FIELD_READINESS_UNSUPPORTED"


@pytest.mark.parametrize("field_name", list(dataclasses.fields(SimulationLimitationDisclosure))[:5])
def test_missing_any_simulation_limitation_fails(field_name: object) -> None:
    name = field_name.name  # type: ignore[attr-defined]
    values = {
        item.name: "unmodeled" for item in dataclasses.fields(SimulationLimitationDisclosure)
        if item.name != "schema_version"
    }
    values[name] = "  "
    with pytest.raises(ResearchValidationError) as excinfo:
        SimulationLimitationDisclosure(**values)
    assert excinfo.value.code == "MISSING_SIMULATION_LIMITATION"
    assert excinfo.value.path == name


def test_non_substitution_statement_must_say_something() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        NonSubstitutionStatement(statement="")
    assert excinfo.value.code == "MISSING_NON_SUBSTITUTION_STATEMENT"
