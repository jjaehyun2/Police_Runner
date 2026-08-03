"""Property 33 coverage: future-work artifacts cannot satisfy core research claims."""

from __future__ import annotations

from string import ascii_lowercase, digits

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.domain import ClaimRecord, ClaimScope, ClaimStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.paper.scope import (
    FUTURE_WORK_DEMO_LABEL,
    FUTURE_WORK_SYSTEMS,
    FutureWorkArtifact,
    FutureWorkRegistry,
    FutureWorkSystem,
    ProhibitedConnection,
    assert_no_future_work_dependency,
)

# **Property 33: Future-work artifacts cannot satisfy core research claims**
# **Validates: Requirements 18.5-18.6**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_short = st.text(ascii_lowercase + digits, min_size=1, max_size=10)


def _registry(*artifacts: FutureWorkArtifact) -> FutureWorkRegistry:
    return FutureWorkRegistry(
        primary_result_root="results/primary", future_work_root="results/future_work", artifacts=artifacts,
    )


@_PBT_SETTINGS
@given(system=st.sampled_from(list(FutureWorkSystem)), artifact_id=_short)
def test_exactly_the_four_named_systems_are_representable_and_isolated_by_path(system: FutureWorkSystem, artifact_id: str) -> None:
    assert FUTURE_WORK_SYSTEMS == frozenset(FutureWorkSystem)
    assert len(FUTURE_WORK_SYSTEMS) == 4
    artifact = FutureWorkArtifact(
        artifact_id=artifact_id, system=system, description="a demo, not a claim",
        artifact_path=f"results/future_work/{artifact_id}.json",
    )
    registry = _registry(artifact)
    assert registry.system_of(artifact_id) is system
    assert artifact_id in registry.artifact_ids


@_PBT_SETTINGS
@given(artifact_id=_short, label=_short)
def test_a_future_work_artifact_must_carry_the_exact_demo_label(artifact_id: str, label: str) -> None:
    if label == FUTURE_WORK_DEMO_LABEL:
        return
    with pytest.raises(ResearchValidationError) as excinfo:
        FutureWorkArtifact(
            artifact_id=artifact_id, system=FutureWorkSystem.CCTV, description="a demo",
            artifact_path=f"results/future_work/{artifact_id}.json", label=label,
        )
    assert excinfo.value.code == "INVALID_FUTURE_WORK_LABEL"


@_PBT_SETTINGS
@given(artifact_id=_short)
def test_a_future_work_artifact_stored_under_the_primary_root_is_refused(artifact_id: str) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        _registry(
            FutureWorkArtifact(
                artifact_id=artifact_id, system=FutureWorkSystem.ANPR, description="a demo",
                artifact_path=f"results/primary/{artifact_id}.json",
            )
        )
    assert excinfo.value.code == "FUTURE_WORK_PATH_NOT_SEPARATED"


@_PBT_SETTINGS
@given(
    scope=st.sampled_from((ClaimScope.NOVELTY, ClaimScope.RESULT, ClaimScope.FIELD_READINESS)),
    artifact_id=_short,
    attach_via=st.sampled_from(("evidence_ids", "analysis_ids", "comparison_axes")),
)
def test_a_core_claim_that_cites_a_future_work_artifact_is_always_blocked(scope: ClaimScope, artifact_id: str, attach_via: str) -> None:
    artifact = FutureWorkArtifact(
        artifact_id=artifact_id, system=FutureWorkSystem.REALTIME_COMMAND_DASHBOARD, description="a demo",
        artifact_path=f"results/future_work/{artifact_id}.json",
    )
    registry = _registry(artifact)

    kwargs = dict(
        claim_id="claim-1", text="a claim citing a future-work artifact",
        scope=scope, status=ClaimStatus.UNSUPPORTED if scope is ClaimScope.FIELD_READINESS else ClaimStatus.SUPPORTED,
        evidence_ids=(), analysis_ids=(), comparison_axes=(),
    )
    kwargs[attach_via] = (artifact_id,)
    if scope is ClaimScope.NOVELTY:
        kwargs["evidence_ids"] = kwargs["evidence_ids"] or ("evidence-1",)
        kwargs["comparison_axes"] = kwargs["comparison_axes"] or ("axis-1",)
    if not kwargs["evidence_ids"] and scope is not ClaimScope.NOVELTY:
        kwargs["evidence_ids"] = ("evidence-1",) if attach_via != "evidence_ids" else kwargs["evidence_ids"]
    claim = ClaimRecord(**kwargs)

    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_future_work_dependency(claim, registry)
    assert excinfo.value.code == "FUTURE_WORK_DEPENDENCY_BLOCKED"
    assert artifact_id in excinfo.value.actual


@_PBT_SETTINGS
@given(artifact_id=_short)
def test_an_implementation_scope_claim_is_never_screened_a_labelled_demo_may_describe_itself(artifact_id: str) -> None:
    artifact = FutureWorkArtifact(
        artifact_id=artifact_id, system=FutureWorkSystem.DRONE_HETEROGENEOUS_AGENT, description="a demo",
        artifact_path=f"results/future_work/{artifact_id}.json",
    )
    registry = _registry(artifact)
    claim = ClaimRecord(
        claim_id="claim-impl", text="a future-work demo artifact exists and is labelled as such",
        scope=ClaimScope.IMPLEMENTATION, status=ClaimStatus.SUPPORTED, evidence_ids=(artifact_id,),
    )
    assert_no_future_work_dependency(claim, registry)  # must not raise


@_PBT_SETTINGS
@given(artifact_id=_short, connection=st.sampled_from(list(ProhibitedConnection)))
def test_a_bare_identifier_iterable_requires_an_explicit_connection_kind(artifact_id: str, connection: ProhibitedConnection) -> None:
    artifact = FutureWorkArtifact(
        artifact_id=artifact_id, system=FutureWorkSystem.CCTV, description="a demo",
        artifact_path=f"results/future_work/{artifact_id}.json",
    )
    registry = _registry(artifact)
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_future_work_dependency((artifact_id,), registry)
    assert excinfo.value.code == "MISSING_CLASSIFICATION"

    with pytest.raises(ResearchValidationError) as excinfo_blocked:
        assert_no_future_work_dependency((artifact_id,), registry, connection=connection)
    assert excinfo_blocked.value.code == "FUTURE_WORK_DEPENDENCY_BLOCKED"


@_PBT_SETTINGS
@given(claim_evidence_id=_short, artifact_id=_short)
def test_a_claim_citing_no_future_work_artifact_at_all_is_never_blocked(claim_evidence_id: str, artifact_id: str) -> None:
    if claim_evidence_id == artifact_id:
        claim_evidence_id = claim_evidence_id + "-distinct"
    artifact = FutureWorkArtifact(
        artifact_id=artifact_id, system=FutureWorkSystem.ANPR, description="a demo",
        artifact_path=f"results/future_work/{artifact_id}.json",
    )
    registry = _registry(artifact)
    claim = ClaimRecord(
        claim_id="claim-clean", text="a result grounded only in primary evidence",
        scope=ClaimScope.RESULT, status=ClaimStatus.SUPPORTED, evidence_ids=(claim_evidence_id,),
    )
    assert_no_future_work_dependency(claim, registry)  # must not raise
