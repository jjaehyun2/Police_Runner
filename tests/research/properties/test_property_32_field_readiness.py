"""Property 32 coverage: field readiness is invariantly unsupported."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.domain import ClaimRecord, ClaimScope, ClaimStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.paper.scope import (
    FIELD_READINESS_DERIVATION_PATTERNS,
    assert_no_field_readiness_derivation,
    find_field_readiness_derivations,
)

# **Property 32: Field readiness is invariantly unsupported**
# **Validates: Requirements 17.1, 17.4, 19.10**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=20)


@_PBT_SETTINGS
@given(
    capture_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    status=st.sampled_from(list(ClaimStatus)),
)
def test_no_performance_result_can_move_a_field_readiness_claim_off_unsupported(capture_rate: float, status: ClaimStatus) -> None:
    text = f"observed capture rate {capture_rate:.6f} across all seeds"
    if status is ClaimStatus.UNSUPPORTED:
        claim = ClaimRecord(
            claim_id="fr-1", text=text, scope=ClaimScope.FIELD_READINESS, status=status,
            evidence_ids=("evidence-1",),
        )
        assert claim.status is ClaimStatus.UNSUPPORTED
        return
    with pytest.raises(ResearchValidationError) as excinfo:
        ClaimRecord(
            claim_id="fr-1", text=text, scope=ClaimScope.FIELD_READINESS, status=status,
            evidence_ids=("evidence-1",),
        )
    assert excinfo.value.code == "FIELD_READINESS_UNSUPPORTED"


@_PBT_SETTINGS
@given(capture_rate=st.just(1.0))
def test_a_perfect_capture_rate_specifically_still_cannot_escape_unsupported(capture_rate: float) -> None:
    """完了検証: capture rate 1.0 is the adversarial edge case named by the task."""
    with pytest.raises(ResearchValidationError) as excinfo:
        ClaimRecord(
            claim_id="fr-perfect", text=f"capture rate {capture_rate} in every simulated episode",
            scope=ClaimScope.FIELD_READINESS, status=ClaimStatus.SUPPORTED, evidence_ids=("evidence-1",),
        )
    assert excinfo.value.code == "FIELD_READINESS_UNSUPPORTED"


@_PBT_SETTINGS
@given(
    pattern=st.sampled_from(FIELD_READINESS_DERIVATION_PATTERNS),
    prefix=_words,
    suffix=_words,
)
def test_every_named_field_readiness_derivation_pattern_is_caught_wherever_it_appears(pattern: str, prefix: str, suffix: str) -> None:
    text = f"{prefix} {pattern} {suffix}"
    found = find_field_readiness_derivations(text)
    assert pattern in found
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_field_readiness_derivation(text)
    assert excinfo.value.code == "FIELD_READINESS_DERIVATION_BLOCKED"
    assert pattern in excinfo.value.actual


@_PBT_SETTINGS
@given(
    pattern=st.sampled_from(FIELD_READINESS_DERIVATION_PATTERNS),
    upper=st.booleans(),
)
def test_the_derivation_check_is_case_insensitive(pattern: str, upper: bool) -> None:
    text = pattern.upper() if upper and pattern.isascii() else pattern
    assert pattern in find_field_readiness_derivations(text)


@_PBT_SETTINGS
@given(text=_words.filter(lambda t: not any(p.casefold() in t.casefold() for p in FIELD_READINESS_DERIVATION_PATTERNS)))
def test_text_with_no_derivation_language_passes_cleanly(text: str) -> None:
    assert find_field_readiness_derivations(text) == ()
    assert_no_field_readiness_derivation(text)  # must not raise


@_PBT_SETTINGS
@given(
    other_scope=st.sampled_from([s for s in ClaimScope if s is not ClaimScope.FIELD_READINESS]),
    status=st.sampled_from(list(ClaimStatus)),
)
def test_only_field_readiness_scope_is_forced_to_unsupported_other_scopes_are_free(other_scope: ClaimScope, status: ClaimStatus) -> None:
    kwargs = dict(claim_id="c-1", text="a plain simulation result", scope=other_scope, status=status, evidence_ids=("evidence-1",))
    if other_scope is ClaimScope.NOVELTY:
        kwargs["comparison_axes"] = ("axis-1",)
    claim = ClaimRecord(**kwargs)
    assert claim.status is status
