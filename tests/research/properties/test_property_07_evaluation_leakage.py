"""Property 7 coverage: held-out data cannot influence selection, and contamination propagates."""

from __future__ import annotations

from string import ascii_lowercase, digits, hexdigits

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.domain import ExecutionStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope
from pursuit_evasion_rl.research.protocol import (
    HELD_OUT_SCOPES,
    ContaminationRecord,
    SelectionContext,
    SelectionLeakageError,
    SelectionPurpose,
    contaminated_run_lineage,
    guard_selection_context,
    inspect_selection_context,
    propagate_contamination,
)
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance

# **Property 7: Held-out data cannot influence selection, and contamination propagates**
# **Validates: Requirements 15.5-15.6**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_hex64 = st.text(hexdigits.lower(), min_size=64, max_size=64)
_short = st.text(ascii_lowercase + digits, min_size=1, max_size=16)
_clean_scope = st.sampled_from((SplitScope.TRAIN, SplitScope.VALIDATION))
_leaked_scope = st.sampled_from(tuple(HELD_OUT_SCOPES))
_handle_kind = st.sampled_from(list(HandleKind))
_purpose = st.sampled_from(list(SelectionPurpose))


def _handle(scope: SplitScope, kind: HandleKind, identifier: str) -> DataHandle:
    return DataHandle(scope=scope, kind=kind, identifier=identifier)


@st.composite
def clean_handles(draw: st.DrawFn) -> tuple[DataHandle, ...]:
    count = draw(st.integers(min_value=0, max_value=5))
    return tuple(
        _handle(draw(_clean_scope), draw(_handle_kind), draw(_short) + f"-{i}") for i in range(count)
    )


@_PBT_SETTINGS
@given(purpose=_purpose, handles=clean_handles(), context_id=_short)
def test_a_context_touching_only_train_and_validation_data_never_contaminates(purpose, handles, context_id) -> None:
    context = SelectionContext(context_id=context_id, purpose=purpose, handles=handles)
    assert inspect_selection_context(context) is None
    view = guard_selection_context(context)
    assert all(handle.scope is SplitScope.TRAIN for handle in view._train)  # noqa: SLF001
    assert all(handle.scope is SplitScope.VALIDATION for handle in view._validation)  # noqa: SLF001


@_PBT_SETTINGS
@given(
    purpose=_purpose,
    clean=clean_handles(),
    leaked_scope=_leaked_scope,
    leaked_kind=_handle_kind,
    context_id=_short,
    run_id=st.one_of(st.none(), _short),
)
def test_a_context_touching_any_held_out_handle_always_contaminates_and_is_refused(
    purpose, clean, leaked_scope, leaked_kind, context_id, run_id
) -> None:
    leaked = _handle(leaked_scope, leaked_kind, "leaked-0")
    handles = clean + (leaked,)
    context = SelectionContext(context_id=context_id, purpose=purpose, handles=handles, run_id=run_id)

    record = inspect_selection_context(context)
    assert record is not None
    assert leaked in record.leaked_handles
    assert all(handle.scope not in HELD_OUT_SCOPES for handle in record.leaked_handles if handle is not leaked)
    assert leaked_scope in record.leaked_scopes
    if run_id:
        assert run_id in record.excluded_run_ids
    else:
        assert record.excluded_run_ids == ()

    with pytest.raises(SelectionLeakageError) as excinfo:
        guard_selection_context(context)
    assert excinfo.value.contamination.context_id == context_id
    assert excinfo.value.code == "SELECTION_PATH_LEAKAGE"


@_PBT_SETTINGS
@given(purpose=_purpose, context_id=_short)
def test_test_scope_and_cross_city_scope_are_both_individually_sufficient_to_contaminate(purpose, context_id) -> None:
    test_only = SelectionContext(
        context_id=context_id, purpose=purpose,
        handles=(_handle(SplitScope.TEST, HandleKind.MAP, "test-map"),),
    )
    cross_city_only = SelectionContext(
        context_id=context_id, purpose=purpose,
        handles=(_handle(SplitScope.CROSS_CITY, HandleKind.MAP, "cross-city-map"),),
    )
    test_record = inspect_selection_context(test_only)
    cross_city_record = inspect_selection_context(cross_city_only)
    assert test_record is not None and test_record.references_test_data
    assert not test_record.references_target_city
    assert cross_city_record is not None and cross_city_record.references_target_city
    assert not cross_city_record.references_test_data


def _provenance(draw: st.DrawFn) -> RunProvenance:
    return RunProvenance(
        code_hash=draw(_hex64), dirty_tree=draw(st.booleans()), dependency_hash=draw(_hex64),
        runtime=draw(_short), device="cpu", map_hash=draw(_hex64),
        split=draw(st.sampled_from(("train", "validation"))),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        input_hashes={"map": draw(_hex64)},
    )


@st.composite
def provenances(draw: st.DrawFn) -> RunProvenance:
    return _provenance(draw)


def _seal(store: RunManifestStore, run_id: str) -> None:
    store.seal(run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok")


@_PBT_SETTINGS
@given(provenance=provenances(), chain_length=st.integers(min_value=1, max_value=4))
def test_contamination_propagates_to_every_descendant_of_the_tainted_run_and_no_further(
    tmp_path_factory, provenance, chain_length
) -> None:
    store = RunManifestStore(tmp_path_factory.mktemp("manifests"))
    root = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=provenance, run_id="root")
    _seal(store, root.run_id)

    chain_ids = ["root"]
    parent_id = "root"
    for index in range(chain_length):
        child_id = f"child-{index}"
        store.fork(parent_id, new_run_id=child_id)
        _seal(store, child_id)
        chain_ids.append(child_id)
        parent_id = child_id

    # An unrelated sibling rooted independently must never be swept in.
    sibling = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=provenance, run_id="sibling")
    _seal(store, sibling.run_id)

    lineage_descendants = set(contaminated_run_lineage(store, "root"))
    assert lineage_descendants == set(chain_ids)
    assert "sibling" not in lineage_descendants

    leaked = _handle(SplitScope.TEST, HandleKind.MAP, "leaked-map")
    context = SelectionContext(
        context_id="ctx-1", purpose=SelectionPurpose.CHECKPOINT_SELECTION, handles=(leaked,), run_id="root",
    )
    record = inspect_selection_context(context)
    assert record is not None
    propagated = propagate_contamination(record, store, origin_run_id="root")

    assert set(propagated.excluded_run_ids) >= set(chain_ids)
    assert "sibling" not in propagated.excluded_run_ids
    # Propagation only ever grows the excluded set, never drops the original entries.
    assert set(record.excluded_run_ids) <= set(propagated.excluded_run_ids)


@_PBT_SETTINGS
@given(purpose=_purpose, context_id=_short)
def test_a_run_not_present_in_the_store_cannot_be_used_to_propagate_contamination(purpose, context_id, tmp_path_factory) -> None:
    store = RunManifestStore(tmp_path_factory.mktemp("manifests"))
    leaked = _handle(SplitScope.TEST, HandleKind.MAP, "leaked-map")
    context = SelectionContext(context_id=context_id, purpose=purpose, handles=(leaked,), run_id="ghost")
    record = inspect_selection_context(context)
    assert record is not None
    with pytest.raises(ResearchValidationError) as excinfo:
        propagate_contamination(record, store, origin_run_id="ghost")
    assert excinfo.value.code == "RUN_NOT_FOUND"
