"""Property 25 coverage for content-addressed run manifests that fork on mutation."""

from __future__ import annotations

from dataclasses import replace
from string import ascii_lowercase, digits, hexdigits

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.domain import ExecutionStatus, ManifestState
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance

# **Property 25: Run manifests are complete, content-addressed, and fork on mutation**
# **Validates: Requirements 14.1-14.2, 14.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_hex64 = st.text(hexdigits.lower(), min_size=64, max_size=64)
_short = st.text(ascii_lowercase + digits, min_size=1, max_size=16)
_status = st.sampled_from((ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.NOT_RUN))
_provenance_required_fields = ("code_hash", "dirty_tree", "dependency_hash", "runtime", "device", "map_hash", "split", "seed", "input_hashes")


def _provenance(draw: st.DrawFn) -> RunProvenance:
    return RunProvenance(
        code_hash=draw(_hex64),
        dirty_tree=draw(st.booleans()),
        dependency_hash=draw(_hex64),
        runtime=draw(_short),
        device=draw(st.sampled_from(("cpu", "cuda:0", "cuda:1"))),
        map_hash=draw(_hex64),
        split=draw(st.sampled_from(("train", "validation", "held_out", "transfer_city"))),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        input_hashes={"map": draw(_hex64)},
    )


@st.composite
def provenances(draw: st.DrawFn) -> RunProvenance:
    return _provenance(draw)


@_PBT_SETTINGS
@given(provenance=provenances())
def test_create_always_produces_a_draft_manifest_whose_hash_matches_its_content(tmp_path_factory, provenance) -> None:
    store = RunManifestStore(tmp_path_factory.mktemp("manifests"))
    manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=provenance)
    assert manifest.run.manifest_state is ManifestState.DRAFT
    assert not manifest.is_sealed
    reread = store.read(manifest.run_id)
    assert reread.manifest_hash == manifest.manifest_hash
    # Two independently created runs with different run ids never collide in hash.
    other = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=provenance)
    assert other.manifest_hash != manifest.manifest_hash


@_PBT_SETTINGS
@given(provenance=provenances(), missing_field=st.sampled_from(_provenance_required_fields))
def test_a_provenance_missing_any_required_field_is_rejected(provenance, missing_field) -> None:
    base = {name: getattr(provenance, name) for name in _provenance_required_fields}
    invalid: dict = dict(base)
    if missing_field in ("code_hash", "dependency_hash", "runtime", "device", "map_hash", "split"):
        invalid[missing_field] = ""
    elif missing_field == "dirty_tree":
        invalid[missing_field] = "not-a-bool"
    elif missing_field == "seed":
        invalid[missing_field] = -1
    elif missing_field == "input_hashes":
        invalid[missing_field] = {"map": ""}  # a malformed entry, not merely an empty map
    else:  # pragma: no cover - keeps the parametrization honest
        pytest.fail(f"unhandled field {missing_field}")

    with pytest.raises(ResearchValidationError):
        RunProvenance(**invalid)


@_PBT_SETTINGS
@given(provenance=provenances(), status=_status, reason=_short)
def test_sealing_transitions_state_and_a_sealed_manifest_can_never_be_mutated_in_place(
    tmp_path_factory, provenance, status, reason
) -> None:
    store = RunManifestStore(tmp_path_factory.mktemp("manifests"))
    draft = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=provenance)
    sealed = store.seal(draft.run_id, execution_status=status, status_reason=reason)
    assert sealed.is_sealed
    assert sealed.run.execution_status is status
    assert sealed.run.status_reason == reason

    # In-place mutation attempts are always rejected once sealed, regardless of
    # execution status (completed/failed/not_run all seal identically).
    with pytest.raises(ResearchValidationError) as reseal:
        store.seal(draft.run_id, execution_status=status, status_reason="different reason")
    assert reseal.value.code == "SEALED_MANIFEST_MUTATION_BLOCKED"
    with pytest.raises(ResearchValidationError) as update:
        store.update_draft(draft.run_id, provenance=provenance)
    assert update.value.code == "SEALED_MANIFEST_MUTATION_BLOCKED"

    unchanged = store.read(draft.run_id)
    assert unchanged.manifest_hash == sealed.manifest_hash


@_PBT_SETTINGS
@given(
    provenance=provenances(),
    status=_status,
    reason=_short,
    child_provenance=provenances(),
)
def test_any_field_or_artifact_change_after_sealing_forks_a_new_child_identity_and_lineage(
    tmp_path_factory, provenance, status, reason, child_provenance
) -> None:
    store = RunManifestStore(tmp_path_factory.mktemp("manifests"))
    root = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=provenance)
    sealed = store.seal(root.run_id, execution_status=status, status_reason=reason, artifact_hashes={"artifact": "d" * 64})

    child = store.fork(sealed.run_id, provenance=child_provenance)

    # A new run identity, never the original's, carrying explicit lineage.
    assert child.run_id != sealed.run_id
    assert child.run.parent_id == sealed.run_id
    assert child.run.manifest_state is ManifestState.DRAFT
    assert child.provenance == child_provenance

    # The parent is byte-identical to what sealing produced -- forking never
    # touches it.
    reread_parent = store.read(sealed.run_id)
    assert reread_parent.manifest_hash == sealed.manifest_hash
    assert reread_parent.run.artifact_hashes == sealed.run.artifact_hashes

    # Forking a still-draft run is meaningless (it can just be updated) and is
    # rejected outright.
    with pytest.raises(ResearchValidationError) as excinfo:
        store.fork(child.run_id)
    assert excinfo.value.code == "CANNOT_FORK_DRAFT_RUN"


@_PBT_SETTINGS
@given(
    provenance=provenances(),
    status=_status,
    reason=_short,
    depth=st.integers(min_value=1, max_value=4),
)
def test_lineage_chains_of_arbitrary_depth_always_walk_oldest_to_newest(
    tmp_path_factory, provenance, status, reason, depth
) -> None:
    store = RunManifestStore(tmp_path_factory.mktemp("manifests"))
    current = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=provenance)
    ids = [current.run_id]
    for _ in range(depth):
        store.seal(current.run_id, execution_status=status, status_reason=reason)
        current = store.fork(current.run_id, provenance=provenance)
        ids.append(current.run_id)

    lineage = store.lineage(current.run_id)
    assert [item.run_id for item in lineage] == ids
    assert lineage[0].run.parent_id is None
    assert all(lineage[index].run.parent_id == lineage[index - 1].run_id for index in range(1, len(lineage)))
