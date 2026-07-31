"""Task 6.1 regressions for the content-addressed, immutable run manifest store."""
from __future__ import annotations

from pathlib import Path

import pytest

from pursuit_evasion_rl.research.domain import ExecutionStatus, ManifestState
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.runs.manifest import (
    RunManifestStore,
    RunProvenance,
    generate_run_id,
)

pytestmark = pytest.mark.offline


def _provenance(**overrides) -> RunProvenance:
    fields = {
        "code_hash": "a" * 64,
        "dirty_tree": False,
        "dependency_hash": "b" * 64,
        "runtime": "python3.11+torch2.8.0+cpu",
        "device": "cpu",
        "map_hash": "c" * 64,
        "split": "train",
        "seed": 7,
        "input_hashes": {"map": "c" * 64},
    }
    fields.update(overrides)
    return RunProvenance(**fields)


def test_create_starts_draft_and_seal_transitions_to_sealed(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    assert manifest.run.manifest_state is ManifestState.DRAFT
    assert not manifest.is_sealed

    sealed = store.seal(
        manifest.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="episode budget exhausted",
        artifact_hashes={"checkpoint": "d" * 64}, output_hashes={"checkpoint": "d" * 64},
        result={"capture_rate": 0.62},
    )
    assert sealed.is_sealed
    assert sealed.run.execution_status is ExecutionStatus.COMPLETED
    assert sealed.sealed_at_utc


@pytest.mark.parametrize("status", [ExecutionStatus.FAILED, ExecutionStatus.NOT_RUN])
def test_failed_and_not_run_seal_identically_to_completed(tmp_path: Path, status: ExecutionStatus) -> None:
    store = RunManifestStore(tmp_path)
    manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    sealed = store.seal(manifest.run_id, execution_status=status, status_reason="reason recorded", result=None)
    assert sealed.is_sealed
    assert sealed.run.execution_status is status
    assert sealed.run.result is None
    assert sealed.run.status_reason == "reason recorded"


def test_sealed_manifest_cannot_be_resealed_or_updated_in_place(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    store.seal(manifest.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok")

    with pytest.raises(ResearchValidationError) as reseal:
        store.seal(manifest.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok again")
    assert reseal.value.code == "SEALED_MANIFEST_MUTATION_BLOCKED"

    with pytest.raises(ResearchValidationError) as update:
        store.update_draft(manifest.run_id, provenance=_provenance(seed=99))
    assert update.value.code == "SEALED_MANIFEST_MUTATION_BLOCKED"

    # On-disk content is untouched by either rejected attempt.
    reread = store.read(manifest.run_id)
    assert reread.provenance.seed == 7


def test_draft_manifests_can_still_be_updated_in_place(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    updated = store.update_draft(manifest.run_id, provenance=_provenance(seed=123))
    assert updated.provenance.seed == 123
    assert store.read(manifest.run_id).provenance.seed == 123


def test_field_or_artifact_mutation_after_sealing_forks_a_new_child_not_the_original(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    original = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    sealed = store.seal(
        original.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok",
        artifact_hashes={"checkpoint": "d" * 64}, result={"capture_rate": 0.5},
    )

    child = store.fork(sealed.run_id, provenance=_provenance(seed=999))

    assert child.run_id != sealed.run_id
    assert child.run.parent_id == sealed.run_id
    assert child.run.manifest_state is ManifestState.DRAFT
    assert child.provenance.seed == 999

    # The original, sealed run is completely untouched by the fork.
    unchanged = store.read(sealed.run_id)
    assert unchanged.manifest_hash == sealed.manifest_hash
    assert unchanged.provenance.seed == 7
    assert unchanged.run.result == {"capture_rate": 0.5}


def test_forking_a_draft_run_is_rejected(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    with pytest.raises(ResearchValidationError) as excinfo:
        store.fork(manifest.run_id)
    assert excinfo.value.code == "CANNOT_FORK_DRAFT_RUN"


def test_lineage_walks_from_oldest_ancestor_to_the_requested_run(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    root = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    store.seal(root.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok")
    child = store.fork(root.run_id, provenance=_provenance(seed=2))
    store.seal(child.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok")
    grandchild = store.fork(child.run_id, provenance=_provenance(seed=3))

    lineage = store.lineage(grandchild.run_id)
    assert [item.run_id for item in lineage] == [root.run_id, child.run_id, grandchild.run_id]


def test_run_id_collision_and_missing_run_are_rejected(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    run_id = generate_run_id()
    store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance(), run_id=run_id)
    with pytest.raises(ResearchValidationError) as collision:
        store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance(), run_id=run_id)
    assert collision.value.code == "RUN_ID_COLLISION"

    with pytest.raises(ResearchValidationError) as missing:
        store.read("does-not-exist")
    assert missing.value.code == "RUN_NOT_FOUND"


def test_assert_writable_blocks_writes_under_a_sealed_run_directory(tmp_path: Path) -> None:
    store = RunManifestStore(tmp_path)
    manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
    run_directory = store._path(manifest.run_id).parent
    store.assert_writable(manifest.run_id, run_directory / "checkpoint.pt")  # draft: allowed

    store.seal(manifest.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok")
    with pytest.raises(ResearchValidationError) as excinfo:
        store.assert_writable(manifest.run_id, run_directory / "checkpoint.pt")
    assert excinfo.value.code == "SEALED_RUN_ARTIFACT_MUTATION_BLOCKED"


def test_manifest_hash_round_trips_through_disk() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = RunManifestStore(tmp)
        manifest = store.create(protocol_hash="p" * 64, condition_hash="q" * 64, provenance=_provenance())
        sealed = store.seal(
            manifest.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="ok",
            artifact_hashes={"checkpoint": "d" * 64}, result={"capture_rate": 0.5},
        )
        reread = store.read(manifest.run_id)
        assert reread.manifest_hash == sealed.manifest_hash
        assert reread.run.content_hash == sealed.run.content_hash


def test_invalid_provenance_fields_are_rejected() -> None:
    with pytest.raises(ResearchValidationError):
        _provenance(code_hash="")
    with pytest.raises(ResearchValidationError):
        _provenance(seed=-1)
    with pytest.raises(ResearchValidationError):
        _provenance(dirty_tree="yes")  # type: ignore[arg-type]
