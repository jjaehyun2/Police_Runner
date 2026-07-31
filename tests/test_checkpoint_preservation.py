"""Task 1.3 unit tests for immutable best_v2 preservation."""

from pathlib import Path
import stat

import pytest

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.preservation import (
    DESIGN_AUDIT_OBSERVATION_SHA256,
    ProtectedPathGuard,
    RunPaths,
    checkpoint_run_directory,
    is_readonly,
    preserve_artifact,
    validate_checkpoint_output,
    verify_before_run,
)


def _preserved(tmp_path: Path):
    project = tmp_path / "project"
    source = project / "checkpoints" / "osm_mappo" / "best_v2.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"legacy-best-v2\x00weights")
    copy = project / "artifacts" / "research" / "preserved" / "best_v2.pt"
    return project, source, copy, preserve_artifact(source, copy)


def _run_paths(project: Path) -> RunPaths:
    checkpoint = checkpoint_run_directory(project, "full", 17, "run-001") / "model.pt"
    return RunPaths(
        project_root=project,
        condition="full",
        seed=17,
        run_id="run-001",
        input_paths=(project / "inputs" / "map.json",),
        output_paths=(project / "artifacts" / "research" / "runs" / "run-001",),
        temporary_paths=(project / ".tmp" / "run-001",),
        checkpoint_outputs=(checkpoint,),
    )


def _change_one_byte(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0x01
    path.write_bytes(payload)


def test_preservation_records_runtime_identity_and_seals_byte_identical_copy(tmp_path):
    project, source, copy, artifact = _preserved(tmp_path)

    assert project.exists()
    assert copy.read_bytes() == source.read_bytes()
    assert artifact.original.sha256 == artifact.preserved_copy.sha256
    assert artifact.original.byte_size == len(source.read_bytes())
    assert artifact.original.real_path != artifact.preserved_copy.real_path
    assert artifact.prior_audit_sha256 == DESIGN_AUDIT_OBSERVATION_SHA256
    assert artifact.prior_audit_matches_current is False
    assert is_readonly(copy)
    assert artifact.verify_hash()

    report = verify_before_run(artifact, _run_paths(project))
    assert report.original_sha256 == artifact.original.sha256
    assert report.preserved_sha256 == artifact.preserved_copy.sha256
    assert report.checked_path_count == 4
    assert report.verify_hash()


def test_prior_design_hash_is_an_audit_observation_not_a_success_constant(tmp_path):
    project = tmp_path / "project"
    source = project / "checkpoints" / "osm_mappo" / "best_v2.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"a different but valid checkpoint")

    artifact = preserve_artifact(
        source,
        project / "artifacts" / "research" / "preserved" / "best_v2.pt",
    )

    assert artifact.original.sha256 != DESIGN_AUDIT_OBSERVATION_SHA256
    assert artifact.prior_audit_matches_current is False
    assert verify_before_run(artifact, _run_paths(project)).original_sha256 == artifact.original.sha256


def test_existing_preservation_copy_is_never_overwritten(tmp_path):
    project = tmp_path / "project"
    source = project / "checkpoints" / "osm_mappo" / "best_v2.pt"
    destination = project / "artifacts" / "research" / "preserved" / "best_v2.pt"
    source.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    destination.write_bytes(b"existing sealed evidence")

    with pytest.raises(ResearchValidationError) as raised:
        preserve_artifact(source, destination)

    assert raised.value.code == "PRESERVATION_DESTINATION_EXISTS"
    assert destination.read_bytes() == b"existing sealed evidence"


def test_pre_run_gate_rejects_one_byte_original_tamper(tmp_path):
    project, source, _, artifact = _preserved(tmp_path)
    _change_one_byte(source)

    with pytest.raises(ResearchValidationError) as raised:
        verify_before_run(artifact, _run_paths(project))

    assert raised.value.code == "ORIGINAL_INTEGRITY_MISMATCH"
    assert len(raised.value.record.content_hash) == 64


def test_pre_run_gate_rejects_one_byte_preserved_copy_tamper(tmp_path):
    project, _, copy, artifact = _preserved(tmp_path)
    copy.chmod(copy.stat().st_mode | stat.S_IWUSR)
    _change_one_byte(copy)

    with pytest.raises(ResearchValidationError) as raised:
        verify_before_run(artifact, _run_paths(project))

    assert raised.value.code == "PRESERVED_INTEGRITY_MISMATCH"
    assert len(raised.value.record.content_hash) == 64


def test_pre_run_gate_rejects_lost_readonly_seal(tmp_path):
    project, _, copy, artifact = _preserved(tmp_path)
    copy.chmod(copy.stat().st_mode | stat.S_IWUSR)

    with pytest.raises(ResearchValidationError) as raised:
        verify_before_run(artifact, _run_paths(project))

    assert raised.value.code == "PRESERVED_COPY_NOT_READONLY"


def test_checkpoint_outputs_are_restricted_to_condition_seed_run_directory(tmp_path):
    project, source, _, artifact = _preserved(tmp_path)
    allowed = checkpoint_run_directory(project, "full", 17, "run-001") / "latest.pt"
    assert validate_checkpoint_output(
        allowed, project_root=project, condition="full", seed=17, run_id="run-001"
    ) == allowed.resolve()

    outside = project / "checkpoints" / "new.pt"
    with pytest.raises(ResearchValidationError) as raised:
        validate_checkpoint_output(
            outside, project_root=project, condition="full", seed=17, run_id="run-001"
        )
    assert raised.value.code == "CHECKPOINT_OUTPUT_OUTSIDE_RUN_DIRECTORY"

    protected_paths = _run_paths(project)
    protected_paths = RunPaths(
        project_root=project,
        condition="full",
        seed=17,
        run_id="run-001",
        checkpoint_outputs=(source,),
    )
    with pytest.raises(ResearchValidationError) as raised:
        verify_before_run(artifact, protected_paths)
    assert raised.value.code == "RUN_PATH_OVERLAPS_PROTECTED_ARTIFACT"


def test_guard_blocks_write_move_delete_for_original_copy_and_containing_paths(tmp_path):
    project, source, copy, artifact = _preserved(tmp_path)
    guard = ProtectedPathGuard(artifact)

    blocked = (
        (guard.assert_write_allowed, (source,)),
        (guard.assert_write_allowed, (copy,)),
        (guard.assert_move_allowed, (project / "elsewhere.pt", source)),
        (guard.assert_move_allowed, (copy, project / "elsewhere.pt")),
        (guard.assert_delete_allowed, (source.parent,)),
    )
    for operation, arguments in blocked:
        with pytest.raises(ResearchValidationError) as raised:
            operation(*arguments)
        assert raised.value.code == "PROTECTED_ARTIFACT_MUTATION_BLOCKED"

    guard.assert_write_allowed(project / "artifacts" / "research" / "runs" / "run-001.json")
    guard.assert_move_allowed(project / "scratch" / "a", project / "scratch" / "b")
    guard.assert_delete_allowed(project / "scratch" / "old")


def test_symlink_alias_to_protected_path_is_rejected(tmp_path):
    project, source, _, artifact = _preserved(tmp_path)
    alias = project / "alias-to-checkpoints"
    try:
        alias.symlink_to(source.parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    paths = RunPaths(
        project_root=project,
        condition="full",
        seed=17,
        run_id="run-001",
        output_paths=(alias / source.name,),
        checkpoint_outputs=(checkpoint_run_directory(project, "full", 17, "run-001") / "model.pt",),
    )
    with pytest.raises(ResearchValidationError) as raised:
        verify_before_run(artifact, paths)

    assert raised.value.code == "RUN_PATH_OVERLAPS_PROTECTED_ARTIFACT"
    assert raised.value.record.details["category"] == "output"
