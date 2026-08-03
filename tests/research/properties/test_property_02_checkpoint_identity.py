"""Property 2 coverage for immutable checkpoint identity gates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import stat
from string import ascii_lowercase, digits
from tempfile import TemporaryDirectory
from typing import Iterator

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.errors import ErrorRecord, ResearchValidationError
from pursuit_evasion_rl.research.preservation import (
    PreservedArtifact,
    ProtectedPathGuard,
    RunPaths,
    checkpoint_run_directory,
    is_readonly,
    preserve_artifact,
    verify_before_run,
)

# **Property 2: Preserved checkpoint identity gates every run**
# **Validates: Requirements 2.3–2.6**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)
_SAFE_ALPHABET = ascii_lowercase + digits + "_-"
artifact_bytes = st.binary(min_size=1, max_size=512)
path_components = st.lists(
    st.text(_SAFE_ALPHABET, min_size=1, max_size=12),
    min_size=1,
    max_size=3,
).map(tuple)
checkpoint_run_ids = st.tuples(
    st.sampled_from(ascii_lowercase + digits),
    st.text(ascii_lowercase + digits + "_-", min_size=0, max_size=11),
).map(lambda parts: "".join(parts))
_READ_ONLY_USAGES = ("baseline", "initialization", "read")
read_only_usages = st.sampled_from(_READ_ONLY_USAGES)
mutation_usages = st.sampled_from(("write", "move", "delete"))


@dataclass(frozen=True)
class CheckpointCase:
    payload: bytes
    path_parts: tuple[str, ...]
    usage: str
    condition: str
    seed: int
    run_id: str


@st.composite
def checkpoint_cases(draw: st.DrawFn) -> CheckpointCase:
    """Generate artifact bytes, separate path identities, and read-only usages."""
    return CheckpointCase(
        payload=draw(artifact_bytes),
        path_parts=draw(path_components),
        usage=draw(read_only_usages),
        condition=draw(st.text(ascii_lowercase + digits, min_size=1, max_size=10)),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        run_id=draw(checkpoint_run_ids),
    )


@dataclass(frozen=True)
class BuiltCase:
    root: Path
    original: Path
    preserved: Path
    artifact: PreservedArtifact
    paths: RunPaths


@contextmanager
def _built_case(case: CheckpointCase) -> Iterator[BuiltCase]:
    with TemporaryDirectory() as directory:
        root = Path(directory) / "project"
        original = root.joinpath("legacy", *case.path_parts, "best_v2.pt")
        preserved = root.joinpath("preserved", *case.path_parts, "best_v2.pt")
        original.parent.mkdir(parents=True)
        original.write_bytes(case.payload)
        artifact = preserve_artifact(original, preserved)
        checkpoint = checkpoint_run_directory(
            root, case.condition, case.seed, case.run_id
        ) / "model.pt"
        paths = RunPaths(
            project_root=root,
            condition=case.condition,
            seed=case.seed,
            run_id=case.run_id,
            input_paths=(root.joinpath("inputs", *case.path_parts, "map.json"),),
            output_paths=(root.joinpath("runs", *case.path_parts, "result.json"),),
            temporary_paths=(root.joinpath("tmp", *case.path_parts),),
            checkpoint_outputs=(checkpoint,),
        )
        try:
            yield BuiltCase(root, original, preserved, artifact, paths)
        finally:
            if preserved.exists():
                preserved.chmod(preserved.stat().st_mode | stat.S_IWUSR)


def _mutation_paths(usage: str, target: Path, safe: Path) -> tuple[Path, ...]:
    return (target, safe) if usage == "move" else (target,)


def _assert_preserved_error_hash(error: ResearchValidationError) -> str:
    serialized = error.as_dict()
    digest = error.record.content_hash
    assert digest is not None and len(digest) == 64
    assert serialized["content_hash"] == digest
    assert ErrorRecord(**serialized).content_hash == digest
    return digest


def _perturb_bytes(path: Path, *, change_size: bool) -> None:
    payload = bytearray(path.read_bytes())
    if change_size:
        payload.append(payload[0] ^ 0xFF)
    else:
        payload[0] ^= 0x01
    path.write_bytes(payload)


@_PBT_SETTINGS
@given(case=checkpoint_cases())
def test_matching_identity_readonly_seal_and_separate_paths_allow_run(
    case: CheckpointCase,
) -> None:
    with _built_case(case) as built:
        expected_hash = hashlib.sha256(case.payload).hexdigest()
        assert built.artifact.original.sha256 == expected_hash
        assert built.artifact.preserved_copy.sha256 == expected_hash
        assert built.artifact.original.byte_size == len(case.payload)
        assert built.artifact.preserved_copy.byte_size == len(case.payload)
        assert is_readonly(built.preserved)
        assert built.original.resolve() != built.preserved.resolve()

        report = verify_before_run(built.artifact, built.paths)
        assert report.original_sha256 == expected_hash
        assert report.preserved_sha256 == expected_hash
        assert report.byte_size == len(case.payload)
        assert report.checked_path_count == 4

        # A successful case represents non-mutating baseline/initialization/read use.
        assert case.usage in _READ_ONLY_USAGES
        assert built.preserved.read_bytes() == case.payload
        assert is_readonly(built.preserved)
        assert hashlib.sha256(built.preserved.read_bytes()).hexdigest() == expected_hash


_GATE_DEFECTS = st.sampled_from(
    (
        "original_hash",
        "original_size",
        "preserved_hash",
        "preserved_size",
        "preserved_readonly",
        "output_to_original",
        "output_to_preserved",
    )
)


@_PBT_SETTINGS
@given(case=checkpoint_cases(), defect=_GATE_DEFECTS)
def test_one_original_preserved_or_output_perturbation_closes_gate_and_keeps_error_hash(
    case: CheckpointCase, defect: str
) -> None:
    with _built_case(case) as built:
        paths = built.paths
        expected_code: str
        if defect.startswith("original_"):
            _perturb_bytes(built.original, change_size=defect.endswith("size"))
            expected_code = "ORIGINAL_INTEGRITY_MISMATCH"
        elif defect in {"preserved_hash", "preserved_size"}:
            built.preserved.chmod(built.preserved.stat().st_mode | stat.S_IWUSR)
            _perturb_bytes(built.preserved, change_size=defect.endswith("size"))
            built.preserved.chmod(built.preserved.stat().st_mode & ~stat.S_IWUSR)
            expected_code = "PRESERVED_INTEGRITY_MISMATCH"
        elif defect == "preserved_readonly":
            built.preserved.chmod(built.preserved.stat().st_mode | stat.S_IWUSR)
            expected_code = "PRESERVED_COPY_NOT_READONLY"
        else:
            protected = (
                built.original if defect == "output_to_original" else built.preserved
            )
            paths = replace(paths, output_paths=(protected,))
            expected_code = "RUN_PATH_OVERLAPS_PROTECTED_ARTIFACT"

        with pytest.raises(ResearchValidationError) as first:
            verify_before_run(built.artifact, paths)
        assert first.value.code == expected_code
        first_hash = _assert_preserved_error_hash(first.value)

        with pytest.raises(ResearchValidationError) as repeated:
            verify_before_run(built.artifact, paths)
        assert repeated.value.code == expected_code
        assert _assert_preserved_error_hash(repeated.value) == first_hash


@_PBT_SETTINGS
@given(
    case=checkpoint_cases(),
    mutation_usage=mutation_usages,
    target=st.sampled_from(("original", "preserved")),
)
def test_generated_write_move_delete_usage_is_blocked_for_each_protected_identity(
    case: CheckpointCase, mutation_usage: str, target: str
) -> None:
    with _built_case(case) as built:
        protected = built.original if target == "original" else built.preserved
        safe = built.root / "scratch" / "destination.bin"
        guard = ProtectedPathGuard(built.artifact)
        arguments = _mutation_paths(mutation_usage, protected, safe)

        with pytest.raises(ResearchValidationError) as first:
            guard.assert_allowed(mutation_usage, *arguments)
        assert first.value.code == "PROTECTED_ARTIFACT_MUTATION_BLOCKED"
        first_hash = _assert_preserved_error_hash(first.value)

        with pytest.raises(ResearchValidationError) as repeated:
            guard.assert_allowed(mutation_usage, *arguments)
        assert repeated.value.code == "PROTECTED_ARTIFACT_MUTATION_BLOCKED"
        assert _assert_preserved_error_hash(repeated.value) == first_hash
