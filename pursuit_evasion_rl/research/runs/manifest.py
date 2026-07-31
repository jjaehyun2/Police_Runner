"""Content-addressed, immutable run manifest store (Requirement 14.1-14.4).

A manifest begins ``draft`` and stays freely mutable while draft.  Reaching a
terminal :class:`~pursuit_evasion_rl.research.domain.ExecutionStatus` seals it
after validating every required field and every referenced artifact
content hash.  Once sealed, the manifest and its referenced artifacts can
never change in place again: any further change must :meth:`fork
<RunManifestStore.fork>` a new child run with a ``parent_id`` link instead
(Requirement 14.3-14.4), including ``failed``/``not_run``/``null`` results,
which seal exactly like ``completed`` ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from ..canonical import canonical_data, content_hash
from ..domain import ExecutionStatus, ManifestState, RunRecord, exactly_one
from ..errors import ResearchValidationError

RUN_MANIFEST_SCHEMA_VERSION = "1.0"


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be a non-empty string", path=name, actual=value)
    return value


def _freeze_hash_map(value: Mapping[str, str], name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in value.items()
    ):
        _fail("INVALID_HASH_MAP", f"{name} must map non-empty string keys to non-empty string hashes", path=name)
    return MappingProxyType(dict(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_run_id() -> str:
    """A unique, sortable run identifier (Requirement 14.1)."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:12]}"


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """The reproducibility-critical identity of one run (Requirement 14.1, 14.6).

    ``input_hashes``/``output_hashes`` are open string->hash maps (e.g.
    ``{"map": ..., "checkpoint": ...}``); ``output_hashes`` is legitimately
    empty for a draft run and is populated only at seal time.
    """

    code_hash: str
    dirty_tree: bool
    dependency_hash: str
    runtime: str
    device: str
    map_hash: str
    split: str
    seed: int
    input_hashes: Mapping[str, str]
    output_hashes: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("code_hash", "dependency_hash", "runtime", "device", "map_hash", "split"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.dirty_tree, bool):
            _fail("INVALID_PROVENANCE_FIELD", "dirty_tree must be a boolean", path="dirty_tree", actual=self.dirty_tree)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            _fail("INVALID_PROVENANCE_FIELD", "seed must be a nonnegative integer", path="seed", actual=self.seed)
        object.__setattr__(self, "input_hashes", _freeze_hash_map(self.input_hashes, "input_hashes"))
        object.__setattr__(self, "output_hashes", _freeze_hash_map(self.output_hashes, "output_hashes") if self.output_hashes else MappingProxyType({}))

    @property
    def provenance_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """An immutable run identity paired with its reproducibility provenance."""

    run: RunRecord
    provenance: RunProvenance
    created_at_utc: str
    sealed_at_utc: str | None = None
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunRecord):
            _fail("INVALID_RUN_MANIFEST", "run must be a RunRecord", path="run")
        if not isinstance(self.provenance, RunProvenance):
            _fail("INVALID_RUN_MANIFEST", "provenance must be a RunProvenance", path="provenance")
        _required_text(self.created_at_utc, "created_at_utc")
        if self.run.manifest_state is ManifestState.SEALED:
            _required_text(self.sealed_at_utc or "", "sealed_at_utc")
            if self.run.execution_status is None:
                _fail(
                    "SEALED_MANIFEST_WITHOUT_TERMINAL_STATUS",
                    "a sealed manifest requires one terminal execution status",
                    path="run.execution_status",
                )

    @property
    def manifest_hash(self) -> str:
        return content_hash(self)

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def is_sealed(self) -> bool:
        return self.run.manifest_state is ManifestState.SEALED


def _run_to_dict(run: RunRecord) -> dict[str, Any]:
    return canonical_data(run)


def _run_from_dict(payload: Mapping[str, Any]) -> RunRecord:
    execution_status = payload["execution_status"]
    return RunRecord(
        run_id=payload["run_id"],
        protocol_hash=payload["protocol_hash"],
        condition_hash=payload["condition_hash"],
        execution_status=ExecutionStatus(execution_status) if execution_status is not None else None,
        status_reason=payload["status_reason"],
        manifest_state=ManifestState(payload["manifest_state"]),
        parent_id=payload["parent_id"],
        artifact_hashes=dict(payload["artifact_hashes"]),
        result=payload["result"],
        schema_version=payload["schema_version"],
        content_hash=payload["content_hash"],
    )


def _manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "manifest_hash": manifest.manifest_hash,
        "run": _run_to_dict(manifest.run),
        "provenance": canonical_data(manifest.provenance),
        "created_at_utc": manifest.created_at_utc,
        "sealed_at_utc": manifest.sealed_at_utc,
    }


def _manifest_from_dict(payload: Mapping[str, Any]) -> RunManifest:
    provenance_payload = payload["provenance"]
    provenance = RunProvenance(
        code_hash=provenance_payload["code_hash"],
        dirty_tree=provenance_payload["dirty_tree"],
        dependency_hash=provenance_payload["dependency_hash"],
        runtime=provenance_payload["runtime"],
        device=provenance_payload["device"],
        map_hash=provenance_payload["map_hash"],
        split=provenance_payload["split"],
        seed=provenance_payload["seed"],
        input_hashes=dict(provenance_payload["input_hashes"]),
        output_hashes=dict(provenance_payload["output_hashes"]),
        schema_version=provenance_payload["schema_version"],
    )
    return RunManifest(
        run=_run_from_dict(payload["run"]),
        provenance=provenance,
        created_at_utc=payload["created_at_utc"],
        sealed_at_utc=payload["sealed_at_utc"],
        schema_version=payload["schema_version"],
    )


class RunManifestStore:
    """A directory of one-manifest-per-run JSON files, atomically written."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / run_id / "manifest.json"

    def _write_atomic(self, manifest: RunManifest) -> None:
        path = self._path(manifest.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_manifest_to_dict(manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def read(self, run_id: str) -> RunManifest:
        path = self._path(run_id)
        if not path.is_file():
            _fail("RUN_NOT_FOUND", f"no manifest exists for run {run_id!r}", path="run_id", actual=run_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = _manifest_from_dict(payload)
        if manifest.manifest_hash != payload["manifest_hash"]:
            _fail(
                "MANIFEST_HASH_MISMATCH", "stored manifest hash does not match its content",
                path="manifest_hash", expected=manifest.manifest_hash, actual=payload["manifest_hash"],
            )
        return manifest

    def exists(self, run_id: str) -> bool:
        return self._path(run_id).is_file()

    def create(
        self,
        *,
        protocol_hash: str,
        condition_hash: str,
        provenance: RunProvenance,
        run_id: str | None = None,
        parent_id: str | None = None,
    ) -> RunManifest:
        """Create a new ``draft`` manifest (Requirement 14.1)."""
        run_id = run_id or generate_run_id()
        if self.exists(run_id):
            _fail("RUN_ID_COLLISION", f"a manifest already exists for run {run_id!r}", path="run_id", actual=run_id)
        run = RunRecord(
            run_id=run_id,
            protocol_hash=protocol_hash,
            condition_hash=condition_hash,
            execution_status=None,
            status_reason=None,
            manifest_state=ManifestState.DRAFT,
            parent_id=parent_id,
            artifact_hashes={},
            result=None,
        )
        manifest = RunManifest(run=run, provenance=provenance, created_at_utc=_utc_now())
        self._write_atomic(manifest)
        return manifest

    def update_draft(self, run_id: str, *, provenance: RunProvenance | None = None) -> RunManifest:
        """Freely update a still-``draft`` manifest's provenance (e.g. output hashes as they appear)."""
        current = self.read(run_id)
        if current.is_sealed:
            _fail(
                "SEALED_MANIFEST_MUTATION_BLOCKED",
                "a sealed manifest cannot be updated in place; fork a child run instead",
                path="run_id", actual=run_id,
            )
        updated = replace(current, provenance=provenance if provenance is not None else current.provenance)
        self._write_atomic(updated)
        return updated

    def seal(
        self,
        run_id: str,
        *,
        execution_status: ExecutionStatus,
        status_reason: str,
        artifact_hashes: Mapping[str, str] | None = None,
        output_hashes: Mapping[str, str] | None = None,
        result: Any = None,
    ) -> RunManifest:
        """Validate and seal a draft manifest at a terminal status (Requirement 14.2).

        ``execution_status`` may be ``completed``, ``failed``, or ``not_run``
        -- all three seal identically; ``result``/``artifact_hashes`` may be
        empty for ``failed``/``not_run`` but ``status_reason`` is always
        required, and a null/None ``result`` is preserved, not omitted.
        """
        current = self.read(run_id)
        if current.is_sealed:
            _fail(
                "SEALED_MANIFEST_MUTATION_BLOCKED",
                "a sealed manifest cannot be re-sealed in place; fork a child run instead",
                path="run_id", actual=run_id,
            )
        status = exactly_one(execution_status, ExecutionStatus, path="execution_status")
        _required_text(status_reason, "status_reason")
        sealed_run = replace(
            current.run,
            execution_status=status,
            status_reason=status_reason,
            manifest_state=ManifestState.SEALED,
            artifact_hashes=artifact_hashes or {},
            result=result,
            content_hash=None,
        )
        sealed_provenance = (
            replace(current.provenance, output_hashes=output_hashes)
            if output_hashes is not None
            else current.provenance
        )
        sealed = replace(
            current, run=sealed_run, provenance=sealed_provenance, sealed_at_utc=_utc_now(),
        )
        self._write_atomic(sealed)
        return sealed

    def fork(
        self,
        run_id: str,
        *,
        protocol_hash: str | None = None,
        condition_hash: str | None = None,
        provenance: RunProvenance | None = None,
        new_run_id: str | None = None,
    ) -> RunManifest:
        """Fork a sealed run into a new ``draft`` child (Requirement 14.4).

        The child starts fresh (draft, no execution status, no artifacts) but
        keeps its ``parent_id`` lineage; any field the caller does not
        override is copied from the sealed parent unchanged.
        """
        parent = self.read(run_id)
        if not parent.is_sealed:
            _fail(
                "CANNOT_FORK_DRAFT_RUN",
                "only a sealed run needs to fork; a draft run can still be updated in place",
                path="run_id", actual=run_id,
            )
        child_id = new_run_id or generate_run_id()
        if self.exists(child_id):
            _fail("RUN_ID_COLLISION", f"a manifest already exists for run {child_id!r}", path="run_id", actual=child_id)
        child = self.create(
            protocol_hash=protocol_hash if protocol_hash is not None else parent.run.protocol_hash,
            condition_hash=condition_hash if condition_hash is not None else parent.run.condition_hash,
            provenance=provenance if provenance is not None else parent.provenance,
            run_id=child_id,
            parent_id=run_id,
        )
        return child

    def lineage(self, run_id: str) -> tuple[RunManifest, ...]:
        """The chain of manifests from the oldest ancestor to ``run_id`` inclusive."""
        chain: list[RunManifest] = []
        current: str | None = run_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                _fail("RUN_LINEAGE_CYCLE", "run lineage contains a cycle", path="run_id", actual=current)
            seen.add(current)
            manifest = self.read(current)
            chain.append(manifest)
            current = manifest.run.parent_id
        return tuple(reversed(chain))

    def assert_writable(self, run_id: str, path: Path) -> None:
        """Guard for artifact writers: refuse any write under a sealed run's tree."""
        manifest = self.read(run_id)
        run_directory = self._path(run_id).parent
        resolved = path.resolve()
        if manifest.is_sealed and (resolved == run_directory.resolve() or run_directory.resolve() in resolved.parents):
            _fail(
                "SEALED_RUN_ARTIFACT_MUTATION_BLOCKED",
                "sealed run artifacts cannot be modified in place; fork a child run instead",
                path="path", actual=str(resolved),
            )


__all__ = (
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "RunManifestStore",
    "RunProvenance",
    "generate_run_id",
)
