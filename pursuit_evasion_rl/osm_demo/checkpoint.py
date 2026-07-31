"""Safe, CPU-only checkpoint inspection and compatibility reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

import torch

from .canonical import content_hash
from .models import (
    POLICE_COUNT,
    ArchitectureSpec,
    CheckpointManifest,
    CheckStatus,
    CompatibilityCheck,
    CompatibilityReport,
    InferredCheckpointContract,
    ObservationContract,
)

EXPECTED_ARCHITECTURE = ArchitectureSpec(21, (128, 128), 6)
EXPECTED_FIXED_MAX_DEGREE = 5
EXPECTED_LAYER_SHAPES = (
    (128, 21),
    (128,),
    (128, 128),
    (128,),
    (6, 128),
    (6,),
)
OSM_OBSERVATION_PROFILE = "osm_topology_v1"
LEGACY_OBSERVATION_PROFILE = "legacy_grid_v0"
UNKNOWN_SEMANTICS = frozenset(
    {"police_count", "observation_semantics", "training_region", "performance"}
)


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """Tensor-derived facts and their sectioned compatibility report."""

    checkpoint_hash: str
    inferred_contract: InferredCheckpointContract
    report: CompatibilityReport

    def as_dict(self) -> dict[str, Any]:
        inferred = {
            "actor_key": self.inferred_contract.actor_key,
            "layer_shapes": [list(shape) for shape in self.inferred_contract.layer_shapes],
            "observation_dim": self.inferred_contract.observation_dim,
            "hidden_dims": list(self.inferred_contract.hidden_dims),
            "action_dim": self.inferred_contract.action_dim,
            "inferable": sorted(self.inferred_contract.inferable),
            "unknown": sorted(self.inferred_contract.unknown),
        }
        return {"checkpoint": inferred, "compatibility": self.report.as_dict()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_cpu_load(path: Path) -> Any:
    """Load only tensor/primitive allowlisted values and never unsafe pickle classes."""
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Installed PyTorch does not support mandatory weights-only checkpoint loading"
        ) from exc


def _mapping_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    return value


def _actor_state(payload: Any) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint root must be a mapping")

    nested = payload.get("police_actor")
    if isinstance(nested, Mapping):
        return "police_actor", nested

    # Repository checkpoints use both a top-level actor state and a
    # ``model_state.police_actor`` container.  Inspect only mappings and
    # tensors; never reconstruct a saved model or import checkpoint classes.
    for container_name in ("model_state", "state_dict", "model_state_dict"):
        candidate = payload.get(container_name)
        if isinstance(candidate, Mapping):
            nested_actor = candidate.get("police_actor")
            if isinstance(nested_actor, Mapping):
                return f"{container_name}.police_actor", nested_actor
            prefixed = _prefixed_actor_state(candidate)
            if prefixed:
                return f"{container_name}.police_actor", prefixed
            if candidate and all(torch.is_tensor(value) for value in candidate.values()):
                return container_name, candidate

    prefixed = _prefixed_actor_state(payload)
    if prefixed:
        return "police_actor", prefixed
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return "state_dict", payload
    raise ValueError("Checkpoint does not contain a police_actor tensor state dictionary")


def _prefixed_actor_state(state: Mapping[str, Any]) -> dict[str, Any]:
    actor: dict[str, Any] = {}
    marker = "police_actor."
    for key, value in state.items():
        if isinstance(key, str) and marker in key:
            actor[key.split(marker, 1)[1]] = value
    return actor


def _ordered_parameter_shapes(actor: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    shapes: list[tuple[int, ...]] = []
    for key, value in actor.items():
        if not isinstance(key, str) or key.rsplit(".", 1)[-1] not in {"weight", "bias"}:
            continue
        if not torch.is_tensor(value):
            raise ValueError(f"police_actor parameter {key!r} is not a tensor")
        shapes.append(tuple(int(size) for size in value.shape))
    if not shapes:
        raise ValueError("police_actor has no ordered weight or bias tensors")
    return tuple(shapes)


def _infer_dimensions(
    shapes: tuple[tuple[int, ...], ...]
) -> tuple[int | None, tuple[int, ...], int | None]:
    weights = [shape for shape in shapes if len(shape) == 2]
    if not weights or any(weights[index][0] != weights[index + 1][1] for index in range(len(weights) - 1)):
        return None, (), None
    observation_dim = weights[0][1]
    action_dim = weights[-1][0]
    hidden_dims = tuple(shape[0] for shape in weights[:-1])
    return observation_dim, hidden_dims, action_dim


def _empty_contract() -> InferredCheckpointContract:
    return InferredCheckpointContract(
        actor_key="",
        layer_shapes=(),
        observation_dim=None,
        hidden_dims=(),
        action_dim=None,
        inferable=frozenset(),
        unknown=UNKNOWN_SEMANTICS | {"architecture"},
    )


def _check(
    code: str,
    status: CheckStatus,
    expected: Any,
    actual: Any,
    message: str,
) -> CompatibilityCheck:
    return CompatibilityCheck(code, status, _plain(expected), _plain(actual), message)


def _comparison(
    code: str,
    expected: Any,
    actual: Any,
    *,
    message: str,
    unknown_message: str | None = None,
) -> CompatibilityCheck:
    if actual is None:
        return _check(
            code,
            CheckStatus.UNKNOWN,
            expected,
            None,
            unknown_message or f"{message}; actual value is not inferable",
        )
    status = CheckStatus.PASS if _plain(actual) == _plain(expected) else CheckStatus.FAIL
    return _check(code, status, expected, actual, message)


def _manifest_architecture(manifest: Any) -> Any:
    return _mapping_value(manifest, "architecture") if manifest is not None else None


def _manifest_layer_shapes(manifest: Any) -> tuple[bool, Any]:
    """Return whether shapes were explicitly declared and their raw value."""
    if manifest is None:
        return False, None
    for owner in (manifest, _manifest_architecture(manifest)):
        for name in ("police_actor_shapes", "actor_layer_shapes", "layer_shapes"):
            value = _mapping_value(owner, name)
            if value is not None:
                return True, value
    return False, None


def _shape_value(value: Any) -> tuple[tuple[int, ...], ...] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        shapes = tuple(tuple(int(size) for size in shape) for shape in value)
    except (TypeError, ValueError):
        return None
    if not shapes or any(not shape or any(size <= 0 for size in shape) for shape in shapes):
        return None
    return shapes


def _nonempty_string_sequence(value: Any) -> tuple[str, ...] | None:
    """Normalize manifest provenance arrays without accepting strings as arrays."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    normalized = tuple(str(item) for item in value)
    if not normalized or any(not item for item in normalized):
        return None
    return normalized


def _overall(sections: Sequence[Sequence[CompatibilityCheck]]) -> CheckStatus:
    statuses = {check.status for section in sections for check in section}
    for status in (CheckStatus.FAIL, CheckStatus.WARNING, CheckStatus.UNKNOWN):
        if status in statuses:
            return status
    return CheckStatus.PASS


def _report_id(
    checkpoint_hash: str,
    network_hash: str,
    sections: Sequence[Sequence[CompatibilityCheck]],
) -> str:
    return content_hash(
        {
            "checkpoint_hash": checkpoint_hash,
            "network_hash": network_hash,
            "sections": [[check.as_dict() for check in section] for section in sections],
        }
    )


def _architecture_shapes(architecture: Any) -> tuple[tuple[int, ...], ...] | None:
    if architecture is None:
        return None
    observation_dim = _mapping_value(architecture, "observation_dim")
    hidden_dims = _mapping_value(architecture, "hidden_dims")
    action_dim = _mapping_value(architecture, "action_dim")
    if observation_dim is None or hidden_dims is None or action_dim is None:
        return None
    try:
        dimensions = (int(observation_dim), *(int(item) for item in hidden_dims), int(action_dim))
    except (TypeError, ValueError):
        return None
    result: list[tuple[int, ...]] = []
    for input_dim, output_dim in zip(dimensions, dimensions[1:]):
        result.extend(((output_dim, input_dim), (output_dim,)))
    return tuple(result)


def _structure_checks(
    inferred: InferredCheckpointContract,
    checkpoint_hash: str,
    manifest: Any,
) -> list[CompatibilityCheck]:
    checks = [
        _comparison(
            "CHECKPOINT_OBSERVATION_DIM",
            EXPECTED_ARCHITECTURE.observation_dim,
            inferred.observation_dim,
            message="Checkpoint actor observation dimension must match the current contract",
        ),
        _comparison(
            "CHECKPOINT_HIDDEN_DIMS",
            EXPECTED_ARCHITECTURE.hidden_dims,
            inferred.hidden_dims or None,
            message="Checkpoint actor hidden dimensions must match the current contract",
        ),
        _comparison(
            "CHECKPOINT_ACTION_DIM",
            EXPECTED_ARCHITECTURE.action_dim,
            inferred.action_dim,
            message="Checkpoint actor action dimension must match the current contract",
        ),
        _comparison(
            "CHECKPOINT_LAYER_SHAPES",
            EXPECTED_LAYER_SHAPES,
            inferred.layer_shapes or None,
            message="Ordered police_actor weight and bias shapes must match 21→128→128→6",
        ),
    ]
    manifest_hash = _mapping_value(manifest, "checkpoint_hash") if manifest is not None else None
    checks.append(
        _comparison(
            "MANIFEST_CHECKPOINT_HASH",
            manifest_hash if manifest_hash is not None else "checkpoint SHA-256",
            checkpoint_hash if manifest_hash is not None else None,
            message="Checkpoint bytes must match the manifest hash",
            unknown_message="No manifest hash is available for comparison",
        )
    )

    architecture = _manifest_architecture(manifest)
    manifest_shapes = _shape_value(_manifest_layer_shapes(manifest))
    if manifest_shapes is None:
        manifest_shapes = _architecture_shapes(architecture)
    checks.append(
        _comparison(
            "MANIFEST_ACTOR_LAYER_SHAPES",
            manifest_shapes if manifest_shapes is not None else "ordered actor layer shapes",
            inferred.layer_shapes if manifest_shapes is not None else None,
            message="Checkpoint police_actor tensor shapes must match the manifest",
            unknown_message="Manifest does not declare enough architecture detail for layer-shape comparison",
        )
    )

    police_count = _mapping_value(manifest, "police_count") if manifest is not None else None
    fixed_degree = _mapping_value(manifest, "fixed_max_degree") if manifest is not None else None
    if fixed_degree is None:
        fixed_degree = _mapping_value(architecture, "fixed_max_degree")
    checks.extend(
        [
            _comparison(
                "POLICE_COUNT",
                POLICE_COUNT,
                police_count,
                message="Police count must match the current checkpoint contract",
                unknown_message="Police count cannot be inferred from actor tensors",
            ),
            _comparison(
                "FIXED_MAX_DEGREE",
                EXPECTED_FIXED_MAX_DEGREE,
                fixed_degree,
                message="Fixed maximum degree must match the current checkpoint contract",
                unknown_message="Fixed maximum degree cannot be inferred from actor tensors",
            ),
        ]
    )
    return checks


def _semantic_value(contract: Any, name: str) -> Any:
    value = _mapping_value(contract, name)
    if name == "field_sizes" and not value:
        return None
    return value


def _semantics_checks(
    manifest: Any,
    selected: ObservationContract | Mapping[str, Any] | None,
) -> list[CompatibilityCheck]:
    manifest_contract = _mapping_value(manifest, "observation_contract") if manifest is not None else None
    manifest_profile = _semantic_value(manifest_contract, "profile_id")
    selected_profile = _semantic_value(selected, "profile_id")

    if manifest_profile == LEGACY_OBSERVATION_PROFILE:
        profile_check = _check(
            "OBSERVATION_PROFILE",
            CheckStatus.WARNING,
            selected_profile or OSM_OBSERVATION_PROFILE,
            manifest_profile,
            "Legacy observation semantics are not compatible with the OSM observation profile",
        )
    elif manifest_profile is None:
        profile_check = _check(
            "OBSERVATION_PROFILE",
            CheckStatus.UNKNOWN,
            selected_profile or OSM_OBSERVATION_PROFILE,
            None,
            "Checkpoint observation profile is not proven by tensor shapes",
        )
    elif selected_profile is None:
        profile_check = _check(
            "OBSERVATION_PROFILE",
            CheckStatus.UNKNOWN,
            "selected runtime observation profile",
            manifest_profile,
            "A manifest profile is declared, but no runtime profile was selected for comparison",
        )
    else:
        profile_check = _comparison(
            "OBSERVATION_PROFILE",
            selected_profile,
            manifest_profile,
            message="Checkpoint and selected observation profile identifiers must match",
        )

    checks = [profile_check]
    semantic_fields = (
        ("fields", "OBSERVATION_FIELDS", "Observation field meanings and order must match"),
        ("field_sizes", "OBSERVATION_FIELD_SIZES", "Observation field sizes must match"),
        ("dtype", "OBSERVATION_DTYPE", "Observation dtype must match"),
        ("padding_value", "OBSERVATION_PADDING", "Observation padding value must match"),
        ("normalization", "OBSERVATION_NORMALIZATION", "Observation normalization rules must match"),
        ("action_ordering", "ACTION_SLOT_ORDERING", "Action slot ordering rules must match"),
    )
    for name, code, message in semantic_fields:
        actual = _semantic_value(manifest_contract, name)
        expected = _semantic_value(selected, name)
        if selected is None:
            checks.append(
                _check(
                    code,
                    CheckStatus.UNKNOWN,
                    "selected ObservationContract",
                    actual,
                    f"{message}; no selected ObservationContract was provided",
                )
            )
        elif actual is None:
            checks.append(
                _check(
                    code,
                    CheckStatus.UNKNOWN,
                    expected,
                    None,
                    f"{message}; checkpoint manifest does not prove this value",
                )
            )
        elif expected is None:
            checks.append(
                _check(
                    code,
                    CheckStatus.UNKNOWN,
                    "complete selected ObservationContract value",
                    actual,
                    f"{message}; selected ObservationContract omits this value",
                )
            )
        else:
            checks.append(_comparison(code, expected, actual, message=message))

    training_networks = _mapping_value(manifest, "training_networks") if manifest is not None else None
    checks.append(
        _check(
            "TRAINING_REGION_PROVENANCE",
            CheckStatus.PASS if training_networks else CheckStatus.UNKNOWN,
            "manifest-declared training networks",
            training_networks or None,
            "Training regions are manifest provenance and are never inferred from tensors",
        )
    )
    performance = _mapping_value(manifest, "performance_evidence") if manifest is not None else None
    claim_status = _mapping_value(manifest, "claim_status") if manifest is not None else None
    checks.append(
        _check(
            "PERFORMANCE_EVIDENCE",
            CheckStatus.PASS if performance else CheckStatus.UNKNOWN,
            "explicit reproducible performance evidence",
            performance or ({"claim_status": _plain(claim_status)} if claim_status is not None else None),
            "A claim status alone is not performance evidence and performance is never inferred from weights",
        )
    )
    return checks


def _network_checks(manifest: Any, network_hash: str) -> list[CompatibilityCheck]:
    training_networks = _mapping_value(manifest, "training_networks") if manifest is not None else None
    if not network_hash:
        return [
            _check(
                "NETWORK_COMPATIBILITY_NOT_EVALUATED",
                CheckStatus.UNKNOWN,
                "validated Model_Network hash",
                None,
                "No network hash was supplied to checkpoint inspection",
            )
        ]
    if not training_networks:
        return [
            _check(
                "NETWORK_TRAINING_PROVENANCE",
                CheckStatus.UNKNOWN,
                network_hash,
                None,
                "The manifest does not declare training networks; generalization is not inferred",
            )
        ]
    trained_here = network_hash in training_networks
    return [
        _check(
            "NETWORK_TRAINING_PROVENANCE",
            CheckStatus.PASS if trained_here else CheckStatus.WARNING,
            network_hash,
            training_networks,
            "Network appears in training provenance" if trained_here else "Network is unseen in declared training provenance",
        )
    ]


def _execution_checks(execution: Mapping[str, Any] | Any | None) -> list[CompatibilityCheck]:
    expected_values = (
        ("police_count", "EXECUTION_POLICE_COUNT", POLICE_COUNT),
        ("fixed_max_degree", "EXECUTION_FIXED_MAX_DEGREE", EXPECTED_FIXED_MAX_DEGREE),
        ("observation_dim", "EXECUTION_OBSERVATION_DIM", EXPECTED_ARCHITECTURE.observation_dim),
        ("action_dim", "EXECUTION_ACTION_DIM", EXPECTED_ARCHITECTURE.action_dim),
        ("hidden_dims", "EXECUTION_HIDDEN_DIMS", EXPECTED_ARCHITECTURE.hidden_dims),
    )
    if execution is None:
        return [
            _check(
                "EXECUTION_CONFIG_NOT_PROVIDED",
                CheckStatus.UNKNOWN,
                {name: _plain(expected) for name, _, expected in expected_values},
                None,
                "Runtime contract was not supplied; execution compatibility is not assumed",
            )
        ]

    checks: list[CompatibilityCheck] = []
    for name, code, expected in expected_values:
        actual = _mapping_value(execution, name)
        if name == "hidden_dims" and actual is not None:
            try:
                actual = tuple(actual)
            except TypeError:
                pass
        checks.append(
            _comparison(
                code,
                expected,
                actual,
                message=f"Runtime {name} must match the current checkpoint contract",
                unknown_message=f"Runtime {name} was not supplied",
            )
        )
    return checks


def _make_report(
    checkpoint_hash: str,
    network_hash: str,
    structure: Sequence[CompatibilityCheck],
    semantics: Sequence[CompatibilityCheck],
    network: Sequence[CompatibilityCheck],
    execution: Sequence[CompatibilityCheck],
) -> CompatibilityReport:
    sections = (tuple(structure), tuple(semantics), tuple(network), tuple(execution))
    return CompatibilityReport(
        report_id=_report_id(checkpoint_hash, network_hash, sections),
        checkpoint_hash=checkpoint_hash,
        network_hash=network_hash,
        structure=sections[0],
        semantics=sections[1],
        network=sections[2],
        execution=sections[3],
        overall=_overall(sections),
    )


def _failed_inspection(
    *,
    checkpoint_hash: str,
    code: str,
    actual: Any,
    selected_observation_contract: ObservationContract | Mapping[str, Any] | None,
    execution_config: Mapping[str, Any] | Any | None,
    network_hash: str,
) -> CheckpointInspection:
    inferred = _empty_contract()
    structure = [
        _check(
            code,
            CheckStatus.FAIL,
            "safe CPU weights-only checkpoint containing police_actor tensors",
            actual,
            "Checkpoint could not be safely inspected",
        )
    ]
    report = _make_report(
        checkpoint_hash,
        network_hash,
        structure,
        _semantics_checks(None, selected_observation_contract),
        _network_checks(None, network_hash),
        _execution_checks(execution_config),
    )
    return CheckpointInspection(checkpoint_hash, inferred, report)


class CheckpointInspector:
    """Inspect checkpoint tensors without constructing or importing saved classes."""

    def inspect(
        self,
        path: str | Path,
        optional_manifest: CheckpointManifest | Mapping[str, Any] | None = None,
        *,
        selected_observation_contract: ObservationContract | Mapping[str, Any] | None = None,
        execution_config: Mapping[str, Any] | Any | None = None,
        network_hash: str = "",
    ) -> CheckpointInspection:
        checkpoint_path = Path(path)
        try:
            checkpoint_hash = _file_sha256(checkpoint_path)
        except OSError as exc:
            return _failed_inspection(
                checkpoint_hash="",
                code="CHECKPOINT_READ_FAILED",
                actual=type(exc).__name__,
                selected_observation_contract=selected_observation_contract,
                execution_config=execution_config,
                network_hash=network_hash,
            )

        try:
            payload = _safe_cpu_load(checkpoint_path)
            actor_key, actor = _actor_state(payload)
            shapes = _ordered_parameter_shapes(actor)
            observation_dim, hidden_dims, action_dim = _infer_dimensions(shapes)
        except Exception as exc:  # safe loader and format failures become report data
            return _failed_inspection(
                checkpoint_hash=checkpoint_hash,
                code="CHECKPOINT_SAFE_INSPECTION_FAILED",
                actual=type(exc).__name__,
                selected_observation_contract=selected_observation_contract,
                execution_config=execution_config,
                network_hash=network_hash,
            )

        inferred = InferredCheckpointContract(
            actor_key=actor_key,
            layer_shapes=shapes,
            observation_dim=observation_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            inferable=frozenset(
                {"layer_shapes", "observation_dim", "hidden_dims", "action_dim"}
            ),
            unknown=UNKNOWN_SEMANTICS,
        )
        report = _make_report(
            checkpoint_hash,
            network_hash,
            _structure_checks(inferred, checkpoint_hash, optional_manifest),
            _semantics_checks(optional_manifest, selected_observation_contract),
            _network_checks(optional_manifest, network_hash),
            _execution_checks(execution_config),
        )
        return CheckpointInspection(checkpoint_hash, inferred, report)


__all__ = (
    "EXPECTED_ARCHITECTURE",
    "EXPECTED_FIXED_MAX_DEGREE",
    "EXPECTED_LAYER_SHAPES",
    "CheckpointInspection",
    "CheckpointInspector",
)
