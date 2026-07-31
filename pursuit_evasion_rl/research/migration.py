"""Fail-closed migration contracts for legacy root research assets.

This module converts legacy configuration and checkpoint *metadata* into
canonical, content-addressed records.  It deliberately does not implement a
research trainer: root scripts keep using their legacy bodies until a real
package CLI entry point exists.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import warnings

from .canonical import canonical_data, content_hash, sha256_bytes
from .errors import ResearchValidationError

MIGRATION_SCHEMA_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
LEGACY_ONLY = "legacy_compatibility_only"

TRAIN_CLI_DEFAULTS = MappingProxyType({
    "episodes": 6000, "max_steps": 450, "police_speed": 16.0,
    "fugitive_speed": 9.0, "capture": 25.0, "vision": 150.0,
    "clip": 2000.0, "lr": 3e-4, "gamma": 0.99,
    "capture_bonus": 20.0, "time_penalty": 0.02, "dist_scale": 50.0,
    "team_coef": 1.0, "own_coef": 0.6, "regress_mult": 1.6,
    "entropy_coef": 0.01, "ppo_epochs": 4, "update_every": 8,
    "log_every": 50, "ckpt_every": 500, "resume": "", "start_ep": 0,
})
EVALUATE_CLI_DEFAULTS = MappingProxyType({
    "seeds": 40, "ckpt": "checkpoints/osm_mappo/best.pt",
    "max_steps": 450, "police_speed": 16.0, "fugitive_speed": 9.0,
    "capture": 25.0, "vision": 150.0,
})

_SCRIPT_ALIASES = {
    "train_osm_pursuit": "train_osm_pursuit",
    "train_osm_pursuit.py": "train_osm_pursuit",
    "eval_trained": "eval_trained",
    "eval_trained.py": "eval_trained",
}
_SCRIPT_ENTRYPOINTS = {
    "train_osm_pursuit": "pursuit_evasion_rl.research.cli.train",
    "eval_trained": "pursuit_evasion_rl.research.cli.evaluate",
}
FORBIDDEN_ROOT_MODULES = frozenset({
    "demo_pursuit", "osm_obs_aug", "train_osm_pursuit", "eval_trained",
    "diag_episode", "render_zoom", "render_trained", "run_pursuit_eval",
})

LEGACY_IMPORT_ALIASES = MappingProxyType({
    "demo_pursuit": "pursuit_evasion_rl.research.policies.baselines",
    "osm_obs_aug": "pursuit_evasion_rl.research.variants.observations",
})

_CHECKPOINT_STATE_KEYS = frozenset({
    "police_actor", "police_critic", "fugitive_actor", "fugitive_critic",
    "police_optimizer", "fugitive_optimizer", "actor", "critic",
    "optimizer", "optimizer_state", "model_state", "state_dict",
    "scheduler", "scaler", "rollout_buffer", "rng_state",
})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _script_name(script: str) -> str:
    try:
        return _SCRIPT_ALIASES[script]
    except KeyError as exc:
        raise ResearchValidationError(
            "UNSUPPORTED_LEGACY_SCRIPT", "unsupported legacy script",
            path="script", expected=sorted(_SCRIPT_ENTRYPOINTS), actual=script,
        ) from exc


def _argument_mapping(arguments: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(arguments, Mapping):
        return dict(arguments)
    try:
        return dict(vars(arguments))
    except TypeError as exc:
        raise ResearchValidationError(
            "INVALID_LEGACY_ARGUMENTS", "arguments must be a mapping or argparse namespace",
            path="arguments", actual=type(arguments).__name__,
        ) from exc


def _pick(values: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: values[name] for name in names}


@dataclass(frozen=True, slots=True)
class LegacyConfigConversion:
    source_script: str
    package_entrypoint: str
    arguments: Mapping[str, Any]
    canonical_config: Mapping[str, Any]
    compatibility_status: str = LEGACY_ONLY
    research_trainer_eligible: bool = False
    schema_version: str = MIGRATION_SCHEMA_VERSION
    content_hash: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.compatibility_status != LEGACY_ONLY or self.research_trainer_eligible:
            raise ResearchValidationError(
                "LEGACY_SCOPE_ESCALATION",
                "legacy conversion cannot claim research-trainer eligibility",
            )
        object.__setattr__(self, "arguments", _freeze(canonical_data(self.arguments)))
        object.__setattr__(self, "canonical_config", _freeze(canonical_data(self.canonical_config)))
        calculated = content_hash(self)
        if self.content_hash is not None and self.content_hash != calculated:
            raise ResearchValidationError(
                "MIGRATION_HASH_MISMATCH", "config conversion hash mismatch",
                expected=calculated, actual=self.content_hash,
            )
        object.__setattr__(self, "content_hash", calculated)

    def as_dict(self) -> dict[str, Any]:
        return canonical_data(self)


def convert_legacy_cli_arguments(
    script: str, arguments: Mapping[str, Any] | Any,
) -> LegacyConfigConversion:
    """Convert a legacy argparse namespace without dropping any argument."""
    name = _script_name(script)
    supplied = _argument_mapping(arguments)
    defaults = TRAIN_CLI_DEFAULTS if name == "train_osm_pursuit" else EVALUATE_CLI_DEFAULTS
    values = dict(defaults)
    values.update(supplied)
    extras = {key: values[key] for key in values.keys() - defaults.keys()}

    if name == "train_osm_pursuit":
        converted = {
            "execution": _pick(values, "episodes", "start_ep", "max_steps"),
            "environment": _pick(values, "police_speed", "fugitive_speed", "capture", "vision"),
            "observation": {"profile": "legacy_augmented_28d", "clip": values["clip"]},
            "optimization": _pick(values, "lr", "gamma", "entropy_coef", "ppo_epochs", "update_every"),
            "reward": _pick(values, "capture_bonus", "time_penalty", "dist_scale", "team_coef", "own_coef", "regress_mult"),
            "checkpoint": _pick(values, "resume", "ckpt_every", "log_every"),
        }
    else:
        converted = {
            "evaluation": _pick(values, "seeds", "max_steps"),
            "environment": _pick(values, "police_speed", "fugitive_speed", "capture", "vision"),
            "checkpoint": _pick(values, "ckpt"),
            "observation": {"profile": "legacy_augmented_28d", "implicit_clip": 2000.0},
        }
    converted["legacy_arguments"] = values
    converted["extra_arguments"] = extras
    converted["limitations"] = (
        "not_a_research_protocol", "not_a_complete_resume_state",
        "no_budget_or_selection_fairness_claim",
    )
    return LegacyConfigConversion(
        source_script=name,
        package_entrypoint=_SCRIPT_ENTRYPOINTS[name],
        arguments=values,
        canonical_config=converted,
    )


def _array_bytes(value: Any) -> bytes:
    """Return deterministic raw bytes for NumPy arrays and Torch tensors."""
    module = type(value).__module__.split(".", 1)[0]
    if module == "torch":
        tensor = value.detach().cpu().contiguous()
        return tensor.view(dtype=importlib.import_module("torch").uint8).numpy().tobytes()
    if module == "numpy":
        return value.tobytes(order="C")
    raise TypeError


def _checkpoint_value(value: Any, *, path: str) -> Any:
    module = type(value).__module__.split(".", 1)[0]
    if module in {"torch", "numpy"} and hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            raw = _array_bytes(value)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ResearchValidationError(
                "UNSUPPORTED_CHECKPOINT_ARRAY", "checkpoint array cannot be fingerprinted",
                path=path, actual=str(exc),
            ) from exc
        return {
            "kind": "tensor" if module == "torch" else "ndarray",
            "dtype": str(value.dtype),
            "shape": tuple(int(size) for size in value.shape),
            "byte_size": len(raw),
            "sha256": sha256_bytes(raw),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ResearchValidationError(
                "INVALID_CHECKPOINT_KEY", "checkpoint mapping keys must be strings",
                path=path,
            )
        return {key: _checkpoint_value(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(
            _checkpoint_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    try:
        return canonical_data(value)
    except ResearchValidationError as exc:
        raise ResearchValidationError(
            "UNSUPPORTED_CHECKPOINT_VALUE",
            "checkpoint value cannot be preserved canonically",
            path=path, actual=type(value).__name__,
        ) from exc


@dataclass(frozen=True, slots=True)
class LegacyCheckpointConversion:
    source_content_hash: str
    source_format: str
    top_level_keys: tuple[str, ...]
    state_fingerprints: Mapping[str, Any]
    metadata: Mapping[str, Any]
    compatibility_status: str = LEGACY_ONLY
    complete_resume_state: bool = False
    schema_version: str = MIGRATION_SCHEMA_VERSION
    content_hash: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.source_content_hash):
            raise ResearchValidationError(
                "INVALID_SOURCE_HASH", "source checkpoint hash must be lowercase SHA-256",
                path="source_content_hash", actual=self.source_content_hash,
            )
        if self.complete_resume_state or self.compatibility_status != LEGACY_ONLY:
            raise ResearchValidationError(
                "LEGACY_SCOPE_ESCALATION",
                "legacy checkpoint conversion is not a complete ResumeState",
            )
        object.__setattr__(self, "state_fingerprints", _freeze(canonical_data(self.state_fingerprints)))
        object.__setattr__(self, "metadata", _freeze(canonical_data(self.metadata)))
        calculated = content_hash(self)
        if self.content_hash is not None and self.content_hash != calculated:
            raise ResearchValidationError(
                "MIGRATION_HASH_MISMATCH", "checkpoint conversion hash mismatch",
                expected=calculated, actual=self.content_hash,
            )
        object.__setattr__(self, "content_hash", calculated)

    def as_dict(self) -> dict[str, Any]:
        return canonical_data(self)


def convert_legacy_checkpoint(
    payload: Mapping[str, Any], *, source_content_hash: str,
) -> LegacyCheckpointConversion:
    """Fingerprint state while preserving every non-state metadata field."""
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise ResearchValidationError(
            "INVALID_LEGACY_CHECKPOINT", "checkpoint must be a string-keyed mapping",
            path="checkpoint", actual=type(payload).__name__,
        )
    state: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        target = state if key in _CHECKPOINT_STATE_KEYS else metadata
        target[key] = _checkpoint_value(value, path=f"checkpoint.{key}")
    source_format = (
        "legacy_mappo_v0"
        if {"police_actor", "police_critic"}.issubset(payload)
        else "legacy_checkpoint_mapping"
    )
    return LegacyCheckpointConversion(
        source_content_hash=source_content_hash,
        source_format=source_format,
        top_level_keys=tuple(sorted(payload)),
        state_fingerprints=state,
        metadata=metadata,
    )


def load_legacy_checkpoint(path: str | Path) -> LegacyCheckpointConversion:
    """Safely load a legacy checkpoint and convert it without executing pickle code."""
    source = Path(path)
    raw = source.read_bytes()
    torch = importlib.import_module("torch")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ResearchValidationError(
            "LEGACY_CHECKPOINT_LOAD_FAILED",
            "legacy checkpoint could not be loaded in weights-only mode",
            path=str(source), actual=str(exc),
        ) from exc
    return convert_legacy_checkpoint(payload, source_content_hash=hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True, slots=True)
class LegacyImportContract:
    legacy_module: str
    package_module: str
    deprecated: bool = True
    schema_version: str = MIGRATION_SCHEMA_VERSION


def legacy_import_contract(module: str) -> LegacyImportContract:
    try:
        target = LEGACY_IMPORT_ALIASES[module]
    except KeyError as exc:
        raise ResearchValidationError(
            "UNSUPPORTED_LEGACY_IMPORT", "unsupported legacy import alias",
            path="module", expected=sorted(LEGACY_IMPORT_ALIASES), actual=module,
        ) from exc
    return LegacyImportContract(module, target)


def resolve_legacy_symbol(module: str, symbol: str) -> Any:
    """Resolve a legacy symbol from its package owner without importing the root shim."""
    contract = legacy_import_contract(module)
    target = importlib.import_module(contract.package_module)
    try:
        return getattr(target, symbol)
    except AttributeError as exc:
        raise ResearchValidationError(
            "UNKNOWN_LEGACY_SYMBOL", "symbol is not exported by the package owner",
            path=f"{module}.{symbol}", actual=symbol,
        ) from exc


@dataclass(frozen=True, slots=True)
class ScriptShimContract:
    legacy_script: str
    package_module: str
    entrypoint: str
    package_entrypoint_available: bool
    stage: str
    forwards_argv_unchanged: bool = True
    schema_version: str = MIGRATION_SCHEMA_VERSION


def resolve_script_shim(script: str) -> ScriptShimContract:
    name = _script_name(script)
    package_module = _SCRIPT_ENTRYPOINTS[name]
    available = importlib.util.find_spec(package_module) is not None
    return ScriptShimContract(
        legacy_script=name,
        package_module=package_module,
        entrypoint="main",
        package_entrypoint_available=available,
        stage="package_entrypoint" if available else "legacy_fallback",
    )


def dispatch_legacy_script(
    script: str,
    argv: Sequence[str] | None,
    *,
    legacy_main: Callable[[Sequence[str] | None], int | None],
) -> int | None:
    """Dispatch to a real package CLI when present, otherwise keep legacy behavior.

    An existing but malformed package module is an error rather than a reason to
    silently fall back.  This prevents an incomplete trainer from being treated
    as the research implementation.
    """
    contract = resolve_script_shim(script)
    if contract.package_entrypoint_available:
        module = importlib.import_module(contract.package_module)
        entrypoint = getattr(module, contract.entrypoint, None)
        if not callable(entrypoint):
            raise ResearchValidationError(
                "INVALID_PACKAGE_ENTRYPOINT",
                "package CLI exists but has no callable main(argv) entrypoint",
                path=contract.package_module,
            )
        return entrypoint(argv)
    warnings.warn(
        f"{contract.legacy_script} remains a legacy compatibility path; "
        f"{contract.package_module} is not implemented yet",
        DeprecationWarning,
        stacklevel=2,
    )
    return legacy_main(argv)


@dataclass(frozen=True, slots=True)
class ForbiddenImport:
    path: str
    line: int
    module: str


def find_forbidden_research_imports(research_root: str | Path) -> tuple[ForbiddenImport, ...]:
    """Statically detect package-to-root import cycles using Python's AST."""
    root = Path(research_root)
    violations: list[ForbiddenImport] = []
    for source in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as exc:
            raise ResearchValidationError(
                "RESEARCH_IMPORT_SCAN_FAILED", "research source could not be parsed",
                path=str(source), actual=str(exc),
            ) from exc
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = (node.module,)
            for module in modules:
                root_module = module.split(".", 1)[0]
                if root_module in FORBIDDEN_ROOT_MODULES:
                    violations.append(ForbiddenImport(
                        path=source.relative_to(root).as_posix(),
                        line=int(getattr(node, "lineno", 0)),
                        module=module,
                    ))
    return tuple(violations)


def assert_research_import_boundary(research_root: str | Path) -> None:
    violations = find_forbidden_research_imports(research_root)
    if violations:
        raise ResearchValidationError(
            "FORBIDDEN_ROOT_IMPORT",
            "research package imports deprecated root modules",
            path=str(research_root),
            actual=[{"path": item.path, "line": item.line, "module": item.module} for item in violations],
        )


__all__ = (
    "EVALUATE_CLI_DEFAULTS", "FORBIDDEN_ROOT_MODULES", "LEGACY_IMPORT_ALIASES",
    "MIGRATION_SCHEMA_VERSION", "TRAIN_CLI_DEFAULTS", "ForbiddenImport",
    "LegacyCheckpointConversion", "LegacyConfigConversion", "LegacyImportContract",
    "ScriptShimContract", "assert_research_import_boundary", "convert_legacy_checkpoint",
    "convert_legacy_cli_arguments", "dispatch_legacy_script",
    "find_forbidden_research_imports", "legacy_import_contract",
    "load_legacy_checkpoint", "resolve_legacy_symbol", "resolve_script_shim",
)
