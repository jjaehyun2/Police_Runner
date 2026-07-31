"""Deterministic, lossless JSON primitives and SHA-256 identities."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Collection, Mapping

from .errors import ResearchValidationError


class CanonicalSerializationError(ResearchValidationError):
    """A value cannot be represented by the canonical JSON contract."""


def _fail(message: str, *, path: str, actual: Any = None) -> None:
    raise CanonicalSerializationError(
        "CANONICAL_SERIALIZATION_ERROR", message, path=path, actual=actual
    )


def _number(value: float | Decimal, path: str) -> int | float:
    if not value.is_finite() if isinstance(value, Decimal) else not math.isfinite(value):
        _fail(f"{path} must be a finite number", path=path, actual=repr(value))
    if value == 0:
        return 0
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        converted = float(value)
        if not math.isfinite(converted) or Decimal(str(converted)) != value.normalize():
            _fail(f"{path} Decimal is not losslessly representable as JSON", path=path, actual=str(value))
        return converted
    return value


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _normalize(value: Any, *, path: str, active: set[int], omit_root: Collection[str], root: bool) -> Any:
    if isinstance(value, Enum):
        return _normalize(value.value, path=path, active=active, omit_root=(), root=False)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, Decimal)):
        return _number(value, path)

    identity = id(value)
    if identity in active:
        _fail(f"{path} contains a reference cycle", path=path)
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            pairs = ((item.name, getattr(value, item.name)) for item in fields(value))
            return {
                key: _normalize(item, path=f"{path}.{key}", active=active, omit_root=(), root=False)
                for key, item in pairs
                if not (root and key in omit_root)
            }
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                _fail(f"{path} mapping keys must be strings", path=path)
            return {
                key: _normalize(item, path=f"{path}.{key}", active=active, omit_root=(), root=False)
                for key, item in value.items()
                if not (root and key in omit_root)
            }
        if isinstance(value, (set, frozenset)):
            items = [
                _normalize(item, path=f"{path}[]", active=active, omit_root=(), root=False)
                for item in value
            ]
            return sorted(items, key=_encoded)
        if isinstance(value, (tuple, list)):
            return [
                _normalize(item, path=f"{path}[{index}]", active=active, omit_root=(), root=False)
                for index, item in enumerate(value)
            ]
    finally:
        active.remove(identity)
    _fail(f"{path} has unsupported type {type(value).__name__}", path=path, actual=type(value).__name__)


def canonical_data(value: Any, *, omit_root_fields: Collection[str] = ()) -> Any:
    """Return canonical JSON-compatible data, rejecting cycles and non-finite values."""
    return _normalize(
        value,
        path="$",
        active=set(),
        omit_root=frozenset(omit_root_fields),
        root=True,
    )


def canonical_json(value: Any, *, omit_root_fields: Collection[str] = ()) -> bytes:
    """Return deterministic UTF-8 JSON; object keys are sorted and null is retained."""
    return _encoded(canonical_data(value, omit_root_fields=omit_root_fields)).encode("utf-8")


def canonical_loads(content: bytes | str) -> Any:
    """Parse canonical JSON and reject non-standard numeric constants."""
    try:
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        return json.loads(
            text,
            parse_constant=lambda token: _fail(
                f"Non-finite JSON constant {token!r} is forbidden", path="$", actual=token
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalSerializationError(
            "INVALID_CANONICAL_JSON", "Input is not valid UTF-8 JSON", actual=str(exc)
        ) from exc


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def content_hash(value: Any) -> str:
    """Hash model content, excluding only its root self-referential hash field."""
    return sha256_bytes(canonical_json(value, omit_root_fields=("content_hash",)))


def verify_content_hash(value: Any, expected: str) -> bool:
    return content_hash(value) == expected


__all__ = (
    "CanonicalSerializationError",
    "canonical_data",
    "canonical_json",
    "canonical_loads",
    "content_hash",
    "sha256_bytes",
    "verify_content_hash",
)
