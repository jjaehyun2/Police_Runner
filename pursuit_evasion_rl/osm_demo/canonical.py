"""Deterministic JSON serialization and content-addressed identities.

Floating-point values are rounded to at most ``FLOAT_QUANTIZATION_DECIMALS``
decimal places with decimal ROUND_HALF_EVEN at the serialization boundary.
Mapping keys and set-like containers are sorted; list/tuple order is preserved
because it is semantically significant. Trailing fractional zeroes are omitted.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from enum import Enum
import hashlib
import json
import math
from typing import Any, Collection, Mapping

from .models import BoundedArea

FLOAT_QUANTIZATION_DECIMALS = 9
PROVENANCE_TIMESTAMP_FIELDS = frozenset(
    {"acquired_at", "committed_at", "created_at", "generated_at", "updated_at"}
)


class CanonicalSerializationError(ValueError):
    """Raised when a value cannot be represented as canonical JSON."""


def _quantize_number(value: float | Decimal, decimals: int, path: str) -> int | float:
    if isinstance(value, float) and not math.isfinite(value):
        raise CanonicalSerializationError(f"{path} must be a finite number")
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal.is_finite():
        raise CanonicalSerializationError(f"{path} must be a finite number")
    quantum = Decimal(1).scaleb(-decimals)
    required_precision = max(34, len(decimal.as_tuple().digits) + abs(decimal.adjusted()) + decimals + 4)
    try:
        with localcontext() as context:
            context.prec = required_precision
            quantized = decimal.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise CanonicalSerializationError(f"{path} cannot be quantized") from exc
    if quantized == 0:
        return 0
    if quantized == quantized.to_integral_value():
        return int(quantized)
    return float(quantized)


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _normalize(
    value: Any,
    *,
    decimals: int,
    excluded_fields: Collection[str],
    path: str,
    active: set[int],
) -> Any:
    if isinstance(value, Enum):
        return _normalize(
            value.value, decimals=decimals, excluded_fields=excluded_fields, path=path, active=active
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, Decimal)):
        return _quantize_number(value, decimals, path)

    identity = id(value)
    if identity in active:
        raise CanonicalSerializationError(f"{path} contains a reference cycle")
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            items = ((item.name, getattr(value, item.name)) for item in fields(value))
            return {
                key: _normalize(
                    item,
                    decimals=decimals,
                    excluded_fields=excluded_fields,
                    path=f"{path}.{key}",
                    active=active,
                )
                for key, item in items
                if key not in excluded_fields
            }
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise CanonicalSerializationError(f"{path} mapping keys must be strings")
            return {
                key: _normalize(
                    item,
                    decimals=decimals,
                    excluded_fields=excluded_fields,
                    path=f"{path}.{key}",
                    active=active,
                )
                for key, item in value.items()
                if key not in excluded_fields
            }
        if isinstance(value, (set, frozenset)):
            normalized = [
                _normalize(
                    item,
                    decimals=decimals,
                    excluded_fields=excluded_fields,
                    path=f"{path}[]",
                    active=active,
                )
                for item in value
            ]
            return sorted(normalized, key=_encode)
        if isinstance(value, (list, tuple)):
            return [
                _normalize(
                    item,
                    decimals=decimals,
                    excluded_fields=excluded_fields,
                    path=f"{path}[{index}]",
                    active=active,
                )
                for index, item in enumerate(value)
            ]
    finally:
        active.remove(identity)
    raise CanonicalSerializationError(f"{path} has unsupported type {type(value).__name__}")


def canonical_json(
    value: Any,
    *,
    float_decimals: int = FLOAT_QUANTIZATION_DECIMALS,
    exclude_fields: Collection[str] = (),
) -> bytes:
    """Return canonical UTF-8 JSON bytes for supported domain values.

    ``exclude_fields`` applies recursively and is intended only for explicitly
    declared provenance fields. Identity callers should use
    :func:`artifact_identity_hash`, which excludes known timestamps.
    """
    if isinstance(float_decimals, bool) or not isinstance(float_decimals, int) or float_decimals < 0:
        raise CanonicalSerializationError("float_decimals must be a nonnegative integer")
    normalized = _normalize(
        value,
        decimals=float_decimals,
        excluded_fields=frozenset(exclude_fields),
        path="$",
        active=set(),
    )
    return _encode(normalized).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of exact artifact bytes."""
    return hashlib.sha256(content).hexdigest()


def content_hash(value: Any, *, float_decimals: int = FLOAT_QUANTIZATION_DECIMALS) -> str:
    """Hash all canonical content, including provenance timestamps."""
    return sha256_bytes(canonical_json(value, float_decimals=float_decimals))


def artifact_identity_hash(
    value: Any,
    *,
    float_decimals: int = FLOAT_QUANTIZATION_DECIMALS,
    timestamp_fields: Collection[str] = PROVENANCE_TIMESTAMP_FIELDS,
) -> str:
    """Hash identity content while retaining timestamps only as provenance."""
    return sha256_bytes(
        canonical_json(value, float_decimals=float_decimals, exclude_fields=timestamp_fields)
    )


def cache_key_payload(
    area: BoundedArea,
    *,
    network_type: str,
    raw_schema_version: str,
    coarsener_version: str,
    preprocessing_settings_hash: str,
) -> dict[str, Any]:
    """Build the complete, timestamp-free cache identity payload."""
    values = {
        "network_type": network_type,
        "raw_schema_version": raw_schema_version,
        "coarsener_version": coarsener_version,
        "preprocessing_settings_hash": preprocessing_settings_hash,
    }
    empty = [name for name, value in values.items() if not isinstance(value, str) or not value.strip()]
    if empty:
        raise CanonicalSerializationError(f"Cache identity fields must be nonempty strings: {empty}")
    return {
        "bbox": {
            "north": area.north,
            "south": area.south,
            "east": area.east,
            "west": area.west,
        },
        **values,
    }


def build_cache_key(
    area: BoundedArea,
    *,
    network_type: str,
    raw_schema_version: str,
    coarsener_version: str,
    preprocessing_settings_hash: str,
    float_decimals: int = FLOAT_QUANTIZATION_DECIMALS,
) -> str:
    """Return a SHA-256 cache key over every configured cache identity input."""
    payload = cache_key_payload(
        area,
        network_type=network_type,
        raw_schema_version=raw_schema_version,
        coarsener_version=coarsener_version,
        preprocessing_settings_hash=preprocessing_settings_hash,
    )
    return content_hash(payload, float_decimals=float_decimals)


__all__ = (
    "FLOAT_QUANTIZATION_DECIMALS",
    "PROVENANCE_TIMESTAMP_FIELDS",
    "CanonicalSerializationError",
    "artifact_identity_hash",
    "build_cache_key",
    "cache_key_payload",
    "canonical_json",
    "content_hash",
    "sha256_bytes",
)
