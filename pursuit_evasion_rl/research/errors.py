"""Stable, serializable error records for fail-closed research gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

ERROR_SCHEMA_VERSION = "1.0"


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_safe(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    code: str
    message: str
    path: str | None = None
    expected: Any = None
    actual: Any = None
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = ERROR_SCHEMA_VERSION
    content_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("ErrorRecord code and message must be non-empty")
        object.__setattr__(self, "expected", _safe(self.expected))
        object.__setattr__(self, "actual", _safe(self.actual))
        object.__setattr__(self, "details", MappingProxyType(_safe(self.details)))
        from .canonical import content_hash
        calculated = content_hash(self)
        if self.content_hash is not None and self.content_hash != calculated:
            raise ValueError("ErrorRecord content_hash does not match its content")
        object.__setattr__(self, "content_hash", calculated)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "details": dict(self.details),
        }


class ResearchValidationError(ValueError):
    """Validation failure with a stable code and content-addressed record."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        expected: Any = None,
        actual: Any = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.record = ErrorRecord(
            code=code,
            message=message,
            path=path,
            expected=expected,
            actual=actual,
            details=details or {},
        )

    @property
    def code(self) -> str:
        return self.record.code

    @property
    def path(self) -> str | None:
        return self.record.path

    @property
    def expected(self) -> Any:
        return self.record.expected

    @property
    def actual(self) -> Any:
        return self.record.actual

    def as_dict(self) -> dict[str, Any]:
        return self.record.as_dict()


__all__ = ("ERROR_SCHEMA_VERSION", "ErrorRecord", "ResearchValidationError")
