"""Checked, bounded network presets for reproducible OSM acquisition."""

from dataclasses import dataclass
from types import MappingProxyType

from .models import (
    BoundedArea,
    CONFIG_VERSION,
    DOMAIN_SCHEMA_VERSION,
    DomainValidationError,
    validate_config_version,
    validate_schema_version,
)


@dataclass(frozen=True, slots=True)
class NetworkPreset:
    name: str
    area: BoundedArea
    network_type: str
    config_version: str
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        if not self.name.strip():
            raise DomainValidationError(
                "EMPTY_NAME", "Network preset name must not be empty", path="name"
            )
        if self.network_type != "drive":
            raise DomainValidationError(
                "INVALID_NETWORK_TYPE",
                "OSM network preset must use network_type 'drive'",
                path="network_type",
                expected="drive",
                actual=self.network_type,
            )
        validate_config_version(self.config_version)
        if self.area.config_version != self.config_version:
            raise DomainValidationError(
                "CONFIG_VERSION_MISMATCH",
                "Preset and bounded area config versions must match",
                path="area.config_version",
                expected=self.config_version,
                actual=self.area.config_version,
            )


DAEJEON_DRIVE_PRESET = NetworkPreset(
    name="daejeon-central-drive",
    area=BoundedArea(
        name="Daejeon Central Demo",
        north=36.365,
        south=36.345,
        east=127.405,
        west=127.380,
        max_area_km2=10.0,
        config_version=CONFIG_VERSION,
    ),
    network_type="drive",
    config_version=CONFIG_VERSION,
)

PRESETS = MappingProxyType({DAEJEON_DRIVE_PRESET.name: DAEJEON_DRIVE_PRESET})
