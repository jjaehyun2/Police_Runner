"""Compatibility imports for the migrated 21D/28D observation adapters.

New code should import from :mod:`pursuit_evasion_rl.research.variants.observations`.
This module intentionally contains no observation calculations.
"""

from pursuit_evasion_rl.research.variants.observations import (
    AUG_EXTRA,
    AUG_OBS_DIM,
    AUGMENTED_FIELD_ORDER,
    AUGMENTED_NORMALIZATION,
    AugmentedObs,
    DECISION_TIMING,
    DEFAULT_NEAR_RADIUS_M,
    OBSERVATION_21D_DIM,
    OBSERVATION_28D_DIM,
    OBSERVATION_28D_PROFILE_ID,
    Observation21DAdapter,
    Observation28DAdapter,
    ObservationVariantContract,
    canonical_observation_config_hash,
    observation_21d_contract,
    observation_28d_contract,
)

__all__ = (
    "AUG_EXTRA",
    "AUG_OBS_DIM",
    "AUGMENTED_FIELD_ORDER",
    "AUGMENTED_NORMALIZATION",
    "AugmentedObs",
    "DECISION_TIMING",
    "DEFAULT_NEAR_RADIUS_M",
    "OBSERVATION_21D_DIM",
    "OBSERVATION_28D_DIM",
    "OBSERVATION_28D_PROFILE_ID",
    "Observation21DAdapter",
    "Observation28DAdapter",
    "ObservationVariantContract",
    "canonical_observation_config_hash",
    "observation_21d_contract",
    "observation_28d_contract",
)
