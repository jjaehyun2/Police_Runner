"""Deprecated compatibility imports for the migrated pursuit baselines."""

from __future__ import annotations

import warnings

warnings.warn(
    "demo_pursuit is deprecated; import pursuit_evasion_rl.research.policies.baselines instead",
    DeprecationWarning,
    stacklevel=2,
)

from pursuit_evasion_rl.research.policies.baselines import (  # noqa: E402,F401
    EncirclementParameters,
    EncirclementPolice,
    GoalAssignment,
    GoalAssignmentPlan,
    GoalEvader,
    GoalEvaderParameters,
    SmartEvader,
    _Graph,
    make_interior_network,
)

__all__ = (
    "EncirclementParameters",
    "EncirclementPolice",
    "GoalAssignment",
    "GoalAssignmentPlan",
    "GoalEvader",
    "GoalEvaderParameters",
    "SmartEvader",
    "_Graph",
    "make_interior_network",
)
