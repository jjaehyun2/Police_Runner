"""Condition/ablation matrix factory (Requirements 7.1-7.8, 9.1-9.7, 10.1-10.5, 11.1-11.2).

The matrix is a star schema, not a full cartesian product: one shared
``default`` Condition per :class:`~pursuit_evasion_rl.research.domain.MapScenario`
(28D observation, the full audited reward set, global placement, both
stabilization factors off, the proposed policy) plus, for every ablation axis,
the arms that differ from that default in exactly one factor. A full cartesian
product across five independent axes would make every arm differ from every
other in multiple factors at once, which Requirement 9.12/10.11 explicitly
forbids for a causal ablation claim.

Every :class:`~pursuit_evasion_rl.research.domain.Condition`'s string fields
carry the *content hash* of the underlying sub-config (observation contract,
reward coefficients, placement config, stabilization arm), so ``Condition``'s
own ``content_hash`` changes exactly when anything result-affecting changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.domain import Condition, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.policies.baselines import (
    DIRECTED_SHORTEST_PATH_ID,
    ENCIRCLEMENT_ID,
    GREEDY_INTERCEPT_ID,
    LEGACY_SHARED_PPO_ID,
    PROPOSED_POLICY_ID,
)
from pursuit_evasion_rl.research.variants.observations import (
    OBSERVATION_21D_DIM,
    OBSERVATION_28D_DIM,
    observation_21d_contract,
    observation_28d_contract,
)
from pursuit_evasion_rl.research.variants.placement import (
    PlacementCurriculumConfig,
    PlacementStyle,
    all_placement_conditions,
)
from pursuit_evasion_rl.research.variants.rewards import (
    AUDITED_REWARD_COMPONENTS,
    RewardComponent,
    leave_one_component_out,
    symmetric_retreat_components,
)
from pursuit_evasion_rl.research.variants.stabilization import (
    StabilizationCondition,
    all_stabilization_arms,
)

FACTORY_SCHEMA_VERSION = "1.0"
DEFAULT_EVADER_ID = "goal_evader"

# Axis labels recorded on every generated Condition so a matrix report can
# group rows without re-deriving which factor a Condition actually varies.
AXIS_DEFAULT = "default"
AXIS_OBSERVATION = "observation"
AXIS_REWARD = "reward"
AXIS_PLACEMENT = "placement"
AXIS_STABILIZATION = "stabilization"
AXIS_BASELINE = "baseline"

_DEFAULT_OBSERVATION = observation_28d_contract()
_DEFAULT_OBSERVATION_HASH = _DEFAULT_OBSERVATION.config_hash
_DEFAULT_REWARD = AUDITED_REWARD_COMPONENTS
_DEFAULT_PLACEMENT = PlacementCurriculumConfig(style=PlacementStyle.GLOBAL)
_DEFAULT_STABILIZATION = StabilizationCondition()  # off/off


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """A generated Condition plus the ablation axis it belongs to and its arm label."""

    condition: Condition
    axis: str
    arm: str


def _label(observation_dim: int) -> str:
    return "observation_21d" if observation_dim == OBSERVATION_21D_DIM else "observation_28d"


def _stabilization_label(condition: StabilizationCondition) -> str:
    return condition.condition_id


def _build(
    *,
    condition_id: str,
    scenario: MapScenario,
    policy: str,
    observation_hash: str,
    reward_hash: str,
    placement_hash: str,
    uturn_label: str,
    hysteresis_label: str,
    axis: str,
    arm: str,
) -> ConditionSpec:
    condition = Condition(
        condition_id=condition_id,
        policy=policy,
        observation=observation_hash,
        reward=reward_hash,
        placement=placement_hash,
        uturn=uturn_label,
        hysteresis=hysteresis_label,
        scenario=scenario,
        evader=DEFAULT_EVADER_ID,
        budget={"axis": axis, "arm": arm},
    )
    return ConditionSpec(condition=condition, axis=axis, arm=arm)


def default_condition(scenario: MapScenario) -> ConditionSpec:
    """The single shared full/default Condition every ablation axis varies from."""
    return _build(
        condition_id=f"default__{scenario.value}",
        scenario=scenario,
        policy=PROPOSED_POLICY_ID,
        observation_hash=_DEFAULT_OBSERVATION.config_hash,
        reward_hash=_DEFAULT_REWARD.config_hash,
        placement_hash=_DEFAULT_PLACEMENT.config_hash,
        uturn_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
        hysteresis_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
        axis=AXIS_DEFAULT,
        arm="full",
    )


def observation_ablation_conditions(scenario: MapScenario) -> tuple[ConditionSpec, ...]:
    """Requirement 9.1: an Observation_21D vs Observation_28D one-factor pair.

    The two contracts are different dataclasses (``ObservationContract`` for
    21D, :class:`ObservationVariantContract` for 28D), so their hash is taken
    generically via :func:`content_hash` rather than a shared property.
    """
    specs = []
    for dimension, contract in ((OBSERVATION_21D_DIM, observation_21d_contract()), (OBSERVATION_28D_DIM, _DEFAULT_OBSERVATION)):
        specs.append(
            _build(
                condition_id=f"observation__{_label(dimension)}__{scenario.value}",
                scenario=scenario,
                policy=PROPOSED_POLICY_ID,
                observation_hash=content_hash(contract),
                reward_hash=_DEFAULT_REWARD.config_hash,
                placement_hash=_DEFAULT_PLACEMENT.config_hash,
                uturn_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                hysteresis_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                axis=AXIS_OBSERVATION,
                arm=_label(dimension),
            )
        )
    return tuple(specs)


def reward_ablation_conditions(scenario: MapScenario) -> tuple[ConditionSpec, ...]:
    """Requirement 9.6-9.7: leave-one-component-out arms plus the symmetric retreat arm.

    The audited full/asymmetric-retreat set is the shared default and is not
    repeated here.
    """
    specs = []
    for component in RewardComponent:
        reward = leave_one_component_out(component)
        specs.append(
            _build(
                condition_id=f"reward__no_{component.value}__{scenario.value}",
                scenario=scenario,
                policy=PROPOSED_POLICY_ID,
                observation_hash=_DEFAULT_OBSERVATION.config_hash,
                reward_hash=reward.config_hash,
                placement_hash=_DEFAULT_PLACEMENT.config_hash,
                uturn_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                hysteresis_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                axis=AXIS_REWARD,
                arm=f"no_{component.value}",
            )
        )
    symmetric = symmetric_retreat_components()
    specs.append(
        _build(
            condition_id=f"reward__symmetric_retreat__{scenario.value}",
            scenario=scenario,
            policy=PROPOSED_POLICY_ID,
            observation_hash=_DEFAULT_OBSERVATION.config_hash,
            reward_hash=symmetric.config_hash,
            placement_hash=_DEFAULT_PLACEMENT.config_hash,
            uturn_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
            hysteresis_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
            axis=AXIS_REWARD,
            arm="symmetric_retreat",
        )
    )
    return tuple(specs)


def placement_ablation_conditions(scenario: MapScenario) -> tuple[ConditionSpec, ...]:
    """Requirement 10.1: global/ring/mixed, one factor apart from the default."""
    specs = []
    for placement in all_placement_conditions(_DEFAULT_PLACEMENT):
        specs.append(
            _build(
                condition_id=f"placement__{placement.style.value}__{scenario.value}",
                scenario=scenario,
                policy=PROPOSED_POLICY_ID,
                observation_hash=_DEFAULT_OBSERVATION.config_hash,
                reward_hash=_DEFAULT_REWARD.config_hash,
                placement_hash=placement.config_hash,
                uturn_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                hysteresis_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                axis=AXIS_PLACEMENT,
                arm=placement.style.value,
            )
        )
    return tuple(specs)


def stabilization_ablation_conditions(scenario: MapScenario) -> tuple[ConditionSpec, ...]:
    """Requirement 10.4: the four off/on x off/on arms."""
    specs = []
    for arm in all_stabilization_arms(_DEFAULT_STABILIZATION):
        specs.append(
            _build(
                condition_id=f"stabilization__{arm.condition_id}__{scenario.value}",
                scenario=scenario,
                policy=PROPOSED_POLICY_ID,
                observation_hash=_DEFAULT_OBSERVATION.config_hash,
                reward_hash=_DEFAULT_REWARD.config_hash,
                placement_hash=_DEFAULT_PLACEMENT.config_hash,
                uturn_label=f"{'on' if arm.u_turn_suppression else 'off'}:{arm.condition_hash}",
                hysteresis_label=f"{'on' if arm.hysteresis else 'off'}:{arm.condition_hash}",
                axis=AXIS_STABILIZATION,
                arm=_stabilization_label(arm),
            )
        )
    return tuple(specs)


_BASELINE_POLICY_IDS: tuple[str, ...] = (
    PROPOSED_POLICY_ID,
    LEGACY_SHARED_PPO_ID,
    ENCIRCLEMENT_ID,
    DIRECTED_SHORTEST_PATH_ID,
    GREEDY_INTERCEPT_ID,
)


def baseline_conditions(scenario: MapScenario) -> tuple[ConditionSpec, ...]:
    """Requirement 11.1-11.4: one Condition per registered baseline, at the default arm.

    Baselines are compared at the shared default observation/reward/placement/
    stabilization arm so a baseline-vs-proposed difference is never confounded
    with an unrelated ablation factor.
    """
    specs = []
    for policy_id in _BASELINE_POLICY_IDS:
        if policy_id == PROPOSED_POLICY_ID:
            continue  # already the shared default Condition; not duplicated here.
        specs.append(
            _build(
                condition_id=f"baseline__{policy_id}__{scenario.value}",
                scenario=scenario,
                policy=policy_id,
                observation_hash=_DEFAULT_OBSERVATION.config_hash,
                reward_hash=_DEFAULT_REWARD.config_hash,
                placement_hash=_DEFAULT_PLACEMENT.config_hash,
                uturn_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                hysteresis_label=f"off:{_DEFAULT_STABILIZATION.condition_hash}",
                axis=AXIS_BASELINE,
                arm=policy_id,
            )
        )
    return tuple(specs)


_SCENARIOS: tuple[MapScenario, ...] = (MapScenario.INTERIOR_CONTAINED, MapScenario.BOUNDARY_ESCAPE)


def full_condition_matrix(scenarios: tuple[MapScenario, ...] = _SCENARIOS) -> tuple[ConditionSpec, ...]:
    """Every Condition in the pre-registered matrix, across the given scenarios."""
    if not scenarios:
        _fail("EMPTY_SCENARIO_SET", "full_condition_matrix requires at least one MapScenario")
    specs: list[ConditionSpec] = []
    for scenario in scenarios:
        specs.append(default_condition(scenario))
        specs.extend(observation_ablation_conditions(scenario))
        specs.extend(reward_ablation_conditions(scenario))
        specs.extend(placement_ablation_conditions(scenario))
        specs.extend(stabilization_ablation_conditions(scenario))
        specs.extend(baseline_conditions(scenario))
    return tuple(specs)


def validate_condition_matrix(specs: tuple[ConditionSpec, ...]) -> None:
    """Reject a matrix with duplicate condition_ids or a non-single-factor arm.

    Every non-default Condition must differ from its scenario's default in
    exactly the field(s) its own axis owns; any other differing field is
    undeclared-factor drift (Requirements 9.12, 10.11).
    """
    if not specs:
        _fail("EMPTY_CONDITION_MATRIX", "a condition matrix must not be empty")
    ids = [spec.condition.condition_id for spec in specs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        _fail("DUPLICATE_CONDITION_ID", "condition matrix contains duplicate condition_id values", actual=duplicates)

    defaults = {spec.condition.scenario: spec.condition for spec in specs if spec.axis == AXIS_DEFAULT}
    axis_fields = {
        AXIS_OBSERVATION: ("observation",),
        AXIS_REWARD: ("reward",),
        AXIS_PLACEMENT: ("placement",),
        AXIS_STABILIZATION: ("uturn", "hysteresis"),
        AXIS_BASELINE: ("policy",),
    }
    for spec in specs:
        if spec.axis == AXIS_DEFAULT:
            continue
        default = defaults.get(spec.condition.scenario)
        if default is None:
            _fail(
                "MISSING_DEFAULT_CONDITION",
                "every scenario in the matrix requires its own default Condition",
                actual=spec.condition.scenario.value,
            )
        owned = axis_fields[spec.axis]
        for field_name in ("policy", "observation", "reward", "placement", "uturn", "hysteresis"):
            if field_name in owned:
                continue
            if getattr(spec.condition, field_name) != getattr(default, field_name):
                _fail(
                    "UNDECLARED_FACTOR_DRIFT",
                    f"Condition {spec.condition.condition_id!r} on axis {spec.axis!r} "
                    f"differs from its scenario default in undeclared field {field_name!r}",
                    path=field_name,
                )


__all__ = (
    "AXIS_BASELINE",
    "AXIS_DEFAULT",
    "AXIS_OBSERVATION",
    "AXIS_PLACEMENT",
    "AXIS_REWARD",
    "AXIS_STABILIZATION",
    "DEFAULT_EVADER_ID",
    "FACTORY_SCHEMA_VERSION",
    "ConditionSpec",
    "baseline_conditions",
    "default_condition",
    "full_condition_matrix",
    "observation_ablation_conditions",
    "placement_ablation_conditions",
    "reward_ablation_conditions",
    "stabilization_ablation_conditions",
    "validate_condition_matrix",
)
