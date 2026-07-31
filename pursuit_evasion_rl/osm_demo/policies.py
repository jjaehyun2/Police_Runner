"""Action masks, learned-policy loading and atomic recommendation decoding.

This module implements the inference side of the OSM road-pursuit demo
(design section 6, "Observation and action adapters" and section 7, "OSM episode
environment and policies").  It is deliberately separate from the environment so
recommendations can be produced for an external API without stepping an episode.

Responsibilities (Requirements 6.4-6.5, 8.1-8.6):

* Map the six-entry action space -- five ordered outgoing exits plus stay at
  index :data:`STAY_ACTION` -- and build a mask whose movement bits correspond
  exactly to the ordered outgoing segments of the officer's decision
  intersection (Requirement 6.4).
* Apply that mask *before* turning actor logits into probabilities so masked
  slots can never receive probability mass, and classify the selection of a
  nonexistent exit as an error rather than silently moving (Requirement 6.5).
* Produce *atomic* :class:`ActionRecommendation` outputs: action index, segment
  id (or ``None`` for stay), next intersection, validity, probability, profile
  and a compatibility-report reference must all be present, or the whole request
  fails with no partial recommendations (Requirements 8.2, 8.3).
* Require structural/observation-profile/network/execution compatibility before
  invoking a learned OSM actor (Requirements 8.1, 8.6, 7.5), while isolating the
  explicit legacy-direct route as experimental only (Requirement 8.5).

Learned weights are loaded on CPU with PyTorch's mandatory weights-only mode via
the shared, class-free loader in :mod:`.checkpoint`; a saved model is never
reconstructed or imported.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .checkpoint import (
    EXPECTED_ARCHITECTURE,
    CheckpointInspection,
    CheckpointInspector,
    _actor_state,
    _safe_cpu_load,
)
from .coarsening import ordered_outgoing_segment_ids
from .models import (
    POLICE_COUNT,
    CheckpointManifest,
    CheckStatus,
    CompatibilityReport,
    DomainValidationError,
    ModelNetwork,
    ObservationContract,
    VehiclePlacement,
)
from .observations import (
    ACTION_DIM,
    ACTION_SLOTS,
    DEFAULT_CLIP_DISTANCE_M,
    LEGACY_PROFILE_ID,
    OSM_PROFILE_ID,
    OSMTopologyV1Adapter,
    osm_topology_v1_contract,
)

# Stay occupies the slot immediately after the five ordered outgoing exits and is
# always a valid action (design section 6; Requirement 6.4).
STAY_ACTION = ACTION_SLOTS

# Observation-profile-style identifier for the non-learned shortest-path baseline.
# It is not an inference observation profile; it labels the policy kind so runs,
# metrics and exports keep the baseline distinct from learned/legacy routes
# (Requirements 8.5-8.10, 14.3).
BASELINE_PROFILE_ID = "baseline_shortest_path"

# ``ActorForward`` maps a single 21-value observation to ``ACTION_DIM`` logits.
ActorForward = Callable[[np.ndarray], "np.ndarray | Sequence[float]"]


@dataclass(frozen=True, slots=True)
class ActionRecommendation:
    """One officer's atomic recommendation.

    Every field is mandatory; a recommendation is only ever constructed when the
    complete decision could be resolved.  A masked or incomplete decode raises
    instead of yielding a partial recommendation (Requirement 8.3).
    """

    agent_id: str
    action_index: int
    segment_id: int | None
    next_intersection_id: int
    valid: bool
    probability: float
    profile: str
    compatibility_report_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_index": self.action_index,
            "segment_id": self.segment_id,
            "next_intersection_id": self.next_intersection_id,
            "valid": self.valid,
            "probability": self.probability,
            "profile": self.profile,
            "compatibility_report_id": self.compatibility_report_id,
        }


# ---------------------------------------------------------------------------
# Masking and probability decoding
# ---------------------------------------------------------------------------


def build_action_mask(
    network: ModelNetwork,
    decision_intersection_id: int,
    incoming_heading: float | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return the six-entry action mask and ordered outgoing segment ids.

    Movement bits ``0..4`` are set for each of the (at most five) ordered
    outgoing segments of ``decision_intersection_id``; the stay bit
    (:data:`STAY_ACTION`) is always set (Requirement 6.4).
    """
    ordered = ordered_outgoing_segment_ids(network, decision_intersection_id, incoming_heading)
    if len(ordered) > ACTION_SLOTS:
        # Degree splitting must have reduced out-degree to at most five before
        # inference; a larger fan-out is a network-preparation error, not an
        # inference decision (Requirements 6.2, 6.3).
        raise DomainValidationError(
            "OUT_DEGREE_EXCEEDS_ACTION_SPACE",
            f"Intersection {decision_intersection_id} exposes more than {ACTION_SLOTS} outgoing exits",
            expected=ACTION_SLOTS,
            actual=len(ordered),
        )
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[STAY_ACTION] = True
    for slot in range(len(ordered)):
        mask[slot] = True
    return mask, ordered


def masked_probabilities(
    logits: "np.ndarray | Sequence[float]", mask: "np.ndarray | Sequence[bool]"
) -> np.ndarray:
    """Softmax over logits after applying ``mask`` (mask before probabilities).

    Masked slots receive zero probability.  The logits must be a finite length
    ``ACTION_DIM`` vector; anything else is an incomplete actor output and fails
    the whole request (Requirement 8.3).
    """
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.shape[0] != ACTION_DIM:
        raise DomainValidationError(
            "INCOMPLETE_ACTOR_OUTPUT",
            f"Actor must produce exactly {ACTION_DIM} logits",
            expected=ACTION_DIM,
            actual=int(values.shape[0]),
        )
    if not np.all(np.isfinite(values)):
        raise DomainValidationError(
            "INCOMPLETE_ACTOR_OUTPUT", "Actor produced a non-finite logit"
        )
    mask_array = np.asarray(mask, dtype=bool).reshape(-1)
    if mask_array.shape[0] != ACTION_DIM:
        raise DomainValidationError(
            "INVALID_ACTION_MASK",
            f"Action mask must have exactly {ACTION_DIM} entries",
            expected=ACTION_DIM,
            actual=int(mask_array.shape[0]),
        )
    if not mask_array.any():
        # Stay is always valid, so an all-false mask indicates a construction bug.
        raise DomainValidationError("INVALID_ACTION_MASK", "Action mask has no valid action")

    masked = np.where(mask_array, values, -np.inf)
    shifted = masked - np.max(masked[mask_array])
    exponentiated = np.where(mask_array, np.exp(shifted), 0.0)
    total = exponentiated.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise DomainValidationError(
            "INCOMPLETE_ACTOR_OUTPUT", "Masked actor logits did not yield a valid distribution"
        )
    return (exponentiated / total).astype(np.float64)


def decode_recommendation(
    *,
    agent_id: str,
    network: ModelNetwork,
    decision_intersection_id: int,
    mask: "np.ndarray | Sequence[bool]",
    ordered_segment_ids: Sequence[int],
    probabilities: "np.ndarray | Sequence[float]",
    profile: str,
    compatibility_report_id: str,
    action_index: int | None = None,
) -> ActionRecommendation:
    """Resolve a single atomic recommendation from a masked distribution.

    When ``action_index`` is ``None`` the greedy (argmax) masked action is used.
    Selecting a masked slot, or a movement slot without a corresponding outgoing
    segment, is a classified error (Requirement 6.5) and never a partial result
    (Requirement 8.3).
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    mask_array = np.asarray(mask, dtype=bool).reshape(-1)
    if probabilities.shape[0] != ACTION_DIM or mask_array.shape[0] != ACTION_DIM:
        raise DomainValidationError(
            "INCOMPLETE_ACTOR_OUTPUT",
            f"Probabilities and mask must have exactly {ACTION_DIM} entries",
            expected=ACTION_DIM,
        )

    if action_index is None:
        # Restrict argmax to valid slots so a masked slot is never chosen.
        selectable = np.where(mask_array, probabilities, -np.inf)
        chosen = int(np.argmax(selectable))
    else:
        chosen = int(action_index)
    if not 0 <= chosen < ACTION_DIM:
        raise DomainValidationError(
            "INVALID_ACTION_INDEX",
            f"Action index must be in [0, {ACTION_DIM - 1}]",
            expected=[0, ACTION_DIM - 1],
            actual=chosen,
        )
    if not bool(mask_array[chosen]):
        raise DomainValidationError(
            "MASKED_ACTION_SELECTED",
            f"Action index {chosen} is masked for intersection {decision_intersection_id}",
            actual=chosen,
        )

    if chosen == STAY_ACTION:
        segment_id: int | None = None
        next_intersection_id = decision_intersection_id
    else:
        if chosen >= len(ordered_segment_ids):
            # A movement slot with no backing segment must be an error, never a
            # silent no-op (Requirement 6.5).
            raise DomainValidationError(
                "NONEXISTENT_EXIT_SELECTED",
                f"Action index {chosen} has no outgoing segment at intersection {decision_intersection_id}",
                expected=len(ordered_segment_ids),
                actual=chosen,
            )
        segment_id = int(ordered_segment_ids[chosen])
        segment = next((item for item in network.segments if item.id == segment_id), None)
        if segment is None:
            raise DomainValidationError(
                "NONEXISTENT_EXIT_SELECTED",
                f"Ordered action slot references unknown segment {segment_id}",
                actual=segment_id,
            )
        next_intersection_id = int(segment.end_id)

    return ActionRecommendation(
        agent_id=agent_id,
        action_index=chosen,
        segment_id=segment_id,
        next_intersection_id=next_intersection_id,
        valid=True,
        probability=float(probabilities[chosen]),
        profile=profile,
        compatibility_report_id=compatibility_report_id,
    )


# ---------------------------------------------------------------------------
# Compatibility gating
# ---------------------------------------------------------------------------


def _section_status(section: Sequence[Any], code: str) -> CheckStatus | None:
    for check in section:
        if check.code == code:
            return check.status
    return None


def osm_execution_config() -> dict[str, Any]:
    """Runtime execution contract for six-police OSM inference (Requirement 7.5)."""
    return {
        "police_count": POLICE_COUNT,
        "fixed_max_degree": ACTION_SLOTS,
        "observation_dim": EXPECTED_ARCHITECTURE.observation_dim,
        "action_dim": EXPECTED_ARCHITECTURE.action_dim,
        "hidden_dims": EXPECTED_ARCHITECTURE.hidden_dims,
    }


def _require_not_blocked(report: CompatibilityReport) -> None:
    if report.as_dict()["inference_blocked"]:
        raise DomainValidationError(
            "INFERENCE_BLOCKED",
            "Structural, network or execution compatibility failed; inference is blocked",
            actual=report.as_dict()["mismatches"],
        )


def _require_osm_semantics(report: CompatibilityReport) -> None:
    """Reject legacy/unknown observation semantics for the verified OSM route.

    A legacy or unproven profile can *load* but only the explicit experimental
    route may invoke it; the OSM route must be pointed at the OSM training path
    instead (Requirements 8.6, 7.6).
    """
    status = _section_status(report.semantics, "OBSERVATION_PROFILE")
    if status is not CheckStatus.PASS:
        raise DomainValidationError(
            "SEMANTIC_PROFILE_REQUIRED",
            "OSM inference requires a matching osm_topology_v1 observation profile; "
            "use the OSM training path (fine-tune or from-scratch) before claiming OSM performance",
            expected=OSM_PROFILE_ID,
            actual=None if status is None else status.value,
        )


# ---------------------------------------------------------------------------
# Learned actor loading
# ---------------------------------------------------------------------------


def _strip_network_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize actor keys to a bare ``Sequential`` layout (``0.weight`` ...)."""
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        if name.startswith("network."):
            name = name[len("network.") :]
        normalized[name] = value
    return normalized


def load_actor_forward(
    checkpoint_path: str | Path,
    observation_dim: int,
    hidden_dims: Sequence[int],
    action_dim: int,
) -> ActorForward:
    """Load the ``police_actor`` MLP on CPU with weights-only safety.

    A saved model class is never imported or reconstructed; only allowlisted
    tensors are read (design section 5) and copied into a locally built MLP with
    the inferred ``observation_dim -> hidden_dims -> action_dim`` shape.
    """
    import torch
    import torch.nn as nn

    payload = _safe_cpu_load(Path(checkpoint_path))
    _, actor_state = _actor_state(payload)
    state = _strip_network_prefix(actor_state)

    layers: list[nn.Module] = []
    previous = int(observation_dim)
    for hidden in hidden_dims:
        layers.append(nn.Linear(previous, int(hidden)))
        layers.append(nn.ReLU())
        previous = int(hidden)
    layers.append(nn.Linear(previous, int(action_dim)))
    network = nn.Sequential(*layers)
    try:
        network.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()}, strict=True)
    except Exception as exc:  # shape/key drift after a passing structural check
        raise DomainValidationError(
            "ACTOR_WEIGHT_LOAD_FAILED",
            "police_actor tensors could not be loaded into the inferred architecture",
            actual=type(exc).__name__,
        ) from exc
    network.eval()

    def forward(observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
            logits = network(tensor)
        return logits.reshape(-1).cpu().numpy()

    return forward


# ---------------------------------------------------------------------------
# Team policies
# ---------------------------------------------------------------------------


@runtime_checkable
class PolicePolicy(Protocol):
    """Uniform team police-policy protocol (design section 7).

    A team policy turns the current six-police / fugitive placement into six
    atomic :class:`ActionRecommendation` outputs (or fails the whole request).
    The learned OSM recommender, its explicit legacy-direct experimental
    subclass and the non-learned shortest-path :class:`BaselinePolicePolicy` all
    implement this protocol, so an :class:`~pursuit_evasion_rl.osm_demo` runner
    can compare them on identical placements and fugitive randomness with the
    same accounting (Requirement 14.3).

    ``profile`` identifies the policy kind (``osm_topology_v1``,
    ``legacy_grid_v0`` or ``baseline_shortest_path``) and ``experimental`` marks
    routes that must be excluded from verified OSM performance evidence
    (Requirements 8.5-8.6, 8.10).
    """

    profile: str
    experimental: bool

    def recommend(
        self,
        *,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_headings: Sequence[float | None] | None = None,
    ) -> tuple[ActionRecommendation, ...]:
        ...


class BaselinePolicePolicy:
    """Deterministic non-learned shortest-path pursuit baseline (Requirement 14.3).

    Each officer *independently* selects the valid outgoing segment that minimizes
    the directed network distance to the fugitive's current (or last observed)
    position, where a candidate's cost is its own arc length plus the directed
    shortest-path distance from its destination intersection to the fugitive.
    Ties are broken by ascending action-slot order (the deterministic
    :func:`ordered_outgoing_segment_ids` ordering), so two runs with the same
    placements produce byte-for-byte identical recommendations. When no outgoing
    segment can reach the fugitive the officer stays.

    The baseline requires no checkpoint and performs no learned inference, so it
    is always non-experimental and eligible as a paired comparison policy. It
    shares the atomic-recommendation contract with :class:`LearnedPolicePolicy`
    via :func:`build_action_mask` and :func:`decode_recommendation`, so an
    incomplete or illegal decision fails the whole request rather than returning
    a partial team recommendation (Requirements 8.2-8.3).
    """

    def __init__(
        self,
        network: ModelNetwork,
        *,
        profile: str = BASELINE_PROFILE_ID,
        police_count: int = POLICE_COUNT,
        report_id: str = BASELINE_PROFILE_ID,
    ) -> None:
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError("INVALID_NETWORK", "A ModelNetwork instance is required")
        self.network = network
        self.profile = str(profile)
        self.police_count = int(police_count)
        self.experimental = False
        self._report_id = str(report_id)
        self._segments = {segment.id: segment for segment in network.segments}
        # Directed adjacency over segment arc lengths; identical accounting to the
        # observation adapter so learned and baseline distances are comparable.
        self._adjacency: dict[int, list[tuple[int, float]]] = {
            item.id: [] for item in network.intersections
        }
        for segment in network.segments:
            self._adjacency[segment.start_id].append((segment.end_id, float(segment.length_m)))
        self._dijkstra_cache: dict[int, dict[int, float]] = {}

    # -- distance helpers ---------------------------------------------------

    def _dijkstra(self, source: int) -> dict[int, float]:
        cached = self._dijkstra_cache.get(source)
        if cached is not None:
            return cached
        distances: dict[int, float] = {source: 0.0}
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances.get(node, math.inf):
                continue
            for neighbor, weight in self._adjacency.get(node, ()):  # directed edges only
                candidate = distance + weight
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        self._dijkstra_cache[source] = distances
        return distances

    def _decision_intersection(self, placement: VehiclePlacement) -> int:
        """First intersection reached moving forward (decisions happen there)."""
        if placement.segment_id is not None:
            if placement.segment_id not in self._segments:
                raise DomainValidationError(
                    "INVALID_POSITION", "Placement references an unknown segment", actual=placement.segment_id
                )
            return int(self._segments[placement.segment_id].end_id)
        if placement.intersection_id is None or placement.intersection_id not in self._adjacency:
            raise DomainValidationError(
                "INVALID_POSITION",
                "Placement references an unknown intersection",
                actual=placement.intersection_id,
            )
        return int(placement.intersection_id)

    def _entry_intersection(self, placement: VehiclePlacement) -> tuple[int, float]:
        """Intersection through which the fugitive position is entered, plus offset."""
        if placement.segment_id is not None:
            segment = self._segments[placement.segment_id]
            return int(segment.start_id), float(placement.progress) * float(segment.length_m)
        return int(placement.intersection_id), 0.0

    def _select_slot(
        self,
        ordered: Sequence[int],
        fugitive_entry: int,
        fugitive_offset: float,
    ) -> int:
        """Return the action slot minimizing directed distance to the fugitive.

        Ties keep the lowest slot (strict ``<`` comparison over the ordered slots),
        which is the deterministic action-slot order (Requirement 14.3). When no
        exit can reach the fugitive the officer stays.
        """
        best_slot = STAY_ACTION
        best_cost = math.inf
        for slot, segment_id in enumerate(ordered):
            segment = self._segments[segment_id]
            distances = self._dijkstra(int(segment.end_id))
            downstream = distances.get(fugitive_entry)
            if downstream is None:
                continue  # this exit cannot reach the fugitive
            cost = float(segment.length_m) + downstream + fugitive_offset
            if cost < best_cost:  # strict keeps the lowest action slot on ties
                best_cost = cost
                best_slot = slot
        return best_slot

    # -- team recommendation ------------------------------------------------

    def recommend(
        self,
        *,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_headings: Sequence[float | None] | None = None,
    ) -> tuple[ActionRecommendation, ...]:
        """Produce six complete baseline recommendations or fail the whole request.

        ``step`` and ``max_steps`` are accepted for interface parity with
        :class:`LearnedPolicePolicy`; the greedy baseline does not depend on them.
        """
        if len(police) != self.police_count:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT",
                f"Exactly {self.police_count} police placements are required",
                expected=self.police_count,
                actual=len(police),
            )
        if incoming_headings is not None and len(incoming_headings) != self.police_count:
            raise DomainValidationError(
                "INVALID_HEADINGS",
                "One incoming heading per officer is required when headings are supplied",
                expected=self.police_count,
                actual=len(incoming_headings),
            )

        fugitive_entry, fugitive_offset = self._entry_intersection(fugitive)

        recommendations: list[ActionRecommendation] = []
        for index in range(self.police_count):
            heading = None if incoming_headings is None else incoming_headings[index]
            decision_id = self._decision_intersection(police[index])
            mask, ordered = build_action_mask(self.network, decision_id, heading)
            chosen = self._select_slot(ordered, fugitive_entry, fugitive_offset)
            probabilities = np.zeros(ACTION_DIM, dtype=np.float64)
            probabilities[chosen] = 1.0
            recommendation = decode_recommendation(
                agent_id=f"police_{index}",
                network=self.network,
                decision_intersection_id=decision_id,
                mask=mask,
                ordered_segment_ids=ordered,
                probabilities=probabilities,
                profile=self.profile,
                compatibility_report_id=self._report_id,
                action_index=chosen,
            )
            recommendations.append(recommendation)
        return tuple(recommendations)


class LearnedPolicePolicy:
    """Learned six-police OSM recommender behind a strict compatibility gate."""

    def __init__(
        self,
        network: ModelNetwork,
        adapter: OSMTopologyV1Adapter,
        actor: ActorForward,
        *,
        report: CompatibilityReport,
        profile: str = OSM_PROFILE_ID,
        police_count: int = POLICE_COUNT,
        experimental: bool = False,
    ) -> None:
        self.network = network
        self.adapter = adapter
        self._actor = actor
        self.report = report
        self.profile = profile
        self.police_count = int(police_count)
        self.experimental = bool(experimental)
        _require_not_blocked(report)
        if not experimental:
            _require_osm_semantics(report)

    @classmethod
    def from_checkpoint(
        cls,
        network: ModelNetwork,
        checkpoint_path: str | Path,
        *,
        adapter: OSMTopologyV1Adapter | None = None,
        manifest: CheckpointManifest | Mapping[str, Any] | None = None,
        network_hash: str = "",
        clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M,
    ) -> "LearnedPolicePolicy":
        """Inspect, gate and load a checkpoint for verified OSM inference."""
        adapter = adapter or OSMTopologyV1Adapter(network, clip_distance_m=clip_distance_m)
        inspection = CheckpointInspector().inspect(
            checkpoint_path,
            manifest,
            selected_observation_contract=adapter.contract,
            execution_config=osm_execution_config(),
            network_hash=network_hash,
        )
        report = inspection.report
        _require_not_blocked(report)
        _require_osm_semantics(report)
        contract = inspection.inferred_contract
        actor = load_actor_forward(
            checkpoint_path,
            contract.observation_dim or EXPECTED_ARCHITECTURE.observation_dim,
            contract.hidden_dims or EXPECTED_ARCHITECTURE.hidden_dims,
            contract.action_dim or EXPECTED_ARCHITECTURE.action_dim,
        )
        return cls(network, adapter, actor, report=report, profile=OSM_PROFILE_ID)

    def recommend(
        self,
        *,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_headings: Sequence[float | None] | None = None,
    ) -> tuple[ActionRecommendation, ...]:
        """Produce six complete recommendations or fail the whole request.

        Any masked, incomplete or otherwise unresolvable officer decision raises,
        so a partial team recommendation is never returned (Requirement 8.3).
        """
        if len(police) != self.police_count:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT",
                f"Exactly {self.police_count} police placements are required",
                expected=self.police_count,
                actual=len(police),
            )
        if incoming_headings is not None and len(incoming_headings) != self.police_count:
            raise DomainValidationError(
                "INVALID_HEADINGS",
                "One incoming heading per officer is required when headings are supplied",
                expected=self.police_count,
                actual=len(incoming_headings),
            )

        recommendations: list[ActionRecommendation] = []
        for index in range(self.police_count):
            heading = None if incoming_headings is None else incoming_headings[index]
            placement = police[index]
            view = self.adapter._view(placement)
            decision_id, _ = self.adapter._reach_intersection(view)
            mask, ordered = build_action_mask(self.network, decision_id, heading)
            observation = self.adapter.observe(
                police_index=index,
                police=police,
                fugitive=fugitive,
                step=step,
                max_steps=max_steps,
                incoming_heading=heading,
            )
            logits = self._actor(observation)
            probabilities = masked_probabilities(logits, mask)
            recommendation = decode_recommendation(
                agent_id=f"police_{index}",
                network=self.network,
                decision_intersection_id=decision_id,
                mask=mask,
                ordered_segment_ids=ordered,
                probabilities=probabilities,
                profile=self.profile,
                compatibility_report_id=self.report.report_id,
            )
            recommendations.append(recommendation)
        return tuple(recommendations)


class LegacyDirectPolicePolicy(LearnedPolicePolicy):
    """Explicit, experimental-only legacy-direct OSM route (Requirement 8.5).

    Structural compatibility alone permits loading; semantic mismatch is expected
    and the results are isolated as ``experimental`` and excluded from verified
    OSM evidence.  The route must be opted into explicitly.
    """

    def __init__(
        self,
        network: ModelNetwork,
        adapter: OSMTopologyV1Adapter,
        actor: ActorForward,
        *,
        report: CompatibilityReport,
        allow_experimental_legacy: bool,
        police_count: int = POLICE_COUNT,
    ) -> None:
        if not allow_experimental_legacy:
            raise DomainValidationError(
                "LEGACY_INFERENCE_NOT_OPTED_IN",
                "Legacy-direct OSM inference must be explicitly opted into and is experimental only",
            )
        super().__init__(
            network,
            adapter,
            actor,
            report=report,
            profile=LEGACY_PROFILE_ID,
            police_count=police_count,
            experimental=True,
        )


__all__ = (
    "ACTION_DIM",
    "ACTION_SLOTS",
    "STAY_ACTION",
    "BASELINE_PROFILE_ID",
    "ActionRecommendation",
    "BaselinePolicePolicy",
    "LearnedPolicePolicy",
    "LegacyDirectPolicePolicy",
    "PolicePolicy",
    "build_action_mask",
    "decode_recommendation",
    "load_actor_forward",
    "masked_probabilities",
    "osm_execution_config",
)
