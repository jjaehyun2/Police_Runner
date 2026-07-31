"""Deterministic single, batch and paired OSM pursuit episode runners.

This module drives :class:`~pursuit_evasion_rl.osm_demo.environment.OSMRoadPursuitEnv`
episodes with a uniform police :class:`~pursuit_evasion_rl.osm_demo.policies.PolicePolicy`
and the :class:`~pursuit_evasion_rl.osm_demo.fugitive.OSMHeuristicFugitive`, and
records every episode as an append-only :class:`~pursuit_evasion_rl.osm_demo.models.EpisodeRecord`
(design section 7, "OSM episode environment and policies"; Requirements 9.1-9.9,
14.2-14.4).

Determinism strategy (Requirement 9.8, design section 7):

* Every random component draws from an *independent* stream seeded by
  ``(run_seed, episode_index, purpose)`` via :func:`derive_stream_seed`.  The
  purposes are separated (``"placement"``, ``"fugitive"``, ``"policy"``) so that
  adding metric, rendering or logging code that consumes a new stream can never
  perturb placement or fugitive/policy randomness.  Two runs with the same
  ``run_seed`` therefore reproduce identical placements, action sequences,
  vehicle-state sequences, events and outcomes.

Paired comparison (Requirements 14.3-14.4):

* :func:`run_paired_batch` replays the *same* placement seed and fugitive random
  stream for a learned and a baseline police policy, so the two policies are
  evaluated on identical initial placements and identical fugitive randomness and
  can be compared with equivalent accounting.  Because both policies share the
  ``(run_seed, episode_index)`` derivation, neither the placement nor the fugitive
  stream depends on which policy is running.

The runner is fully offline: it requires no external OSM access and no
checkpoint of its own.  A learned policy, when supplied, is any object satisfying
the :class:`PolicePolicy` protocol.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .canonical import content_hash
from .environment import FUGITIVE_ID, STAY_ACTION, OSMRoadPursuitEnv
from .fugitive import OSMHeuristicFugitive
from .models import (
    POLICE_COUNT,
    DomainValidationError,
    EpisodeConfig,
    EpisodeOutcome,
    EpisodeRecord,
    EpisodeState,
    EpisodeTransition,
    ModelNetwork,
    VehiclePlacement,
)
from .policies import PolicePolicy

# A fugitive factory builds a fresh, rng-injected fugitive per episode so its
# random stream is reset deterministically for every ``(run_seed, episode_index)``.
FugitiveFactory = Callable[[np.random.Generator], OSMHeuristicFugitive]

# Explicit seven-vehicle placement: six police plus one fugitive.
Placement = tuple[Sequence[VehiclePlacement], VehiclePlacement]

_STREAM_PURPOSES = ("placement", "fugitive", "policy")


def derive_stream_seed(run_seed: int, episode_index: int, purpose: str) -> int:
    """Return an independent 64-bit seed for ``(run_seed, episode_index, purpose)``.

    The derivation is a stable SHA-256 over the three identity inputs, so each
    purpose owns a statistically independent stream and adding a new purpose can
    never shift an existing one (design section 7; Requirement 9.8).
    """
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise DomainValidationError("INVALID_SEED", "run_seed must be an integer", actual=run_seed)
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise DomainValidationError(
            "INVALID_EPISODE_INDEX", "episode_index must be a nonnegative integer", actual=episode_index
        )
    if not isinstance(purpose, str) or not purpose:
        raise DomainValidationError("INVALID_STREAM_PURPOSE", "purpose must be a nonempty string")
    payload = f"{run_seed}:{episode_index}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def default_fugitive_factory(
    *, vision_range_m: float = 30.0, random_rate: float = 0.0
) -> FugitiveFactory:
    """Build a fugitive factory that injects the per-episode random stream.

    The returned factory constructs an :class:`OSMHeuristicFugitive` bound to the
    supplied generator, keeping every random / tie-breaking decision reproducible
    from the derived fugitive seed (Requirements 9.1-9.3, 14.1).
    """

    def factory(rng: np.random.Generator) -> OSMHeuristicFugitive:
        return OSMHeuristicFugitive(vision_range_m=vision_range_m, random_rate=random_rate, rng=rng)

    return factory


@dataclass(frozen=True, slots=True)
class PairedEpisodeRecords:
    """A learned/baseline episode pair sharing placement and fugitive randomness."""

    episode_index: int
    seed: int
    learned: EpisodeRecord
    baseline: EpisodeRecord


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------


def run_single_episode(
    network: ModelNetwork,
    config: EpisodeConfig,
    police_policy: PolicePolicy,
    *,
    run_id: str,
    run_seed: int,
    episode_index: int = 0,
    fugitive_factory: FugitiveFactory | None = None,
    placement: Placement | None = None,
    extra_hashes: dict[str, str] | None = None,
) -> EpisodeRecord:
    """Run one deterministic episode and return its :class:`EpisodeRecord`.

    Placement uses the ``"placement"`` stream seed and the fugitive uses the
    ``"fugitive"`` stream seed, both derived from ``(run_seed, episode_index)``.
    The record captures the initial state, every applied action, the resulting
    vehicle-state sequence, per-step events and the terminal outcome/priority
    together with input hashes (Requirements 9.1-9.9, 14.2-14.4).
    """
    if fugitive_factory is None:
        fugitive_factory = default_fugitive_factory()

    placement_seed = derive_stream_seed(run_seed, episode_index, "placement")
    fugitive_seed = derive_stream_seed(run_seed, episode_index, "fugitive")
    fugitive = fugitive_factory(np.random.default_rng(fugitive_seed))

    env = OSMRoadPursuitEnv(network, config)
    if placement is not None:
        police_placements, fugitive_placement = placement
        observations, _ = env.reset(
            seed=placement_seed,
            options={"police": list(police_placements), "fugitive": fugitive_placement},
        )
    else:
        observations, _ = env.reset(seed=placement_seed)

    initial_state = env.episode_state()
    transitions: list[EpisodeTransition] = []

    # The environment guarantees a terminal outcome by ``max_steps`` (timeout),
    # so the loop is bounded; the explicit cap defends against a malformed config.
    for _ in range(config.max_steps + 1):
        if env.outcome is not None:
            break
        state = env.episode_state()
        actions, police_action_indices, fugitive_action = _build_step_actions(
            network=network,
            config=config,
            police_policy=police_policy,
            fugitive=fugitive,
            state=state,
            observations=observations,
        )
        observations, _, _, _, info = env.step(actions)
        post_state = env.episode_state()
        recorded_actions = tuple(police_action_indices) + (fugitive_action[0],)
        transitions.append(
            EpisodeTransition(
                step=post_state.step,
                actions=recorded_actions,
                state=post_state,
                events=tuple(info["events"]),
            )
        )

    outcome = env.outcome if env.outcome is not None else EpisodeOutcome.TIMEOUT
    terminal_priority = env.terminal_priority or "timeout"

    hashes = _episode_hashes(
        network=network,
        config=config,
        police_policy=police_policy,
        initial_state=initial_state,
        run_seed=run_seed,
        episode_index=episode_index,
        extra_hashes=extra_hashes,
    )

    return EpisodeRecord(
        run_id=run_id,
        episode_id=f"{run_id}:ep{episode_index:06d}",
        seed=derive_stream_seed(run_seed, episode_index, "episode"),
        initial_state=initial_state,
        transitions=tuple(transitions),
        outcome=outcome,
        terminal_priority=terminal_priority,
        hashes=hashes,
    )


def _build_step_actions(
    *,
    network: ModelNetwork,
    config: EpisodeConfig,
    police_policy: PolicePolicy,
    fugitive: OSMHeuristicFugitive,
    state: EpisodeState,
    observations: dict,
) -> tuple[dict, tuple[int, ...], list[int]]:
    """Assemble the per-agent actions for one environment step.

    Police actions come from six atomic recommendations resolved on the current
    state (headings taken from the environment so the policy and environment share
    the same deterministic action ordering).  The fugitive is driven by an
    environment-compatible provider using the police metric positions captured at
    the start of the step; the provider records the fugitive's first-hop action so
    the transition log reflects the applied decision.
    """
    headings = [state.incoming_headings.get(f"police_{index}") for index in range(POLICE_COUNT)]
    recommendations = police_policy.recommend(
        police=state.police,
        fugitive=state.fugitive,
        step=state.step,
        max_steps=config.max_steps,
        incoming_headings=headings,
    )
    police_action_indices = tuple(int(rec.action_index) for rec in recommendations)
    police_actions = {
        f"police_{index}": police_action_indices[index] for index in range(POLICE_COUNT)
    }

    police_xy = [
        tuple(observations[f"police_{index}"]["position_xy"]) for index in range(POLICE_COUNT)
    ]
    base_provider = fugitive.env_provider(network, police_xy)
    # Default to a logged stay: a fugitive still moving on a segment resolves no
    # routing microstep this step and its provider is never invoked.
    recorded_action = [STAY_ACTION]

    def recording_provider(agent_id, intersection_id, ordered, hop):
        action = base_provider(agent_id, intersection_id, ordered, hop)
        if hop == 0:
            recorded_action[0] = int(action)
        return action

    actions = {**police_actions, FUGITIVE_ID: recording_provider}
    return actions, police_action_indices, recorded_action


def _episode_hashes(
    *,
    network: ModelNetwork,
    config: EpisodeConfig,
    police_policy: PolicePolicy,
    initial_state: EpisodeState,
    run_seed: int,
    episode_index: int,
    extra_hashes: dict[str, str] | None,
) -> dict[str, str]:
    """Build the immutable input-hash / provenance map stored on the record."""
    hashes = {
        "network": content_hash(network),
        "config": content_hash(config),
        "initial_state": content_hash(initial_state),
        "policy_profile": str(getattr(police_policy, "profile", "unknown")),
        "policy_experimental": str(bool(getattr(police_policy, "experimental", False))).lower(),
        "run_seed": str(run_seed),
        "episode_index": str(episode_index),
    }
    if extra_hashes:
        hashes.update({str(key): str(value) for key, value in extra_hashes.items()})
    return hashes


# ---------------------------------------------------------------------------
# Batch of independent episodes
# ---------------------------------------------------------------------------


def run_batch(
    network: ModelNetwork,
    config: EpisodeConfig,
    police_policy: PolicePolicy,
    *,
    run_id: str,
    run_seed: int,
    episode_count: int,
    fugitive_factory: FugitiveFactory | None = None,
    placement: Placement | None = None,
    extra_hashes: dict[str, str] | None = None,
) -> tuple[EpisodeRecord, ...]:
    """Run ``episode_count`` independent episodes indexed ``0..episode_count-1``.

    Each episode derives its own independent placement/fugitive/policy streams
    from ``(run_seed, episode_index)``, so an episode's result is identical whether
    it runs alone or within the batch (Requirements 9.1-9.2, 9.8).
    """
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count <= 0:
        raise DomainValidationError(
            "INVALID_EPISODE_COUNT", "episode_count must be a positive integer", actual=episode_count
        )
    return tuple(
        run_single_episode(
            network,
            config,
            police_policy,
            run_id=run_id,
            run_seed=run_seed,
            episode_index=episode_index,
            fugitive_factory=fugitive_factory,
            placement=placement,
            extra_hashes=extra_hashes,
        )
        for episode_index in range(episode_count)
    )


# ---------------------------------------------------------------------------
# Paired learned-vs-baseline batch
# ---------------------------------------------------------------------------


def run_paired_batch(
    network: ModelNetwork,
    config: EpisodeConfig,
    learned_policy: PolicePolicy,
    baseline_policy: PolicePolicy,
    *,
    run_id: str,
    run_seed: int,
    episode_count: int,
    fugitive_factory: FugitiveFactory | None = None,
    placement: Placement | None = None,
) -> tuple[PairedEpisodeRecords, ...]:
    """Run a learned and a baseline policy on identical placements and fugitive
    randomness for every episode index (Requirements 14.3-14.4).

    Both sides derive placement and fugitive streams from the same
    ``(run_seed, episode_index)``, so neither depends on which policy is running.
    The two policies are therefore compared on the same initial placement and the
    same fugitive random stream, enabling equivalent-accounting comparison.
    """
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count <= 0:
        raise DomainValidationError(
            "INVALID_EPISODE_COUNT", "episode_count must be a positive integer", actual=episode_count
        )

    paired: list[PairedEpisodeRecords] = []
    for episode_index in range(episode_count):
        learned_record = run_single_episode(
            network,
            config,
            learned_policy,
            run_id=f"{run_id}:learned",
            run_seed=run_seed,
            episode_index=episode_index,
            fugitive_factory=fugitive_factory,
            placement=placement,
        )
        baseline_record = run_single_episode(
            network,
            config,
            baseline_policy,
            run_id=f"{run_id}:baseline",
            run_seed=run_seed,
            episode_index=episode_index,
            fugitive_factory=fugitive_factory,
            placement=placement,
        )
        paired.append(
            PairedEpisodeRecords(
                episode_index=episode_index,
                seed=derive_stream_seed(run_seed, episode_index, "episode"),
                learned=learned_record,
                baseline=baseline_record,
            )
        )
    return tuple(paired)


__all__ = (
    "FugitiveFactory",
    "Placement",
    "PairedEpisodeRecords",
    "derive_stream_seed",
    "default_fugitive_factory",
    "run_single_episode",
    "run_batch",
    "run_paired_batch",
)
