"""Shared helpers for the beat-greedy work: networks, sealed cases, policies,
and a parallel paired evaluator.

Cases are built exactly like ``main_study.build_episode_case`` (GLOBAL
placement, BOUNDARY_ESCAPE, real Daejeon maps) except that the fugitive RNG
seed comes from SHA-256 instead of Python's per-process randomized ``hash``,
so the same (tag, seed, episode) is the same case in every process and every
run -- required for paired comparisons across separate invocations.
"""
from __future__ import annotations

from dataclasses import replace as _replace
import hashlib
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, ModelNetwork  # noqa: E402
from pursuit_evasion_rl.research.domain import DataKind, MapScenario  # noqa: E402
from pursuit_evasion_rl.research.evaluation.paired import (  # noqa: E402
    FugitiveRNGSpec, TerminationConfig, replay_episode, scenario_outcome_domain, seal_episode_case,
)
from pursuit_evasion_rl.research.experiments.main_study import GreedyPolicyAdapter  # noqa: E402
from pursuit_evasion_rl.research.maps.snapshots import OfflineSnapshotStore, load_snapshot_spec  # noqa: E402
from pursuit_evasion_rl.research.policies.baselines import (  # noqa: E402
    EncirclementPolice, _Graph, directed_shortest_path_policy, greedy_intercept_policy,
)
from pursuit_evasion_rl.research.policies.masked_mappo import ResearchMaskedMAPPO  # noqa: E402
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle, generate_placement  # noqa: E402

import os
MAX_STEPS = int(os.environ.get("BG_MAX_STEPS", "450"))
POLICE_SPEED_MPS = float(os.environ.get("BG_POLICE_SPEED", "16.0"))
FUGITIVE_SPEED_MPS = float(os.environ.get("BG_FUGITIVE_SPEED", "9.0"))
CAPTURE_RADIUS_M = 25.0

_NETWORKS: dict[str, ModelNetwork] = {}


def load_network(name: str) -> ModelNetwork:
    """``name`` is a config stem: daejeon_train / daejeon_validation / daejeon_test / ..."""
    if name not in _NETWORKS:
        store = OfflineSnapshotStore(str(REPO_ROOT))
        spec = load_snapshot_spec(REPO_ROOT / f"configs/research/maps/{name}.yaml")
        _NETWORKS[name] = store.import_snapshot(spec).snapshot.network
    return _NETWORKS[name]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


SOLVABLE_ONLY = os.environ.get("BG_SOLVABLE", "0") == "1"
MIN_FUGITIVE_EXIT_M = 200.0


def is_solvable(network: ModelNetwork, placement) -> bool:
    """Excludes cases no policy can influence: the fugitive starts where no
    officer can ever reach it, or so close to a boundary exit that it is
    gone before anyone could react (< MIN_FUGITIVE_EXIT_M of road)."""
    from obs_ext import boundary_distance, graph_for
    g = graph_for(network)
    fnode = int(placement.fugitive.intersection_id)
    to_f = g.dist_to(fnode)
    if not any(int(p.intersection_id) in to_f for p in placement.police):
        return False
    return boundary_distance(network).get(fnode, float("inf")) >= MIN_FUGITIVE_EXIT_M


def _memoize_outgoing_order() -> None:
    """ordered_outgoing_segment_ids is pure but re-hashes segment content on every
    call (~half of episode wall time); cache it per (network, node, heading)."""
    from pursuit_evasion_rl.osm_demo import coarsening, environment, fugitive, policies
    original = coarsening.ordered_outgoing_segment_ids
    if getattr(original, "_memoized", False):
        return
    cache: dict = {}

    def cached(network, intersection_id, incoming_heading=None):
        key = (id(network), intersection_id, incoming_heading)
        hit = cache.get(key)
        if hit is None or hit[0] is not network:
            hit = cache[key] = (network, original(network, intersection_id, incoming_heading))
        return hit[1]

    cached._memoized = True
    for module in (coarsening, environment, fugitive, policies):
        module.ordered_outgoing_segment_ids = cached


_memoize_outgoing_order()
_CASE_CACHE: dict = {}


def make_case(network: ModelNetwork, tag: str, episode_index: int):
    key = (id(network), tag, episode_index, POLICE_SPEED_MPS, FUGITIVE_SPEED_MPS, SOLVABLE_ONLY, MAX_STEPS)
    hit = _CASE_CACHE.get(key)
    if hit is not None and hit[0] is network:
        return hit[1]
    case = _make_case(network, tag, episode_index)
    if len(_CASE_CACHE) > 512:
        _CASE_CACHE.clear()
    _CASE_CACHE[key] = (network, case)
    return case


def _make_case(network: ModelNetwork, tag: str, episode_index: int):
    rng_seed = stable_seed(tag, episode_index)
    placement = generate_placement(
        network, _replace(PlacementCurriculumConfig(style=PlacementStyle.GLOBAL), placement_seed=rng_seed)
    )
    attempt = 0
    while SOLVABLE_ONLY and not is_solvable(network, placement):
        attempt += 1
        rng_seed = stable_seed(tag, episode_index, "resample", attempt)
        placement = generate_placement(
            network, _replace(PlacementCurriculumConfig(style=PlacementStyle.GLOBAL), placement_seed=rng_seed)
        )
    environment = EpisodeConfig(
        dt_s=1.0, police_speed_mps=POLICE_SPEED_MPS, fugitive_speed_mps=FUGITIVE_SPEED_MPS,
        capture_radius_m=CAPTURE_RADIUS_M, max_steps=MAX_STEPS,
    )
    termination = TerminationConfig(
        outcome_domain=scenario_outcome_domain(MapScenario.BOUNDARY_ESCAPE), timeout_step_limit=MAX_STEPS,
    )
    return seal_episode_case(
        network=network, data_kind=DataKind.ACTUAL_OSM_MAP, scenario=MapScenario.BOUNDARY_ESCAPE,
        police=placement.police, fugitive=placement.fugitive,
        fugitive_rng=FugitiveRNGSpec(stream_id=f"{tag}-ep{episode_index}", seed=rng_seed),
        environment=environment, termination=termination,
    )


def load_learned(checkpoint_path: str | Path) -> tuple[ResearchMaskedMAPPO, dict]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    policy = ResearchMaskedMAPPO(
        actor_obs_dim=payload["actor_obs_dim"], critic_context_dim=payload["critic_context_dim"],
        action_dim=payload["action_dim"], num_officers=payload["num_officers"],
        hidden_dims=tuple(payload["hidden_dims"]), device="cpu",
    )
    policy.load_state_dict(payload["model"])
    return policy, payload


def adapter_factory_for(payload: dict) -> Callable | None:
    """The observation adapter a checkpoint was trained with (None = audited 28D)."""
    profile = payload.get("observation", {}).get("profile", "osm_topology_augmented_v1")
    if profile == "osm_topology_augmented_v1":
        return None
    from obs_ext import ExtendedObservationAdapter, ExtendedObservationAdapterV2, EXT_PROFILE_ID, EXT2_PROFILE_ID
    if profile == EXT2_PROFILE_ID:
        return lambda net, clip, near: ExtendedObservationAdapterV2(
            net, clip_distance_m=clip, near_radius_m=near,
            police_speed_mps=POLICE_SPEED_MPS, fugitive_speed_mps=FUGITIVE_SPEED_MPS,
        )
    if profile == EXT_PROFILE_ID:
        return lambda net, clip, near: ExtendedObservationAdapter(
            net, clip_distance_m=clip, near_radius_m=near,
            police_speed_mps=POLICE_SPEED_MPS, fugitive_speed_mps=FUGITIVE_SPEED_MPS,
        )
    raise ValueError(f"unknown observation profile {profile}")


# -- worker-side policy construction (policies are rebuilt per process) ------
_WORKER: dict = {}


def _build(spec: str, network: ModelNetwork):
    if spec == "greedy":
        return {"team_policy": greedy_intercept_policy(network)}
    if spec == "dsp":
        return {"team_policy": directed_shortest_path_policy(network)}
    if spec == "encircle":
        return {"team_policy": EncirclementPolice(
            network, graph=_Graph(network), fugitive_speed_mps=FUGITIVE_SPEED_MPS, police_speed_mps=POLICE_SPEED_MPS,
        )}
    if spec.startswith(("rmacro:", "emacro:", "tmacro:", "cmacro:")):  # relative features; metric: e straight line,
        import macro                                                       # t travel time, c traffic-aware travel time
        metric = {"e": "euclid", "t": "travel", "c": "traffic"}.get(spec[0], "road")
        arg = spec[7:]
        if arg.startswith("init") and arg[4:].isdigit():
            torch.manual_seed(int(arg[4:]))
            scorer = macro.CandidateScorer((128, 128))
        elif arg.startswith("sinit") and arg[5:].isdigit():     # team-size-agnostic scorer
            torch.manual_seed(int(arg[5:]))
            scorer = macro.CandidateScorer((128, 128), kind="size")
        elif arg == "assign":
            scorer = None
        else:
            payload = torch.load(arg, map_location="cpu", weights_only=False)
            scorer = macro.CandidateScorer(tuple(payload["hidden"]), kind=payload.get("scorer_kind", "onehot"))
            scorer.load_state_dict(payload["scorer"])
        return {"team_policy": macro.MacroTeamPolicy(network, scorer, vp=POLICE_SPEED_MPS, vf=FUGITIVE_SPEED_MPS,
                                                     fixed_rule="assign" if arg == "assign" else None, relative=True,
                                                     metric=metric)}
    if spec.startswith("macro:"):
        import macro
        arg = spec[6:]
        if arg.startswith("init") and arg[4:].isdigit():      # untrained scorer, seeded, argmax
            torch.manual_seed(int(arg[4:]))
            return {"team_policy": macro.MacroTeamPolicy(network, macro.CandidateScorer((128, 128)),
                                                         vp=POLICE_SPEED_MPS, vf=FUGITIVE_SPEED_MPS)}
        if arg.startswith("random") and arg[6:].isdigit():
            return {"team_policy": macro.MacroTeamPolicy(network, macro.CandidateScorer(), vp=POLICE_SPEED_MPS,
                vf=FUGITIVE_SPEED_MPS, mode="sample", rng=np.random.default_rng(int(arg[6:])))}
        if arg in ("pursue", "random", "assign"):
            scorer = macro.CandidateScorer()
            return {"team_policy": macro.MacroTeamPolicy(
                network, scorer, vp=POLICE_SPEED_MPS, vf=FUGITIVE_SPEED_MPS,
                mode="sample" if arg == "random" else "argmax", fixed_rule=arg if arg in ("pursue", "assign") else None,
            )}
        payload = torch.load(arg, map_location="cpu", weights_only=False)
        scorer = macro.CandidateScorer(tuple(payload["hidden"]))
        scorer.load_state_dict(payload["scorer"])
        return {"team_policy": macro.MacroTeamPolicy(network, scorer, vp=POLICE_SPEED_MPS, vf=FUGITIVE_SPEED_MPS)}
    if spec.startswith("attn:"):
        import attn
        arg = spec[5:]
        net = attn.AttnAssigner()
        if arg != "init":
            net.load_state_dict(torch.load(arg, map_location="cpu", weights_only=False)["net"])
        net.eval()
        return {"team_policy": attn.AttnTeamPolicy(network, net, vp=POLICE_SPEED_MPS, vf=FUGITIVE_SPEED_MPS)}
    if spec.startswith("rule:"):
        import rules
        return {"team_policy": rules.build(spec[5:], network)}
    if spec.startswith("pep:"):                            # PEPCRL-MVP port (pepcrl.py)
        import pepcrl
        return {"team_policy": pepcrl.load_policy(spec[4:], network)}
    if spec.startswith("ckpt:"):
        policy, payload = load_learned(spec[5:])
        policy.actor.eval() if hasattr(policy, "actor") else None
        return {
            "policy": GreedyPolicyAdapter(policy, "learned"),
            "observation_adapter_factory": adapter_factory_for(payload),
        }
    raise ValueError(spec)


def _init_worker(network_name: str, spec: str) -> None:
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    network = load_network(network_name)
    _WORKER["network"] = network
    _WORKER["spec"] = spec
    _WORKER["network_name"] = network_name


def _run_one(args: tuple[str, int]) -> tuple[int, str, int]:
    tag, episode_index = args
    network = _WORKER["network"]
    case = make_case(network, tag, episode_index)
    # Stateful team policies (encirclement keeps history) must start fresh per case.
    kwargs = _build(_WORKER["spec"], network)
    record = replay_episode(case, network=network, **kwargs)
    return episode_index, record.outcome.value, record.physical_steps


def evaluate(spec: str, network_name: str, tag: str, episodes: int, workers: int = int(os.environ.get("BG_WORKERS", "40"))) -> dict[int, tuple[str, int]]:
    """Run ``episodes`` sealed cases under ``spec``; returns {episode: (outcome, steps)}."""
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(network_name, spec)) as pool:
        results = pool.map(_run_one, [(tag, i) for i in range(episodes)], chunksize=1)
    return {index: (outcome, steps) for index, outcome, steps in results}


def capture_rate(results: dict[int, tuple[str, int]]) -> float:
    return sum(outcome == "capture" for outcome, _ in results.values()) / len(results)
