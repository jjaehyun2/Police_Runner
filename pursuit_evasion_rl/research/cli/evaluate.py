"""Evaluate, diagnose, or render a frozen research/legacy pursuit checkpoint."""
from __future__ import annotations

import argparse
import copyreg
import hashlib
import json
import math
import pickle
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np
import torch
from torch import nn

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, EpisodeOutcome, POLICE_COUNT, VehiclePlacement
from pursuit_evasion_rl.osm_demo.observations import ACTION_DIM
from pursuit_evasion_rl.osm_demo.policies import build_action_mask, decode_recommendation
from pursuit_evasion_rl.osm_demo.runner import run_single_episode
from ..errors import ResearchValidationError
from ..policies.baselines import EncirclementPolice, GoalEvader, _Graph
from ..policies.masked_mappo import ResearchMaskedMAPPO
from ..training import TRAINER_SCHEMA_VERSION
from ..variants.observations import OBSERVATION_28D_DIM, Observation28DAdapter

copyreg.pickle(MappingProxyType, lambda value: (MappingProxyType, (dict(value),)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_network(path: Path):
    if not path.is_file():
        raise ResearchValidationError("NETWORK_NOT_FOUND", "cached network file does not exist; no external fetch is attempted", path=str(path))
    return pickle.loads(path.read_bytes())


class _LegacyActor(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        weight_keys = sorted(
            (key for key in state if key.startswith("network.") and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        if not weight_keys:
            raise ResearchValidationError("INVALID_LEGACY_CHECKPOINT", "legacy actor has no linear weights")
        modules: list[nn.Module] = []
        for index, key in enumerate(weight_keys):
            weight = state[key]
            modules.append(nn.Linear(int(weight.shape[1]), int(weight.shape[0])))
            if index + 1 < len(weight_keys):
                modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules)
        self.load_state_dict(state)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class LearnedPolice:
    """Compatibility policy that reads both new research and legacy actor checkpoints."""

    def __init__(self, network, graph: _Graph, checkpoint: str | Path, clip: float = 2000.0, greedy: bool = True) -> None:
        self.network = network
        self.graph = graph
        self.greedy = greedy
        self.adapter = Observation28DAdapter(network, clip_distance_m=clip)
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise ResearchValidationError("CHECKPOINT_NOT_FOUND", "checkpoint does not exist", path=str(self.checkpoint))
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # older torch compatibility
            payload = torch.load(self.checkpoint, map_location="cpu")
        self._research_policy: ResearchMaskedMAPPO | None = None
        self._legacy_actor: nn.Module | None = None
        if payload.get("schema_version") == TRAINER_SCHEMA_VERSION:
            observation = payload.get("observation", {})
            self.adapter = Observation28DAdapter(
                network,
                clip_distance_m=float(observation.get("clip_distance_m", clip)),
                near_radius_m=float(observation.get("near_radius_m", 200.0)),
            )
            policy = ResearchMaskedMAPPO(
                actor_obs_dim=int(payload["actor_obs_dim"]),
                critic_context_dim=int(payload["critic_context_dim"]),
                action_dim=int(payload["action_dim"]),
                num_officers=int(payload["num_officers"]),
                hidden_dims=tuple(int(value) for value in payload["hidden_dims"]),
                device="cpu",
            )
            policy.load_state_dict(payload["model"])
            policy.actor.eval()
            self._research_policy = policy
        elif "police_actor" in payload:
            self._legacy_actor = _LegacyActor(payload["police_actor"])
            self._legacy_actor.eval()
        else:
            raise ResearchValidationError("UNSUPPORTED_CHECKPOINT", "checkpoint is neither research nor legacy MAPPO")
        self.profile = "research_masked_mappo" if self._research_policy else "legacy_osm_mappo"
        self.experimental = True
        self._parameter_hash = _sha256(self.checkpoint)

    @property
    def parameter_hash(self) -> str:
        return self._parameter_hash

    def _decision(self, placement: VehiclePlacement) -> int:
        if placement.segment_id is not None:
            return int(self.graph.seg[placement.segment_id].end_id)
        return int(placement.intersection_id)

    def recommend(self, *, police, fugitive, step, max_steps, incoming_headings=None):
        recommendations = []
        for officer_id in range(POLICE_COUNT):
            heading = None if incoming_headings is None else incoming_headings[officer_id]
            decision_id = self._decision(police[officer_id])
            mask, ordered = build_action_mask(self.network, decision_id, heading)
            observation = self.adapter.observe(
                police_index=officer_id,
                police=police,
                fugitive=fugitive,
                step=step,
                max_steps=max_steps,
                incoming_heading=heading,
            )
            with torch.no_grad():
                tensor = torch.from_numpy(observation).float()
                if self._research_policy is not None:
                    logits = self._research_policy.actor_logits(
                        tensor.unsqueeze(0), torch.tensor([officer_id], dtype=torch.long)
                    )
                else:
                    assert self._legacy_actor is not None
                    logits = self._legacy_actor(tensor.unsqueeze(0))
                mask_tensor = torch.from_numpy(np.asarray(mask, dtype=bool)).unsqueeze(0)
                logits = logits.masked_fill(~mask_tensor, float("-inf"))
                action = int(torch.argmax(logits, dim=-1).item())
            probabilities = np.zeros(ACTION_DIM)
            probabilities[action] = 1.0
            recommendations.append(decode_recommendation(
                agent_id=f"police_{officer_id}",
                network=self.network,
                decision_intersection_id=decision_id,
                mask=mask,
                ordered_segment_ids=ordered,
                probabilities=probabilities,
                profile=self.profile,
                compatibility_report_id=self.profile,
                action_index=action,
            ))
        return tuple(recommendations)


def surround_placement(graph: _Graph, rng: np.random.Generator, band: tuple[float, float] = (400.0, 950.0)):
    positions = graph.pos
    drivable = [identity for identity in positions if graph.fwd.get(identity)]
    center = (
        sum(positions[item][0] for item in drivable) / len(drivable),
        sum(positions[item][1] for item in drivable) / len(drivable),
    )
    pool = sorted(drivable, key=lambda item: math.dist(positions[item], center))[:20]
    fugitive = pool[int(rng.integers(0, len(pool)))]
    fugitive_xy = positions[fugitive]
    used = {fugitive}
    police = []
    for slot in range(POLICE_COUNT):
        target = math.radians(60 * slot + float(rng.uniform(-15, 15)))
        candidates = []
        for identity in drivable:
            if identity in used:
                continue
            dx = positions[identity][0] - fugitive_xy[0]
            dy = positions[identity][1] - fugitive_xy[1]
            distance = math.hypot(dx, dy)
            if band[0] <= distance <= band[1]:
                angle = math.atan2(dy, dx)
                difference = abs(math.atan2(math.sin(angle - target), math.cos(angle - target)))
                candidates.append((difference, identity))
        chosen = min(candidates)[1] if candidates else min(
            (identity for identity in drivable if identity not in used),
            key=lambda identity: math.dist(positions[identity], fugitive_xy),
        )
        used.add(chosen)
        police.append(VehiclePlacement(intersection_id=chosen))
    return police, VehiclePlacement(intersection_id=fugitive)


def _episode(network, graph, policy, config, *, seed: int, vision: float, placement_mode: str):
    placement = surround_placement(graph, np.random.default_rng(5000 + seed)) if placement_mode == "surround" else None
    return run_single_episode(
        network,
        config,
        policy,
        run_id=f"evaluation-{seed}",
        run_seed=10000 + seed,
        fugitive_factory=lambda rng: GoalEvader(network, rng=rng, graph=graph, vision_range_m=vision),
        placement=placement,
    )


def evaluate_checkpoint(network, checkpoint: Path, config: EpisodeConfig, *, episodes: int, vision: float, placement_mode: str) -> dict[str, object]:
    graph = _Graph(network)
    captures = 0
    steps: list[int] = []
    for seed in range(episodes):
        record = _episode(network, graph, LearnedPolice(network, graph, checkpoint), config, seed=seed, vision=vision, placement_mode=placement_mode)
        if record.outcome is EpisodeOutcome.CAPTURE:
            captures += 1
            steps.append(len(record.transitions))
    return {
        "episodes": episodes,
        "capture_rate": captures / episodes,
        "median_capture_steps": int(np.median(steps)) if steps else None,
        "checkpoint_sha256": _sha256(checkpoint),
        "selection_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("evaluate", "diagnose", "render"), default="evaluate")
    parser.add_argument("--net", type=Path, default=Path("out_pursuit/interior_net.pkl"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/osm_mappo/best.pt"))
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--police-speed", type=float, default=16.0)
    parser.add_argument("--fugitive-speed", type=float, default=9.0)
    parser.add_argument("--capture", type=float, default=25.0)
    parser.add_argument("--vision", type=float, default=150.0)
    parser.add_argument("--mode", choices=("surround", "random"), default="random")
    parser.add_argument("--output", type=Path, default=Path("out_zoom"))
    return parser


def _diagnose(network, checkpoint: Path, config: EpisodeConfig, vision: float, placement_mode: str) -> dict[str, object]:
    graph = _Graph(network)
    record = _episode(network, graph, LearnedPolice(network, graph, checkpoint), config, seed=0, vision=vision, placement_mode=placement_mode)
    states = [record.initial_state] + [transition.state for transition in record.transitions]
    fugitive = [placement_position(network, state.fugitive) for state in states]
    officers = [[placement_position(network, state.police[index]) for state in states] for index in range(POLICE_COUNT)]
    return {
        "outcome": record.outcome.value,
        "steps": len(record.transitions),
        "fugitive_path_m": sum(math.dist(a, b) for a, b in zip(fugitive, fugitive[1:])),
        "officers": [
            {
                "officer_id": index,
                "path_m": sum(math.dist(a, b) for a, b in zip(track, track[1:])),
                "start_distance_m": math.dist(track[0], fugitive[0]),
                "end_distance_m": math.dist(track[-1], fugitive[-1]),
            }
            for index, track in enumerate(officers)
        ],
    }


def _render(network, checkpoint: Path, config: EpisodeConfig, vision: float, placement_mode: str, output: Path) -> dict[str, object]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ResearchValidationError("RENDER_DEPENDENCY_MISSING", "matplotlib is required for render mode") from exc
    graph = _Graph(network)
    record = _episode(network, graph, LearnedPolice(network, graph, checkpoint), config, seed=0, vision=vision, placement_mode=placement_mode)
    states = [record.initial_state] + [transition.state for transition in record.transitions]
    output.mkdir(parents=True, exist_ok=True)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf")
    fugitive = [placement_position(network, state.fugitive) for state in states]
    tracks = [[placement_position(network, state.police[index]) for state in states] for index in range(POLICE_COUNT)]
    for step in range(len(states)):
        figure, axis = plt.subplots(figsize=(8, 8), dpi=100)
        for segment in network.segments:
            if not segment.virtual:
                axis.plot([point[0] for point in segment.geometry_xy], [point[1] for point in segment.geometry_xy], color="#d8dce0", linewidth=0.7)
        for officer_id, track in enumerate(tracks):
            partial = track[: step + 1]
            axis.plot([point[0] for point in partial], [point[1] for point in partial], color=colors[officer_id])
            axis.scatter(*partial[-1], color=colors[officer_id], label=f"P{officer_id}")
        partial_fugitive = fugitive[: step + 1]
        axis.plot([point[0] for point in partial_fugitive], [point[1] for point in partial_fugitive], "--", color="#d62728")
        axis.scatter(*partial_fugitive[-1], marker="*", s=150, color="#d62728", label="fugitive")
        axis.set_aspect("equal")
        axis.legend(loc="best", fontsize=7)
        axis.set_title(f"step {step}/{len(states)-1} outcome={record.outcome.value}")
        figure.savefig(output / f"frame_{step:03d}.png")
        plt.close(figure)
    return {"outcome": record.outcome.value, "frames": len(states), "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.seeds <= 0:
            raise ResearchValidationError("INVALID_EVALUATION_COUNT", "seeds must be positive")
        network = _load_network(args.net)
        config = EpisodeConfig(
            dt_s=1.0,
            police_speed_mps=args.police_speed,
            fugitive_speed_mps=args.fugitive_speed,
            capture_radius_m=args.capture,
            max_steps=args.max_steps,
        )
        if args.command == "diagnose":
            result = _diagnose(network, args.ckpt, config, args.vision, args.mode)
        elif args.command == "render":
            result = _render(network, args.ckpt, config, args.vision, args.mode, args.output)
        else:
            result = evaluate_checkpoint(network, args.ckpt, config, episodes=args.seeds, vision=args.vision, placement_mode=args.mode)
    except (ResearchValidationError, OSError, pickle.PickleError, RuntimeError, ValueError) as exc:
        payload = exc.as_dict() if isinstance(exc, ResearchValidationError) else {"code": "EVALUATION_ERROR", "message": str(exc)}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("LearnedPolice", "build_parser", "evaluate_checkpoint", "main", "surround_placement")
