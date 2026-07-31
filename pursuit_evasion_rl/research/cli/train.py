"""Train exact stored-mask centralized MAPPO using train/validation data only."""
from __future__ import annotations

import argparse
import copyreg
import json
import math
import pickle
from pathlib import Path
import sys
from types import MappingProxyType

from ..errors import ResearchValidationError
from ..maps import DataHandle, HandleKind, SplitScope, TuningDataView
from ..training import DEFAULT_OUTPUT_ROOT, ResearchTrainer, TrainerConfig

copyreg.pickle(MappingProxyType, lambda value: (MappingProxyType, (dict(value),)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-net", "--net", dest="train_net", type=Path, default=Path("out_pursuit/interior_net.pkl"))
    parser.add_argument("--validation-net", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--condition-id", default="masked-mappo-28d")
    parser.add_argument("--training-seed", type=int, default=20240931)
    parser.add_argument("--run-id")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--update-every", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--police-speed", type=float, default=16.0)
    parser.add_argument("--fugitive-speed", type=float, default=9.0)
    parser.add_argument("--capture", type=float, default=25.0)
    parser.add_argument("--vision", type=float, default=150.0)
    parser.add_argument("--clip", type=float, default=2000.0)
    parser.add_argument("--near-radius", type=float, default=200.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--capture-bonus", type=float, default=20.0)
    parser.add_argument("--time-penalty", type=float, default=0.02)
    parser.add_argument("--dist-scale", type=float, default=50.0)
    parser.add_argument("--team-coef", type=float, default=1.0)
    parser.add_argument("--own-coef", type=float, default=0.6)
    parser.add_argument("--regress-mult", type=float, default=1.6)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument("--ckpt-every", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument("--start-ep", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--resume", default="", help="accepted for migration; full resume arrives via verified ResumeState")
    return parser


def _load_network(path: Path):
    if not path.is_file():
        raise ResearchValidationError("NETWORK_NOT_FOUND", "cached network file does not exist; no external fetch is attempted", path=str(path))
    return pickle.loads(path.read_bytes())


def _config(args: argparse.Namespace) -> TrainerConfig:
    if args.resume:
        raise ResearchValidationError(
            "INCOMPLETE_RESUME_STATE",
            "legacy checkpoints are evaluation-compatible but cannot resume the research trainer",
            path="resume",
            actual=args.resume,
        )
    updates = args.updates if args.updates is not None else max(1, math.ceil(args.episodes / args.update_every))
    episodes_per_update = args.update_every if args.updates is None else max(1, args.update_every)
    return TrainerConfig(
        updates=updates,
        episodes_per_update=episodes_per_update,
        validation_episodes=args.validation_episodes,
        max_steps=args.max_steps,
        police_speed_mps=args.police_speed,
        fugitive_speed_mps=args.fugitive_speed,
        capture_radius_m=args.capture,
        vision_range_m=args.vision,
        clip_distance_m=args.clip,
        near_radius_m=args.near_radius,
        learning_rate=args.lr,
        gamma=args.gamma,
        capture_bonus=args.capture_bonus,
        time_penalty=args.time_penalty,
        distance_scale_m=args.dist_scale,
        team_coefficient=args.team_coef,
        own_coefficient=args.own_coef,
        regress_multiplier=args.regress_mult,
        entropy_coefficient=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validation_path = args.validation_net or args.train_net
        train_network = _load_network(args.train_net)
        validation_network = _load_network(validation_path)
        tuning = TuningDataView(
            train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, str(args.train_net)),),
            validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, str(validation_path)),),
        )
        result = ResearchTrainer(
            train_network=train_network,
            validation_network=validation_network,
            tuning_data=tuning,
            config=_config(args),
            condition_id=args.condition_id,
            training_seed=args.training_seed,
            output_root=args.output_root,
            run_id=args.run_id,
        ).train()
    except (ResearchValidationError, OSError, pickle.PickleError) as exc:
        payload = exc.as_dict() if isinstance(exc, ResearchValidationError) else {"code": "TRAINING_IO_ERROR", "message": str(exc)}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "run_id": result.run_id,
        "run_directory": str(result.run_directory),
        "checkpoint": str(result.checkpoint_path),
        "updates": len(result.history),
        "all_actions_legal": result.all_actions_legal,
        "selection_source": "validation",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
