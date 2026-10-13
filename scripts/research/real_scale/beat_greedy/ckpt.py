"""Checkpoint I/O in the same payload format ``load_frozen_policy`` reads."""
from __future__ import annotations

from pathlib import Path

import torch

import os
from obs_ext import EXT2_OBS_DIM, EXT2_PROFILE_ID, EXT_OBS_DIM as _V1_DIM, EXT_PROFILE_ID as _V1_PROFILE

if os.environ.get("BG_OBS", "2") == "1":
    EXT_OBS_DIM, EXT_PROFILE_ID = _V1_DIM, _V1_PROFILE
else:
    EXT_OBS_DIM, EXT_PROFILE_ID = EXT2_OBS_DIM, EXT2_PROFILE_ID
from pursuit_evasion_rl.research.policies.masked_mappo import CentralizedCritic, SharedOfficerActor

POLICE_COUNT = 6
ACTION_DIM = 6


def new_actor(hidden=(256, 256)) -> SharedOfficerActor:
    return SharedOfficerActor(EXT_OBS_DIM, ACTION_DIM, POLICE_COUNT, hidden)


def new_critic(hidden=(256, 256)) -> CentralizedCritic:
    return CentralizedCritic(EXT_OBS_DIM * POLICE_COUNT, hidden)


def save(path: Path, actor, critic, *, hidden, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "beat-greedy-v1",
        "actor_obs_dim": EXT_OBS_DIM,
        "critic_context_dim": EXT_OBS_DIM * POLICE_COUNT,
        "action_dim": ACTION_DIM,
        "num_officers": POLICE_COUNT,
        "hidden_dims": tuple(hidden),
        "observation": {"profile": EXT_PROFILE_ID, "clip_distance_m": 2000.0, "near_radius_m": 200.0},
        "model": {"actor": actor.state_dict(), "critic": critic.state_dict()},
        "meta": meta,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = tuple(payload["hidden_dims"])
    dim = payload["actor_obs_dim"]
    actor = SharedOfficerActor(dim, ACTION_DIM, POLICE_COUNT, hidden)
    critic = CentralizedCritic(dim * POLICE_COUNT, hidden)
    actor.load_state_dict(payload["model"]["actor"])
    critic.load_state_dict(payload["model"]["critic"])
    return actor, critic, payload
