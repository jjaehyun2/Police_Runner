"""Attention-based target assignment (PEPCRL-MVP "progression cognition" analogue).

Two-level structure, as in Li et al. (IEEE T-ITS, arXiv 2306.05016):

  upper level  every DECISION_PERIOD seconds each officer picks a *target*
               (pursue the fugitive, or guard one of the fugitive's escape
               routes) with an attention network that sees the whole team and
               every candidate target at once.
  lower level  shortest-path movement to the chosen target (see macro.py).

Network (parameter-shared over officers, decentralized execution with radio):

  officer tokens   O [6, DO]  -> MLP -> self-attention over teammates
  target tokens    C [N, DC] + pair features of the querying officer P [N, DP]
                              -> MLP -> cross-attention to the team
  score(k, j)      MLP([h_k, t_j, h_k * t_j])          (masked softmax over j)

Officers still decide in order of distance; tokens carry who has already
committed to what this round, so later officers react to earlier choices.
The attention weights of the officer -> target layer are kept for the
case-study figures.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pursuit_evasion_rl.osm_demo.metrics import placement_position

import macro
from macro import CLIP_M, N_CANDIDATES, POLICE_COUNT

DO = 4 + 2 + 2 + POLICE_COUNT        # officer token
DC = 9 + 6                           # target token (officer-independent)
DP = 4                               # (querying officer, target) pair features
D = 64


class AttnAssigner(nn.Module):
    def __init__(self, d: int = D, heads: int = 4) -> None:
        super().__init__()
        self.off_in = nn.Sequential(nn.Linear(DO, d), nn.ReLU(), nn.Linear(d, d))
        self.tgt_in = nn.Sequential(nn.Linear(DC + DP, d), nn.ReLU(), nn.Linear(d, d))
        self.team_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.tgt_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.last_attn = None

    def forward(self, O, C, P, k):
        """O [B,6,DO], C [B,N,DC], P [B,N,DP], k [B] -> logits [B,N]."""
        h = self.off_in(O)
        h = self.ln1(h + self.team_attn(h, h, h, need_weights=False)[0])
        hk = h[torch.arange(len(k)), k]                                   # [B,d]
        t = self.tgt_in(torch.cat([C, P], -1))
        ctx, w = self.tgt_attn(t, h, h)                                   # targets look at the team
        t = self.ln2(t + ctx)
        self.last_attn = w.detach()
        q = hk[:, None, :].expand_as(t)
        return self.head(torch.cat([q, t, q * t], -1)).squeeze(-1)


class AttnTeamPolicy(macro.MacroTeamPolicy):
    profile = "attn_role_policy"

    def _tokens(self, police, fxy, road, rank, decided, chosen, rows_k, k):
        O = np.zeros((POLICE_COUNT, DO), dtype=np.float32)
        for j in range(POLICE_COUNT):
            x, y = placement_position(self.network, police[j])
            r = road[j]
            O[j, :4] = [min(r / CLIP_M, 1.0) if np.isfinite(r) else 1.0, rank[j] / 5.0,
                        float(np.isfinite(r)), float(j == k)]
            O[j, 4:6] = [float(decided[j]), float(decided[j] and chosen[j] == 0)]
            O[j, 6:8] = np.clip([(x - fxy[0]) / CLIP_M, (y - fxy[1]) / CLIP_M], -1, 1)
            O[j, 8 + j] = 1.0
        C = np.zeros((N_CANDIDATES, DC), dtype=np.float32)
        P = np.zeros((N_CANDIDATES, DP), dtype=np.float32)
        # officer-independent target features from the querying officer's rows
        C[:, :3] = rows_k[:, :3]                     # pursuit flag, fugitive time/dist, open
        C[:, 3] = rows_k[:, 6]                       # holders (others)
        C[:, 4] = rows_k[:, 8]                       # heading alignment
        C[:, 5] = rows_k[:, 9]                       # intercept node distance
        C[:, 6] = np.maximum(rows_k[:, 3], rows_k[:, 5])   # best team margin
        C[:, 7] = [sum(1 for j in range(POLICE_COUNT) if decided[j] and chosen[j] == c) / 3.0
                   for c in range(N_CANDIDATES)]      # committed this round
        C[:, 8] = 1.0
        C[:, 9:] = rows_k[:, 10:16]
        P[:, 0] = rows_k[:, 3]                       # my margin
        P[:, 1] = rows_k[:, 4]                       # my travel time
        P[:, 2] = rows_k[:, 5]                       # best teammate margin
        P[:, 3] = rows_k[:, 7]                       # held by me last round
        return O, C, P

    def _decision_round(self, police, fugitive, step, max_steps, heading, fxy):
        cands, road, order, rank, n_open = self._round_setup(police, fugitive)
        decided = [False] * POLICE_COUNT
        chosen = [0] * POLICE_COUNT
        for k in order:
            rows, mask, intercepts = self._rows(k, cands, heading, fxy, road, rank, step, max_steps, n_open)
            O, C, P = self._tokens(police, fxy, road, rank, decided, chosen, rows, k)
            forced = self._forced(k, step, cands)
            if forced is not None and mask[forced]:
                a, logp = forced, 0.0
            elif self.teacher is not None:          # distillation: an MLP role scorer drives
                with torch.no_grad():
                    tl = self.teacher(torch.from_numpy(rows)[None], torch.tensor([k]))[0].numpy()
                a, logp = int(np.argmax(np.where(mask, tl, -np.inf))), 0.0
            elif self.fixed_rule is not None:
                a, logp = self._choose(k, rows, mask)
            else:
                with torch.no_grad():
                    logits = self.scorer(torch.from_numpy(O)[None], torch.from_numpy(C)[None],
                                         torch.from_numpy(P)[None], torch.tensor([k]))[0].numpy()
                z = np.where(mask, logits / self.temperature, -np.inf)
                z = z - z.max()
                p = np.exp(z)
                p /= p.sum()
                a = int(np.argmax(p)) if self.mode == "argmax" else int(self.rng.choice(N_CANDIDATES, p=p))
                logp = float(np.log(max(p[a], 1e-12)))
            targets = [None] + [c[2] for c in cands[5]] + [None] * (N_CANDIDATES - 1 - len(cands[5]))
            self.log.append(dict(officer=k, O=O, C=C, P=P, rows=rows, mask=mask, action=a, logp=logp, step=step,
                                 targets=targets))
            decided[k], chosen[k] = True, a
            self.roles[k] = a
            self.targets[k] = intercepts[a]
            self.role_exit[k] = cands[5][a - 1][2] if a > 0 else None

    temperature = 1.0
    teacher = None
    override: dict = {}          # {officer: (step_from, step_to, exit node | None=pursue)} probes

    def _forced(self, k, step, cands):
        """Counterfactual probe: officer k is committed to one target (an exit route,
        or pursuit) for a window of decision rounds; None outside the window."""
        o = self.override.get(k)
        if o is None or not (o[0] <= step < o[1]):
            return None
        if o[2] is None:
            return 0
        for c, cand in enumerate(cands[5], start=1):
            if cand[2] == o[2]:
                return c
        return None
