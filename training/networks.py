"""Option-scoring policy/value network (PyTorch, pluggable).

Scores each legal option conditioned on the global state, so it natively handles
cabt's variable action set:  score_i = head([state_emb ; option_i_emb]).
A value head on the state embedding supports PPO / actor-critic.

Swap this module out (JAX, bigger nets, attention over options) without touching
the agents — they only depend on encode_state/encode_options shapes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out),
    )


class OptionScorer(nn.Module):
    def __init__(self, state_dim: int, option_dim: int, hidden: int = 128):
        super().__init__()
        self.state_enc = _mlp(state_dim, hidden, hidden)
        self.opt_enc = _mlp(option_dim, hidden, hidden)
        self.score_head = nn.Linear(2 * hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self,
        state: torch.Tensor,        # [B, STATE_DIM]
        options: torch.Tensor,      # [B, N, OPTION_DIM]
        mask: torch.Tensor | None = None,  # [B, N] bool, True = legal
    ) -> torch.Tensor:              # -> [B, N] logits over options
        s = self.state_enc(state)                       # [B, H]
        o = self.opt_enc(options)                       # [B, N, H]
        s_exp = s.unsqueeze(1).expand(-1, o.size(1), -1)
        scores = self.score_head(torch.cat([s_exp, o], dim=-1)).squeeze(-1)  # [B, N]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores

    def value(self, state: torch.Tensor) -> torch.Tensor:  # [B]
        return self.value_head(self.state_enc(state)).squeeze(-1)
