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


class AZNet(nn.Module):
    """Deck-conditioned AlphaZero net: (state, deck, options) -> (policy, value).

    The deck vector (counts over the legal pool) conditions one network to pilot
    any deck in the roster. Policy is an option-scorer over the dynamic action set;
    value is the learned position evaluation (replaces all heuristics), in [-1, 1]
    from the to-move player's perspective.
    """

    def __init__(self, state_dim: int, option_dim: int, pool_dim: int,
                 hidden: int = 128, deck_hidden: int = 64):
        super().__init__()
        self.deck_enc = _mlp(pool_dim, deck_hidden, deck_hidden)
        self.state_enc = _mlp(state_dim + deck_hidden, hidden, hidden)
        self.opt_enc = _mlp(option_dim, hidden, hidden)
        self.policy_head = nn.Linear(2 * hidden, 1)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state, deck, options, mask=None):
        # state [B,S]  deck [B,P]  options [B,N,O]  mask [B,N] bool
        d = self.deck_enc(deck)
        s = self.state_enc(torch.cat([state, d], dim=-1))         # [B,H]
        o = self.opt_enc(options)                                  # [B,N,H]
        s_exp = s.unsqueeze(1).expand(-1, o.size(1), -1)
        logits = self.policy_head(torch.cat([s_exp, o], dim=-1)).squeeze(-1)  # [B,N]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        value = torch.tanh(self.value_head(s)).squeeze(-1)        # [B]
        return logits, value
