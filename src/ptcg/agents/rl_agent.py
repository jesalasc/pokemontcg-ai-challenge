"""Layer 3 — learned option-scoring policy.

Loads a trained network (see training/) and scores each legal option, then picks
the best. If torch or the checkpoint is unavailable (e.g. on the submission box
before we ship weights), it falls back to the rule_based agent — so "rl" is
always a safe, runnable choice.

The network is an option scorer: score_i = f(state_features, option_i_features).
This handles cabt's dynamic action set and makes distillation from rule_based /
MCTS straightforward (regress toward their chosen option).
"""
from __future__ import annotations

import os
from functools import lru_cache

from ptcg import features
from ptcg import protocol as P
from ptcg.agents import rule_based

CHECKPOINT_ENV = "PTCG_RL_CHECKPOINT"
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "policy.pt")


@lru_cache(maxsize=1)
def _load_policy():
    """Return a callable scorer(state_vec, option_mat)->scores, or None."""
    path = os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)
    if not os.path.isfile(path):
        return None
    try:
        import torch

        from training.networks import OptionScorer  # lazy: torch only here

        ckpt = torch.load(path, map_location="cpu")
        model = OptionScorer(features.STATE_DIM, features.OPTION_DIM)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        model.eval()

        def scorer(state_vec, option_mat):
            with torch.no_grad():
                s = torch.from_numpy(state_vec).float().unsqueeze(0)
                o = torch.from_numpy(option_mat).float().unsqueeze(0)
                return model(s, o).squeeze(0).numpy()

        return scorer
    except Exception:
        return None


def play(obs: dict) -> list[int]:
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    k = min(int(sel.get("maxCount", 1)), len(opts))
    if k <= 0:
        return []

    scorer = _load_policy()
    if scorer is None:
        return rule_based.play(obs)  # no weights yet -> strong baseline

    try:
        state_vec = features.encode_state(obs)
        option_mat = features.encode_options(obs)
        scores = scorer(state_vec, option_mat)
        ranked = sorted(range(len(opts)), key=lambda i: float(scores[i]), reverse=True)
        return ranked[:k]
    except Exception:
        return rule_based.play(obs)
