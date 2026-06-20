"""Self-play data collection over the cabt engine.

Two collectors:
  * record_teacher_games  -> (state, options, target_idx) for imitation/distill
  * play_policy_episode    -> on-policy trajectory for PPO (see ppo.py)

Everything runs the Linux engine, so use Docker / the GPU box.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg import features  # noqa: E402
from ptcg import protocol as P  # noqa: E402
from ptcg.agents.base import make_safe  # noqa: E402
from ptcg.deck import load_deck  # noqa: E402

PlayFn = Callable[[dict], list]


def record_teacher_games(
    teacher: PlayFn,
    opponent: PlayFn,
    n_games: int = 50,
    deck=None,
) -> list[dict]:
    """Record the teacher's single-select decisions for behavioral cloning.

    Returns samples: {"state": [STATE_DIM], "options": [N, OPTION_DIM], "target": int}
    (Only maxCount==1 steps are recorded; multi-select BC is handled separately.)
    """
    from kaggle_environments import make

    deck = deck if deck is not None else load_deck()
    samples: list[dict] = []

    def recorder(obs):
        choice = teacher(obs)
        sel = obs.get("select")
        if sel and int(sel.get("maxCount", 1)) == 1 and choice:
            samples.append(
                {
                    "state": features.encode_state(obs),
                    "options": features.encode_options(obs),
                    "target": int(choice[0]),
                }
            )
        return choice

    for i in range(n_games):
        a = make_safe(recorder, deck)
        b = make_safe(opponent, deck)
        env = make("cabt")
        env.run([a, b] if i % 2 == 0 else [b, a])

    return samples


def play_policy_episode(policy_step: Callable[[dict], tuple[int, float, float]], deck=None):
    """Run one self-play game where BOTH seats use the same learnable policy.

    ``policy_step(obs) -> (chosen_index, log_prob, value)``. Returns per-seat
    trajectories and the terminal rewards, for PPO. Reward shaping is applied by
    the caller (ppo.py) — keep this collector policy-agnostic.
    """
    from kaggle_environments import make

    deck = deck if deck is not None else load_deck()
    traj = {0: [], 1: []}

    def make_seat(seat: int):
        def fn(obs):
            if P.is_deck_selection(obs):
                return list(deck)
            sel = obs["select"]
            n = len(sel["option"])
            k = min(int(sel.get("maxCount", 1)), n)
            try:
                idx, logp, val = policy_step(obs)
                traj[seat].append(
                    {
                        "state": features.encode_state(obs),
                        "options": features.encode_options(obs),
                        "action": idx,
                        "logp": logp,
                        "value": val,
                    }
                )
                return [idx] if k == 1 else _fill(idx, k, n)
            except Exception:
                return list(range(k))

        return fn

    env = make("cabt")
    env.run([make_seat(0), make_seat(1)])
    rewards = [env.state[0]["reward"], env.state[1]["reward"]]
    return traj, rewards


def _fill(idx: int, k: int, n: int) -> list[int]:
    out = [idx] + [i for i in range(n) if i != idx]
    return out[:k]


def to_arrays(samples: list[dict]):
    """Pad variable-length option lists into a batch with a legality mask."""
    if not samples:
        return None
    max_n = max(s["options"].shape[0] for s in samples)
    B = len(samples)
    states = np.stack([s["state"] for s in samples])
    options = np.zeros((B, max_n, features.OPTION_DIM), np.float32)
    mask = np.zeros((B, max_n), bool)
    targets = np.array([s["target"] for s in samples], np.int64)
    for i, s in enumerate(samples):
        n = s["options"].shape[0]
        options[i, :n] = s["options"]
        mask[i, :n] = True
    return states, options, mask, targets
