"""Self-play PPO refinement (skeleton).

Refines the (optionally distilled) OptionScorer by playing itself and rewarding
wins. The control flow is complete; the LEVERS that decide whether RL converges
are intentionally left to the RL engineer + domain expert:

  * REWARD SHAPING (shape_reward below): terminal +1/-1 is sparse over 20-100
    turns. Add domain-informed intermediate rewards — prize taken, KO dealt/
    avoided, energy efficiency. This is THE human lever; tune it here.
  * Opponent pool: self-play vs a frozen snapshot / past versions to avoid
    cycling. Plug a sampler into `opponent_policy`.
  * Advantage: GAE vs Monte-Carlo return; entropy bonus; KL-to-distilled anchor.

Run (after a distill warm-start) on the GPU box:
    python training/ppo.py --init checkpoints/policy.pt --iters 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg import features  # noqa: E402
from training import selfplay  # noqa: E402

# --- reward shaping: THE human lever (default = sparse terminal only) -------
GAMMA = 0.997
CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


def shape_reward(traj_step: dict, terminal_reward: float, is_last: bool) -> float:
    """Per-step reward. TODO(domain expert): add intermediate signals.

    Default: assign the terminal win/loss to the last step only.
    """
    return terminal_reward if is_last else 0.0


def _make_policy_step(model, device):
    """Wrap the net into selfplay's policy_step: obs -> (idx, logp, value)."""
    import torch

    def step(obs):
        s = torch.from_numpy(features.encode_state(obs)).float().unsqueeze(0).to(device)
        o = torch.from_numpy(features.encode_options(obs)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(s, o)              # [1, N]
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(model.value(s).item())

    return step


def train(init: str | None, iters: int, games_per_iter: int, lr: float, out: str) -> None:
    import torch
    import torch.nn.functional as F

    from training.networks import OptionScorer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OptionScorer(features.STATE_DIM, features.OPTION_DIM).to(device)
    if init and Path(init).is_file():
        model.load_state_dict(torch.load(init, map_location=device)["model"])
        print(f"warm-started from {init}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for it in range(iters):
        # 1) Collect self-play episodes.
        batch = []
        wins = 0
        for g in range(games_per_iter):
            traj, rewards = selfplay.play_policy_episode(_make_policy_step(model, device))
            wins += int(rewards[0] == 1)
            for seat in (0, 1):
                steps = traj[seat]
                term = rewards[seat]
                ret = 0.0
                for t in reversed(range(len(steps))):
                    r = shape_reward(steps[t], term, is_last=(t == len(steps) - 1))
                    ret = r + GAMMA * ret
                    steps[t]["return"] = ret
                batch.extend(steps)
        if not batch:
            continue

        # 2) PPO update (single epoch shown; extend to K epochs + minibatches).
        S = torch.from_numpy(_stack(batch, "state")).float().to(device)
        returns = torch.tensor([b["return"] for b in batch], device=device)
        old_logp = torch.tensor([b["logp"] for b in batch], device=device)
        values = model.value(S)
        adv = (returns - values.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Per-step recompute of action logprob (variable option counts -> loop).
        new_logp, entropy = _recompute_logp(model, batch, device)
        ratio = torch.exp(new_logp - old_logp)
        pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
        vloss = F.mse_loss(values, returns)
        loss = pg + VALUE_COEF * vloss - ENTROPY_COEF * entropy.mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if (it + 1) % 10 == 0:
            print(f"iter {it+1}/{iters}  winrate(seat0)={wins/games_per_iter:.2f}  loss={loss.item():.3f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
    print(f"saved -> {out}")


def _stack(batch, key):
    import numpy as np
    return np.stack([b[key] for b in batch])


def _recompute_logp(model, batch, device):
    """Recompute log-prob + entropy per sample (option counts vary)."""
    import torch

    logps, ents = [], []
    for b in batch:
        s = torch.from_numpy(b["state"]).float().unsqueeze(0).to(device)
        o = torch.from_numpy(b["options"]).float().unsqueeze(0).to(device)
        dist = torch.distributions.Categorical(logits=model(s, o))
        logps.append(dist.log_prob(torch.tensor([b["action"]], device=device)))
        ents.append(dist.entropy())
    return torch.cat(logps), torch.cat(ents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/policy.pt")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints/policy_ppo.pt")
    args = ap.parse_args()
    train(args.init, args.iters, args.games, args.lr, args.out)


if __name__ == "__main__":
    main()
