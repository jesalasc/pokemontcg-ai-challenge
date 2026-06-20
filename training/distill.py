"""Behavioral cloning: distill a teacher (rule_based / MCTS) into the net.

This is the fast warm-start the brief calls "Scripted Policy Distillation": let
the strong scripted policy generate decisions, then train the OptionScorer to
imitate them. The result is a NN policy you can then refine with PPO self-play.

    python training/distill.py --teacher rule_based --games 100 --epochs 10
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
from ptcg.agents import get_agent  # noqa: E402
from training import selfplay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="rule_based")
    ap.add_argument("--opponent", default="random")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/policy.pt")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from training.networks import OptionScorer

    print(f"Collecting {args.games} teacher games ({args.teacher} vs {args.opponent})...")
    samples = selfplay.record_teacher_games(
        get_agent(args.teacher), get_agent(args.opponent), args.games
    )
    print(f"  {len(samples)} decision samples")
    states, options, mask, targets = selfplay.to_arrays(samples)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = OptionScorer(features.STATE_DIM, features.OPTION_DIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    S = torch.from_numpy(states).to(dev)
    O = torch.from_numpy(options).to(dev)
    M = torch.from_numpy(mask).to(dev)
    T = torch.from_numpy(targets).to(dev)

    n = len(samples)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        total = 0.0
        for i in range(0, n, args.batch):
            b = perm[i : i + args.batch]
            logits = model(S[b], O[b], M[b])           # [B, N]
            loss = F.cross_entropy(logits, T[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        # train accuracy = how often argmax matches the teacher
        with torch.no_grad():
            acc = (model(S, O, M).argmax(1) == T).float().mean().item()
        print(f"epoch {ep + 1}/{args.epochs}  loss={total / n:.4f}  acc={acc:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"saved -> {args.out}  (run agent with PTCG_AGENT=rl)")


if __name__ == "__main__":
    main()
