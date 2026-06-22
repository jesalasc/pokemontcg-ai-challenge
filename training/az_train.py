"""AlphaZero training loop: self-play -> train AZNet -> checkpoint -> repeat.

Runs in the ptcg-rl image (engine + torch). Strategy is learned end-to-end;
the only reward is the game outcome.

    # plumbing smoke (no torch needed for self-play; stub evaluator):
    python training/az_train.py --evaluator stub --iters 1 --games 1 --sims 4
    # real (net-guided self-play):
    python training/az_train.py --evaluator net  --iters 50 --games 16 --sims 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg import deckcode, features  # noqa: E402
from training import az_selfplay  # noqa: E402


def load_roster(path: str = "decks") -> list[tuple[str, list[int]]]:
    out = []
    for p in sorted(Path(path).glob("*.csv")):
        ids = [int(x) for x in p.read_text().split() if x.strip()]
        if len(ids) == 60:
            out.append((p.stem, ids))
    return out


def collate(samples: list[dict]):
    """Pad variable option counts into batched tensors (numpy)."""
    B = len(samples)
    maxN = max(s["options"].shape[0] for s in samples)
    S = np.stack([s["state"] for s in samples])
    D = np.stack([s["deck"] for s in samples])
    O = np.zeros((B, maxN, features.OPTION_DIM), np.float32)
    M = np.zeros((B, maxN), bool)
    PI = np.zeros((B, maxN), np.float32)
    Z = np.array([s["z"] for s in samples], np.float32)
    for i, s in enumerate(samples):
        n = s["options"].shape[0]
        O[i, :n] = s["options"]
        M[i, :n] = True
        PI[i, :n] = s["pi"]
    return S, D, O, M, PI, Z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", choices=["stub", "net"], default="net")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default="checkpoints/az.pt")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from training.az_mcts import NetEvaluator, stub_evaluator
    from training.networks import AZNet

    roster = load_roster()
    if not roster:
        sys.exit("no 60-card decks in decks/ — add some first")
    print(f"roster: {[n for n, _ in roster]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AZNet(features.STATE_DIM, features.OPTION_DIM, deckcode.pool_size()).to(device)
    if args.init and Path(args.init).is_file():
        net.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"warm-started from {args.init}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    for it in range(args.iters):
        # ---- self-play ----
        net.eval()
        evaluator = stub_evaluator if args.evaluator == "stub" else NetEvaluator(net, device)
        samples, wins = [], 0
        for _ in range(args.games):
            i, j = rng.integers(0, len(roster)), rng.integers(0, len(roster))
            s, z0 = az_selfplay.play_game(evaluator, roster[i][1], roster[j][1],
                                          sims=args.sims, temperature=args.temperature)
            samples += s
            wins += int(z0 == 1)
        if not samples:
            print(f"iter {it+1}: no samples (engine available?)")
            continue

        # ---- train ----
        net.train()
        S, D, O, M, PI, Z = (torch.from_numpy(x).to(device) for x in collate(samples))
        n = len(Z)
        for _ep in range(args.epochs):
            perm = torch.randperm(n, device=device)
            ploss = vloss = 0.0
            for b in range(0, n, args.batch):
                idx = perm[b:b + args.batch]
                logits, value = net(S[idx], D[idx], O[idx], M[idx])
                logp = torch.log_softmax(logits.masked_fill(~M[idx], float("-inf")), dim=-1)
                pol = -(PI[idx] * torch.nan_to_num(logp)).sum(dim=-1).mean()
                val = F.mse_loss(value, Z[idx])
                loss = pol + val
                opt.zero_grad(); loss.backward(); opt.step()
                ploss += pol.item() * len(idx); vloss += val.item() * len(idx)
        print(f"iter {it+1}/{args.iters}  samples={n}  decisive={wins}/{args.games}  "
              f"policy_loss={ploss/n:.3f}  value_loss={vloss/n:.3f}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(),
                    "dims": [features.STATE_DIM, features.OPTION_DIM, deckcode.pool_size()]},
                   args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
