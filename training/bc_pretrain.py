"""Behavioral-clone the AZ net from your captured demonstrations.

Turns games you piloted (data/demos/, via `make play`) into a warm-start
checkpoint: the policy learns to imitate your moves, the value learns your game
outcomes. Then refine with self-play: `make az-train ARGS="--init checkpoints/az.pt"`.
This is how your knowledge enters the net — as data, not hand-coded strategy.

    python training/bc_pretrain.py --epochs 20            # all demos
    python training/bc_pretrain.py --deck crustle         # one deck's demos
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg import deckcode, features  # noqa: E402
from ptcg.deckcode import deck_vector  # noqa: E402


def load_demos(deck: str | None) -> list[dict]:
    root = _ROOT / "data" / "demos"
    folders = [root / deck] if deck else [root]
    samples = []
    for folder in folders:
        for f in folder.rglob("*.json"):
            d = json.loads(f.read_text())
            dv = deck_vector(d["deck"])
            z = float(d.get("z", 0.0))
            for s in d["samples"]:
                opts = np.asarray(s["options"], dtype=np.float32).reshape(-1, features.OPTION_DIM)
                n = len(opts)
                if n == 0:
                    continue
                pi = np.zeros(n, dtype=np.float32)
                for a in s["action"]:
                    if 0 <= a < n:
                        pi[a] = 1.0
                if pi.sum() == 0:
                    continue
                pi /= pi.sum()
                samples.append({"state": np.asarray(s["state"], np.float32),
                                "deck": dv, "options": opts, "pi": pi, "z": z})
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=None, help="only this deck's demos (folder name)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/az.pt")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from training.az_train import collate
    from training.networks import AZNet

    samples = load_demos(args.deck)
    if not samples:
        sys.exit("no demos found in data/demos/ — capture some with `make play` first")
    print(f"{len(samples)} demonstration decisions")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = AZNet(features.STATE_DIM, features.OPTION_DIM, deckcode.pool_size()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    S, D, O, M, PI, Z = (torch.from_numpy(x).to(dev) for x in collate(samples))
    n = len(Z)

    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot_p = tot_v = correct = 0.0
        for b in range(0, n, args.batch):
            idx = perm[b:b + args.batch]
            logits, value = net(S[idx], D[idx], O[idx], M[idx])
            logp = torch.log_softmax(logits.masked_fill(~M[idx], float("-inf")), dim=-1)
            pol = -(PI[idx] * torch.nan_to_num(logp)).sum(-1).mean()
            val = F.mse_loss(value, Z[idx])
            (pol + val).backward(); opt.step(); opt.zero_grad()
            tot_p += pol.item() * len(idx); tot_v += val.item() * len(idx)
        with torch.no_grad():
            pred = net(S, D, O, M)[0].masked_fill(~M, float("-inf")).argmax(1)
            correct = (pred == PI.argmax(1)).float().mean().item()
        print(f"epoch {ep+1}/{args.epochs}  policy={tot_p/n:.3f}  value={tot_v/n:.3f}  match={correct:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(),
                "dims": [features.STATE_DIM, features.OPTION_DIM, deckcode.pool_size()]}, args.out)
    print(f"saved -> {args.out}  (now: make az-train ARGS=\"--init {args.out}\")")


if __name__ == "__main__":
    main()
