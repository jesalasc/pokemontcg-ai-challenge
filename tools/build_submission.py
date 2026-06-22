"""Package a submission .tar.gz with main.py at the archive root.

The ladder runtime provides the cabt engine, so we only ship our code:
    main.py, deck.csv, agent.txt, src/ptcg/**
For an RL submission (--agent rl) we also bundle the network + checkpoint.

    python tools/build_submission.py --agent rule_based
    python tools/build_submission.py --agent rl --checkpoint checkpoints/policy.pt
    python tools/build_submission.py --agent rule_based --smoke   # verify it imports
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _add_tree(tar: tarfile.TarFile, root: Path, arcprefix: str, suffixes=(".py", ".json")) -> None:
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if suffixes and p.suffix not in suffixes:
            continue
        tar.add(p, arcname=f"{arcprefix}/{p.relative_to(root)}")


def build(agent: str, checkpoint: str | None, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"submission_{agent}_{int(time.time())}.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        # root files
        tar.add(_ROOT / "main.py", arcname="main.py")
        tar.add(_ROOT / "deck.csv", arcname="deck.csv")

        # agent selector (ladder can't set env vars)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(agent)
            agent_txt = f.name
        tar.add(agent_txt, arcname="agent.txt")

        # the package (.py + bundled card-DB JSON under _data/)
        _add_tree(tar, _ROOT / "src" / "ptcg", "src/ptcg")

        if agent in ("mcts", "dragapult", "az"):
            # these import cg.api -> bundle the official engine (incl. libcg.so)
            cg_dir = _ROOT / "data" / "engine" / "cg"
            if not cg_dir.is_dir():
                raise FileNotFoundError("data/engine/cg missing — run `make fetch-engine`")
            _add_tree(tar, cg_dir, "cg", suffixes=None)

        if agent == "rl":
            tar.add(_ROOT / "training" / "__init__.py", arcname="training/__init__.py")
            tar.add(_ROOT / "training" / "networks.py", arcname="training/networks.py")
            ckpt = Path(checkpoint or "checkpoints/policy.pt")
            if not ckpt.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            tar.add(ckpt, arcname="checkpoints/policy.pt")

        if agent == "az":
            # NOTE: the AZ agent also needs torch at run time — confirm the ladder
            # runtime provides torch before relying on this submission.
            for m in ("__init__.py", "networks.py", "az_mcts.py"):
                tar.add(_ROOT / "training" / m, arcname=f"training/{m}")
            ckpt = Path(checkpoint or "checkpoints/az.pt")
            if not ckpt.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            tar.add(ckpt, arcname="checkpoints/az.pt")

    size_mb = archive.stat().st_size / 1e6
    print(f"built {archive}  ({size_mb:.2f} MB, agent={agent})")
    return archive


def smoke(archive: Path) -> None:
    """Unpack to a temp dir and import main.py to catch packaging bugs."""
    import subprocess

    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(archive) as tar:
            tar.extractall(d)
        code = (
            "import sys; sys.path.insert(0, 'src'); import main; "
            "print('deck ok' if len(main.agent({'select': None})) == 60 else 'deck BAD'); "
            "print('play ok', main.agent({'select': {'option':[{'type':1}], 'maxCount':1}, "
            "'current': {'yourIndex':0,'result':-1,'players':[]}, 'logs':[]}))"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True, text=True)
        print("--- smoke ---")
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            raise SystemExit("smoke test FAILED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="rule_based",
                    choices=["random", "rule_based", "mcts", "rl"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="submissions")
    ap.add_argument("--smoke", action="store_true", help="verify the archive imports")
    args = ap.parse_args()

    archive = build(args.agent, args.checkpoint, args.out)
    if args.smoke:
        smoke(archive)
    print("\nNext: gate it with harness/evaluate.py, then submit with tools/submit.py")


if __name__ == "__main__":
    main()
