"""Render a single match to an interactive HTML replay (+ JSON trajectory).

The cabt env ships a visualizer; ``env.render(mode="html")`` produces a
self-contained page. Use this to actually *watch* what the agent does — the
fastest way for the domain expert to spot bad lines of play.

    python harness/replay.py --a rule_based --b random --out artifacts/replays
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness.run_match import run_match  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="rule_based")
    ap.add_argument("--b", default="random")
    ap.add_argument("--out", default="artifacts/replays")
    args = ap.parse_args()

    result, env = run_match(args.a, args.b, return_env=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    base = out / f"{args.a}_vs_{args.b}_{stamp}"

    html = env.render(mode="html")
    base.with_suffix(".html").write_text(html, encoding="utf-8")
    base.with_suffix(".json").write_text(json.dumps(env.toJSON(), default=str))

    print(f"result: {result}")
    print(f"replay -> {base.with_suffix('.html')}")
    print(f"trajectory -> {base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
