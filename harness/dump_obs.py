"""Capture real observations from a live game -> ground-truth schema.

The docs only describe the obs dict at a high level. This records every obs an
agent actually receives during a real game so we can build features/heuristics
against the true field layout. Run once inside Docker, then read the JSON.

    python harness/dump_obs.py --out artifacts/schema
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ptcg.agents import get_agent  # noqa: E402
from ptcg.agents.base import make_safe  # noqa: E402
from ptcg.deck import load_deck  # noqa: E402


def _summarize_keys(obj, depth=0, max_depth=4):
    """Recursively describe the shape of an obs object (types + keys)."""
    if depth > max_depth:
        return "..."
    if isinstance(obj, dict):
        return {k: _summarize_keys(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        head = obj[0] if obj else None
        return [f"list[{len(obj)}]", _summarize_keys(head, depth + 1, max_depth)]
    return type(obj).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/schema")
    ap.add_argument("--a", default="random")
    ap.add_argument("--b", default="random")
    args = ap.parse_args()

    from kaggle_environments import make

    deck = load_deck()
    records: list[dict] = []

    def recorder(inner):
        def f(obs):
            snap = copy.deepcopy(obs)
            # search_begin_input is a large opaque blob; keep only its length.
            if isinstance(snap, dict) and isinstance(snap.get("search_begin_input"), str):
                snap["search_begin_input"] = f"<{len(snap['search_begin_input'])} chars>"
            records.append(snap)
            return inner(obs)

        return f

    a = recorder(make_safe(get_agent(args.a), deck))
    b = make_safe(get_agent(args.b), deck)

    env = make("cabt")
    env.run([a, b])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Full per-step observations seen by player A.
    (out / "observations.json").write_text(json.dumps(records, indent=2, default=str))

    # A compact schema description from the first non-deck-selection step.
    play_obs = next((r for r in records if r.get("select") is not None), records[-1])
    (out / "schema.json").write_text(
        json.dumps(_summarize_keys(play_obs), indent=2, default=str)
    )

    print(f"Recorded {len(records)} observations -> {out}/observations.json")
    print(f"Schema sketch -> {out}/schema.json")
    if play_obs.get("select"):
        print("First select.option example:")
        print(json.dumps(play_obs["select"]["option"][:3], indent=2, default=str))


if __name__ == "__main__":
    main()
