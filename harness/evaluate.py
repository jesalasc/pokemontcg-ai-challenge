"""The submission gate: measure agent A vs agent B over many games.

Alternates seats (first/second player) to cancel first-move advantage, reports
win-rate with a Wilson 95% CI, the implied Elo gap, and average game length, and
writes a JSON report. Nothing should be submitted to the ladder unless it clears
a meaningful margin here.

    python harness/evaluate.py --a rule_based --b random -n 100
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

from harness import elo  # noqa: E402
from harness.run_match import run_match  # noqa: E402


def evaluate(agent_a: str, agent_b: str, n: int = 100, verbose: bool = True) -> dict:
    wins = draws = losses = errors = 0
    total_steps = 0

    for i in range(n):
        # Alternate seats: even games A is player 0, odd games A is player 1.
        a_is_first = (i % 2 == 0)
        first, second = (agent_a, agent_b) if a_is_first else (agent_b, agent_a)
        res = run_match(first, second)
        total_steps += res["steps"]
        if res["error"]:
            errors += 1
        winner = res["winner"]
        # Translate the seat winner back to "did A win?"
        if winner is None:
            draws += 1
        else:
            a_won = (winner == 0) if a_is_first else (winner == 1)
            wins += int(a_won)
            losses += int(not a_won)
        if verbose and (i + 1) % max(1, n // 10) == 0:
            print(f"  {i + 1}/{n}: A win-rate so far {wins / (i + 1):.3f}")

    score = (wins + 0.5 * draws) / n if n else 0.0
    lo, hi = elo.wilson_interval(wins + 0.5 * draws, n)
    report = {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "errors": errors,
        "score": round(score, 4),
        "winrate_ci95": [round(lo, 4), round(hi, 4)],
        "elo_gap_est": round(elo.elo_diff_from_score(score), 1),
        "avg_steps": round(total_steps / n, 1) if n else 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate agent A vs B (the submission gate).")
    ap.add_argument("--a", default="rule_based", help="challenger agent")
    ap.add_argument("--b", default="random", help="baseline agent")
    ap.add_argument("-n", "--games", type=int, default=100)
    ap.add_argument("--out", default="artifacts/eval")
    args = ap.parse_args()

    rep = evaluate(args.a, args.b, args.games)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{args.a}_vs_{args.b}_{int(time.time())}.json"
    fname.write_text(json.dumps(rep, indent=2))

    print("\n=== EVAL REPORT ===")
    for k, v in rep.items():
        print(f"{k:>16}: {v}")
    print(f"\nsaved -> {fname}")
    # Convenience verdict against a 55% gate (tune per matchup).
    gate = 0.55
    verdict = "PASS" if rep["score"] >= gate and rep["winrate_ci95"][0] > 0.5 else "HOLD"
    print(f"gate(>= {gate} & CI>0.5): {verdict}")


if __name__ == "__main__":
    main()
