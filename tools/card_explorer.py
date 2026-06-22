"""Explore the legal card pool for deck construction.

Runs off the cached card DB (src/ptcg/_data/) — no engine needed, works on macOS.
For the domain expert to design decks from the actual ~1267 legal cards.

    python tools/card_explorer.py --type pokemon --ex --min-hp 200
    python tools/card_explorer.py --name dragapult
    python tools/card_explorer.py --id 121            # full detail + attacks
    python tools/card_explorer.py --stats             # pool composition
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptcg import cards as C  # noqa: E402

TYPE = {0: "Pokemon", 1: "Item", 2: "Tool", 3: "Supporter", 4: "Stadium",
        5: "BasicEnergy", 6: "SpecialEnergy"}
TYPE_ALIAS = {v.lower(): k for k, v in TYPE.items()}
TYPE_ALIAS.update({"energy": -1})  # -1 -> both energy types


def _all_cards() -> list[dict]:
    return list(C._cards().values())


def _detail(cid: int) -> None:
    c = C.card(cid)
    if not c:
        print(f"no card {cid}")
        return
    print(f"#{c['cardId']}  {c['name']}  [{TYPE.get(c['cardType'])}]")
    if c["cardType"] == 0:
        flags = [k for k in ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec") if c.get(k)]
        print(f"  HP {c['hp']}  retreat {c['retreatCost']}  prize_value {C.prize_value(cid)}  {' '.join(flags)}")
        if c.get("evolvesFrom"):
            print(f"  evolves from: {c['evolvesFrom']}")
        for aid in c.get("attacks", []):
            a = C.attack(aid)
            if a:
                print(f"    attack: {a['name']:24s} dmg {a['damage']:>4}  cost {a.get('energies')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", help="pokemon|item|supporter|tool|stadium|energy")
    ap.add_argument("--name", help="case-insensitive substring")
    ap.add_argument("--ex", action="store_true", help="only ex / megaEx")
    ap.add_argument("--evolves-from")
    ap.add_argument("--min-hp", type=int, default=0)
    ap.add_argument("--id", type=int)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    if not C.is_loaded():
        sys.exit("card DB not cached — run `make fetch-engine` then `python -m ptcg.cards` in Docker")

    if args.id is not None:
        return _detail(args.id)

    if args.stats:
        from collections import Counter
        c = Counter(TYPE.get(x["cardType"], "?") for x in _all_cards())
        print(f"legal pool: {len(_all_cards())} cards")
        for k, v in c.most_common():
            print(f"  {v:5d}  {k}")
        return

    rows = _all_cards()
    if args.type:
        t = TYPE_ALIAS.get(args.type.lower())
        rows = [c for c in rows if (c["cardType"] in (5, 6) if t == -1 else c["cardType"] == t)]
    if args.name:
        rows = [c for c in rows if args.name.lower() in c["name"].lower()]
    if args.ex:
        rows = [c for c in rows if c.get("ex") or c.get("megaEx")]
    if args.evolves_from:
        rows = [c for c in rows if (c.get("evolvesFrom") or "").lower() == args.evolves_from.lower()]
    rows = [c for c in rows if c.get("hp", 0) >= args.min_hp]
    rows.sort(key=lambda c: (-c.get("hp", 0), c["name"]))

    print(f"{len(rows)} cards (showing {min(len(rows), args.limit)}):")
    for c in rows[: args.limit]:
        tag = "ex" if c.get("ex") else ("MEGA" if c.get("megaEx") else "")
        hp = f"HP{c['hp']}" if c["cardType"] == 0 else ""
        print(f"  #{c['cardId']:<5} {c['name']:<34} {TYPE.get(c['cardType']):<12} {hp:<6} {tag}")


if __name__ == "__main__":
    main()
