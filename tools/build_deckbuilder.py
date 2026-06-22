"""Generate tools/deckbuilder.html from the template + cached card DB.

Pure-stdlib, runs on macOS (uses src/ptcg/_data/*.json). Re-run after the card
pool changes. The server calls this automatically if the page is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "ptcg" / "_data"
TEMPLATE = ROOT / "tools" / "deckbuilder.template.html"
OUT = ROOT / "tools" / "deckbuilder.html"

ENERGY = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P",
          6: "F", 7: "D", 8: "M", 9: "N", 10: "*", 11: "TR"}


def build_cards_json() -> str:
    cards = json.loads((DATA / "cards.json").read_text())
    atk = {int(a["attackId"]): a for a in json.loads((DATA / "attacks.json").read_text())}
    out = []
    for c in cards:
        stage = "B" if c.get("basic") else "1" if c.get("stage1") else "2" if c.get("stage2") else ""
        rec = {"id": c["cardId"], "n": c["name"], "t": c["cardType"]}
        if c["cardType"] == 0:
            rec.update(hp=c["hp"], rc=c.get("retreatCost", 0),
                       et=ENERGY.get(c.get("energyType"), "?"), s=stage)
            if c.get("ex"):
                rec["x"] = 1
            if c.get("megaEx"):
                rec["m"] = 1
            if c.get("evolvesFrom"):
                rec["ev"] = c["evolvesFrom"]
            ats = [[atk[a]["name"], atk[a]["damage"]] for a in c.get("attacks", []) if a in atk]
            if ats:
                rec["a"] = ats
        else:
            if c["cardType"] in (5, 6):
                rec["et"] = ENERGY.get(c.get("energyType"), "?")
            if c.get("aceSpec"):
                rec["ace"] = 1
        out.append(rec)
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False)


def main() -> None:
    if not (DATA / "cards.json").is_file():
        raise SystemExit("card DB not cached — run `make fetch-engine` then "
                         "`python -m ptcg.cards` in Docker first.")
    tpl = TEMPLATE.read_text()
    data = build_cards_json()
    assert tpl.count("__CARDS_JSON__") == 1 and "</script" not in data.lower()
    OUT.write_text(tpl.replace("__CARDS_JSON__", data))
    print(f"built {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
