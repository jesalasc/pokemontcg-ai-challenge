"""Human-readable rendering of engine observations and options.

Turns the raw obs/option dicts into text a player can read, so you can pilot games
(demonstration capture) and review replays. Uses the verified enums + card DB.
"""
from __future__ import annotations

from ptcg import cards as C
from ptcg import engine_codes as E


def _players(obs: dict) -> list:
    return (obs.get("current") or {}).get("players") or []


def _card(obs: dict, player: int | None, area, index) -> dict | None:
    if player is None or area is None or index is None:
        return None
    ps = _players(obs)
    if not (0 <= player < len(ps)):
        return None
    zone = E.ZONE_NAME.get(area)
    cards = ps[player].get(zone) if zone else None
    if isinstance(cards, list) and 0 <= index < len(cards) and isinstance(cards[index], dict):
        return cards[index]
    return None


def _name(obs: dict, player, area, index) -> str:
    c = _card(obs, player, area, index)
    return C.name(c["id"]) if c else f"{E.ZONE_NAME.get(area, area)}[{index}]"


def describe_option(o: dict, obs: dict) -> str:
    me = (obs.get("current") or {}).get("yourIndex")
    t = o.get("type")
    OT = E.OptionType
    if E.is_attack(o):
        a = C.attack(o.get("attackId"))
        return f"Attack — {a['name']} ({a['damage']})" if a else "Attack"
    if t == OT.END:
        return "End turn"
    if t == OT.RETREAT:
        return "Retreat active"
    if t == OT.PLAY:
        return f"Play {_name(obs, me, E.AreaType.HAND, o.get('index'))}"
    if t == OT.ATTACH:
        what = _name(obs, me, o.get("area"), o.get("index"))
        onto = _name(obs, o.get("playerIndex", me), o.get("inPlayArea"), o.get("inPlayIndex"))
        return f"Attach {what} → {onto}"
    if t == OT.EVOLVE:
        evo = _name(obs, me, o.get("area"), o.get("index"))
        tgt = _name(obs, o.get("playerIndex", me), o.get("inPlayArea"), o.get("inPlayIndex"))
        return f"Evolve {tgt} → {evo}"
    if t == OT.ABILITY:
        return f"Ability — {_name(obs, o.get('playerIndex', me), o.get('area'), o.get('index'))}"
    if t == OT.DISCARD:
        return f"Discard {_name(obs, o.get('playerIndex', me), o.get('area'), o.get('index'))}"
    if t in (OT.CARD, OT.ENERGY_CARD, OT.TOOL_CARD, OT.ENERGY):
        pidx = o.get("playerIndex")
        whose = "" if pidx is None or pidx == me else "opp "
        nm = _name(obs, pidx, o.get("area"), o.get("index"))
        cnt = f" x{o['count']}" if o.get("count") else ""
        return f"Select {whose}{nm}{cnt}"
    if t == OT.NUMBER:
        return f"Choose {o.get('number')}"
    if t == OT.YES:
        return "Yes"
    if t == OT.NO:
        return "No"
    return f"Option(type={t})"


def _pokemon(obs: dict, player: int, zone: str) -> list[str]:
    out = []
    for c in (_players(obs)[player].get(zone) or []):
        if isinstance(c, dict):
            nm = C.name(c.get("id"))
            if "hp" in c:
                nm += f" ({c['hp']}/{c.get('maxHp', '?')})"
                e = c.get("energies") or c.get("energyCards") or []
                if e:
                    nm += f" [{len(e)}E]"
            out.append(nm)
    return out


def player_summary(obs: dict, player: int, hide_hand: bool = False) -> dict:
    ps = _players(obs)
    if player >= len(ps):
        return {}
    p = ps[player]
    active = _pokemon(obs, player, "active")
    hand = p.get("hand")
    return {
        "active": active[0] if active else "—",
        "bench": _pokemon(obs, player, "bench"),
        "prizes": len(p.get("prize") or []),
        "deck": p.get("deckCount", 0),
        "hand": ([C.name(c.get("id")) for c in hand] if isinstance(hand, list) and not hide_hand
                 else p.get("handCount", 0)),
        "status": [s for s in ("poisoned", "burned", "asleep", "paralyzed", "confused") if p.get(s)],
    }


def context_label(obs: dict) -> str:
    sel = obs.get("select") or {}
    try:
        return E.SelectContext(sel.get("context")).name
    except Exception:
        return str(sel.get("context"))
