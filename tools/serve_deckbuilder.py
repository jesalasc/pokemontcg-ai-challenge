"""Serve the deck builder locally and manage named decks.

Stdlib only — no Docker, no deps. Runs on macOS directly (uses the cached card DB).

    python tools/serve_deckbuilder.py        # http://localhost:8000
    python tools/serve_deckbuilder.py 8080    # custom port

Decks are saved by NAME under decks/<name>.csv (never clobbered by a different
name). Saving also sets that deck as the ACTIVE deck (deck.csv) — the one the
agent plays. So no deck is ever lost behind a single filename.

  GET  /                 -> the deck-builder page
  GET  /decks            -> list saved decks [{name, count, legal}]
  GET  /deck?name=NAME    -> {ids:[...]} for a saved deck
  POST /save             -> {name, ids, overwrite?} -> writes decks/<name>.csv
                             (+ activates deck.csv); refuses silent overwrite
"""
from __future__ import annotations

import http.server
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

HTML = ROOT / "tools" / "deckbuilder.html"
DECK = ROOT / "deck.csv"          # the active deck (what the agent plays)
DECKS = ROOT / "decks"            # named, preserved decks
DECKS.mkdir(exist_ok=True)

SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_name(raw: str) -> str:
    return SAFE.sub("-", (raw or "").strip()).strip("-")[:60]


def list_decks() -> list[dict]:
    import deck_check
    out = []
    for p in sorted(DECKS.glob("*.csv")):
        ids = [int(x) for x in p.read_text().split() if x.strip()]
        ok, _ = deck_check.validate(ids)
        out.append({"name": p.stem, "count": len(ids), "legal": ok})
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            if not HTML.is_file():
                return self._send(500, "deckbuilder.html not built", "text/plain")
            return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if route.path == "/decks":
            return self._send(200, json.dumps({"decks": list_decks()}))
        if route.path == "/deck":
            name = safe_name(parse_qs(route.query).get("name", [""])[0])
            p = DECKS / f"{name}.csv"
            if not p.is_file():
                return self._send(404, json.dumps({"ok": False, "message": "not found"}))
            ids = [int(x) for x in p.read_text().split() if x.strip()]
            return self._send(200, json.dumps({"ok": True, "ids": ids}))
        self._send(404, "not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/save":
            return self._send(404, json.dumps({"ok": False, "message": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            ids = [int(x) for x in data.get("ids", [])]
            name = safe_name(data.get("name", ""))
            overwrite = bool(data.get("overwrite"))
        except Exception as e:  # noqa: BLE001
            return self._send(400, json.dumps({"ok": False, "message": f"bad request: {e}"}))

        if not name:
            return self._send(200, json.dumps({"ok": False, "message": "Name your deck first."}))

        import deck_check
        path = DECKS / f"{name}.csv"
        if path.is_file() and not overwrite:
            return self._send(200, json.dumps(
                {"ok": False, "exists": True,
                 "message": f"'{name}' already exists — Save again to overwrite it."}))

        body = "\n".join(str(i) for i in ids) + "\n"
        path.write_text(body)        # named, preserved
        DECK.write_text(body)         # activate: the agent now plays this deck
        ok, problems = deck_check.validate(ids)
        msg = (f"Saved decks/{name}.csv & set active (deck.csv) — "
               + ("LEGAL ✓" if ok else "INVALID: " + "; ".join(problems)))
        self._send(200, json.dumps({"ok": ok, "message": msg, "problems": problems,
                                    "decks": list_decks()}))

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    if not HTML.is_file():
        import build_deckbuilder
        build_deckbuilder.main()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"Deck Forge -> http://localhost:{port}   (Ctrl-C to stop)")
    print(f"  saved decks: {DECKS}   active deck: {DECK}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
