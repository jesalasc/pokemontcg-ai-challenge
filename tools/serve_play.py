"""Interactive play + demonstration capture.

You pilot a deck move-by-move against an agent; every decision you make is recorded
as a demonstration (data/demos/). Those demos BC-pretrain the AZ net
(training/bc_pretrain.py) — this is how your domain knowledge enters the learned
system without hand-coding strategy (see docs/TRAINING.md).

Runs INSIDE the engine container (needs cg + data/engine on the path):
    make play            # -> http://localhost:8000
"""
from __future__ import annotations

import http.server
import json
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "data" / "engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg.sim import Battle  # noqa: E402
from ptcg import describe, features  # noqa: E402
from ptcg.agents import available, get_agent  # noqa: E402
from ptcg.agents.base import make_safe  # noqa: E402

DEMOS = ROOT / "data" / "demos"
DEMOS.mkdir(parents=True, exist_ok=True)

S: dict = {"active": False, "deck": None, "opp": None, "opp_agent": None,
           "human": 0, "samples": [], "name": "deck"}


def _decks() -> dict[str, list[int]]:
    out = {}
    for p in sorted((ROOT / "decks").glob("*.csv")):
        ids = [int(x) for x in p.read_text().split() if x.strip()]
        if len(ids) == 60:
            out[p.stem] = ids
    return out


def _advance() -> None:
    """Auto-play opponent turns until it's the human's decision or the game ends."""
    while True:
        cur = Battle.obs["current"]
        if cur["result"] >= 0 or cur["yourIndex"] == S["human"]:
            return
        battle_select(S["opp_agent"](Battle.obs))


def _save_demo() -> str:
    r = Battle.obs["current"]["result"]
    z = 1.0 if r == S["human"] else (-1.0 if r in (0, 1) else 0.0)
    demo = {"deck": S["deck"], "opp_deck": S["opp"], "result": r, "z": z,
            "samples": S["samples"]}
    sub = DEMOS / S["name"]
    sub.mkdir(parents=True, exist_ok=True)
    f = sub / f"{int(time.time())}_{len(S['samples'])}moves.json"
    f.write_text(json.dumps(demo))
    S["active"] = False
    return str(f.relative_to(ROOT))


def _state() -> dict:
    obs = Battle.obs
    cur = obs["current"]
    if cur["result"] >= 0:
        who = "draw" if cur["result"] not in (0, 1) else ("You win!" if cur["result"] == S["human"] else "You lose.")
        return {"over": True, "message": who}
    sel = obs["select"]
    return {
        "over": False,
        "turn": cur.get("turn"),
        "context": describe.context_label(obs),
        "maxCount": sel.get("maxCount", 1),
        "minCount": sel.get("minCount", 1),
        "you": describe.player_summary(obs, S["human"], hide_hand=False),
        "opp": describe.player_summary(obs, 1 - S["human"], hide_hand=True),
        "options": [{"i": i, "text": describe.describe_option(o, obs)}
                    for i, o in enumerate(sel.get("option", []))],
    }


def new_game(human_name: str, opp_name: str, opp_agent: str) -> None:
    decks = _decks()
    if S["active"]:
        try:
            battle_finish()
        except Exception:
            pass
    S.update(deck=decks[human_name], opp=decks[opp_name], name=human_name,
             human=0, samples=[], active=True,
             opp_agent=make_safe(get_agent(opp_agent), decks[opp_name]))
    battle_start(S["deck"], S["opp"])
    _advance()


def apply_move(indices: list[int]) -> dict:
    obs = Battle.obs
    S["samples"].append({
        "state": features.encode_state(obs).tolist(),
        "options": features.encode_options(obs).tolist(),
        "action": [int(i) for i in indices],
    })
    battle_select([int(i) for i in indices])
    _advance()
    out = _state()
    if Battle.obs["current"]["result"] >= 0:
        out["saved"] = _save_demo()
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            return self._send(200, _UI, "text/html; charset=utf-8")
        if route == "/decks":
            return self._send(200, json.dumps({"decks": list(_decks()), "agents": available()}))
        if route == "/state":
            return self._send(200, json.dumps(_state() if S["active"] or Battle.obs else {"over": True, "message": "no game"}))
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/new":
                new_game(data["human"], data["opp"], data.get("opp_agent", "random"))
                return self._send(200, json.dumps(_state()))
            if self.path == "/move":
                return self._send(200, json.dumps(apply_move(data.get("indices", []))))
        except Exception as e:  # noqa: BLE001
            return self._send(200, json.dumps({"error": repr(e)}))
        self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *a):
        pass


_UI = """<!doctype html><meta charset=utf-8><title>PTCG Play</title>
<style>
body{background:#14171F;color:#EAEDF2;font:14px system-ui;margin:0;padding:18px;max-width:900px;margin:auto}
h1{font-size:20px;letter-spacing:-.02em} .gold{color:#F6C945}
select,button{font:14px system-ui;border-radius:8px;border:1px solid #2E3445;background:#1B1F2A;color:#EAEDF2;padding:8px}
button{cursor:pointer} .opt{display:block;width:100%;text-align:left;margin:4px 0;border-left:3px solid #2DD4BF}
.opt.sel{background:#223; border-color:#F6C945}
.board{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}
.pane{background:#1B1F2A;border:1px solid #2E3445;border-radius:10px;padding:10px}
.pane h3{margin:0 0 6px;font-size:12px;color:#8B93A7;text-transform:uppercase}
.row{font-size:13px;margin:2px 0} .muted{color:#8B93A7}
#ctx{color:#F6C945;font-weight:600;margin:8px 0}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
</style>
<h1>PTCG <span class=gold>Play</span> &amp; capture</h1>
<div class=bar>
  You: <select id=you></select> vs <select id=opp></select>
  opponent: <select id=agent></select>
  <button onclick=newGame()>New game</button>
  <span id=status class=muted></span>
</div>
<div id=ctx></div>
<div class=board>
  <div class=pane><h3>You</h3><div id=mine></div></div>
  <div class=pane><h3>Opponent</h3><div id=theirs></div></div>
</div>
<div id=opts></div>
<button id=confirm style=display:none onclick=confirmMulti()>Confirm selection</button>
<script>
let sel=[], maxc=1;
const $=s=>document.querySelector(s);
async function init(){
  const d=await (await fetch('/decks')).json();
  const opt=a=>a.map(x=>`<option>${x}</option>`).join('');
  $('#you').innerHTML=opt(d.decks); $('#opp').innerHTML=opt(d.decks); $('#agent').innerHTML=opt(d.agents);
}
async function newGame(){
  sel=[]; const r=await (await fetch('/new',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({human:$('#you').value,opp:$('#opp').value,opp_agent:$('#agent').value})})).json();
  render(r);
}
function pane(p){
  if(!p||!p.active&&!p.hand) {}
  const hand=Array.isArray(p.hand)?p.hand.join(', '):p.hand+' cards';
  return `<div class=row><b>Active:</b> ${p.active}</div>
   <div class=row><b>Bench:</b> ${(p.bench||[]).join(', ')||'—'}</div>
   <div class=row class=muted>Prizes left: ${p.prizes} · Deck: ${p.deck}</div>
   <div class=row><b>Hand:</b> ${hand}</div>
   ${(p.status||[]).length?`<div class=row style=color:#E8503A>${p.status.join(', ')}</div>`:''}`;
}
function render(r){
  if(r.error){$('#status').textContent=r.error;return;}
  if(r.over){$('#ctx').textContent=r.message+(r.saved?'  (demo saved: '+r.saved+')':'');
    $('#opts').innerHTML='';$('#confirm').style.display='none';return;}
  maxc=r.maxCount; sel=[];
  $('#status').textContent='turn '+r.turn;
  $('#ctx').textContent='Your decision: '+r.context+(maxc>1?` (pick ${r.minCount}-${maxc})`:'');
  $('#mine').innerHTML=pane(r.you); $('#theirs').innerHTML=pane(r.opp);
  $('#opts').innerHTML=r.options.map(o=>`<button class=opt data-i=${o.i} onclick=pick(${o.i})>${o.text}</button>`).join('');
  $('#confirm').style.display = maxc>1?'inline-block':'none';
}
async function pick(i){
  if(maxc===1){ return move([i]); }
  const b=document.querySelector(`[data-i="${i}"]`);
  if(sel.includes(i)){sel=sel.filter(x=>x!==i);b.classList.remove('sel');}
  else if(sel.length<maxc){sel.push(i);b.classList.add('sel');}
}
async function confirmMulti(){ if(sel.length) move(sel); }
async function move(indices){
  const r=await (await fetch('/move',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({indices})})).json();
  render(r);
}
init();
</script>"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = http.server.HTTPServer(("0.0.0.0", port), Handler)
    print(f"PTCG Play -> http://localhost:{port}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
