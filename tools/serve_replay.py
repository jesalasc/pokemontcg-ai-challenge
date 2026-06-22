"""Watch recorded replays as a readable, step-through move log.

Host-side, stdlib only (no engine) — reads artifacts/replays/*.json produced by
harness/record_replay.py. Scrub through every decision to spot bad lines, then
feed fixes back via demonstrations / curriculum (docs/TRAINING.md, step 4).

    make replay-view    # -> http://localhost:8001
"""
from __future__ import annotations

import http.server
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
REPLAYS = ROOT / "artifacts" / "replays"


def _list():
    if not REPLAYS.is_dir():
        return []
    out = []
    for p in sorted(REPLAYS.glob("*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
            out.append({"file": p.name, "a": d.get("a"), "b": d.get("b"),
                        "winner": d.get("winner"), "n": d.get("n_steps")})
        except Exception:
            pass
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        route = urlparse(self.path)
        if route.path in ("/", "/index.html"):
            return self._send(200, _UI, "text/html; charset=utf-8")
        if route.path == "/replays":
            return self._send(200, json.dumps({"replays": _list()}))
        if route.path == "/replay":
            name = Path(parse_qs(route.query).get("file", [""])[0]).name
            p = REPLAYS / name
            if not p.is_file():
                return self._send(404, json.dumps({"error": "not found"}))
            return self._send(200, p.read_text())
        self._send(404, "not found", "text/plain")

    def log_message(self, *a):
        pass


_UI = """<!doctype html><meta charset=utf-8><title>PTCG Replay</title>
<style>
body{background:#14171F;color:#EAEDF2;font:14px system-ui;margin:0;padding:18px;max-width:1000px;margin:auto}
h1{font-size:20px;letter-spacing:-.02em} .gold{color:#F6C945} .muted{color:#8B93A7}
select,button{font:14px system-ui;border-radius:8px;border:1px solid #2E3445;background:#1B1F2A;color:#EAEDF2;padding:8px;cursor:pointer}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.layout{display:grid;grid-template-columns:1fr 280px;gap:14px}
.board{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0}
.pane{background:#1B1F2A;border:1px solid #2E3445;border-radius:10px;padding:10px}
.pane h3{margin:0 0 6px;font-size:12px;color:#8B93A7;text-transform:uppercase}
.row{font-size:13px;margin:2px 0}
#ctx{color:#F6C945;font-weight:600;margin:6px 0}
.opt{padding:3px 6px;border-radius:6px;margin:2px 0;font-size:13px;border-left:3px solid #2E3445}
.opt.chosen{background:#223;border-color:#F6C945;font-weight:600}
.log{background:#1B1F2A;border:1px solid #2E3445;border-radius:10px;padding:8px;max-height:70vh;overflow:auto}
.li{font-size:12px;padding:3px 5px;border-radius:5px;cursor:pointer;border-left:3px solid transparent}
.li:hover{background:#222}.li.cur{background:#223;border-color:#F6C945}
.p0{color:#2DD4BF}.p1{color:#E8857A}
input[type=range]{width:260px}
</style>
<h1>PTCG <span class=gold>Replay</span> review</h1>
<div class=bar>
  <select id=file></select><button onclick=load()>Load</button>
  <span id=meta class=muted></span>
</div>
<div class=bar>
  <button onclick=step(-1)>◀ Prev</button>
  <input type=range id=slider min=0 value=0 oninput=goto(+this.value)>
  <button onclick=step(1)>Next ▶</button>
  <span id=pos class=muted></span>
</div>
<div class=layout>
  <div>
    <div id=ctx></div>
    <div class=board><div class=pane><h3 id=h0>Player 0</h3><div id=p0></div></div>
      <div class=pane><h3 id=h1>Player 1</h3><div id=p1></div></div></div>
    <div class=pane><h3>Options (chosen highlighted)</h3><div id=opts></div></div>
  </div>
  <div class=log id=log></div>
</div>
<script>
let R=null, idx=0;
const $=s=>document.querySelector(s);
async function init(){
  const d=await (await fetch('/replays')).json();
  $('#file').innerHTML=d.replays.map(r=>`<option value="${r.file}">${r.a} vs ${r.b} · ${r.n} steps</option>`).join('')
    || '<option>(no replays — run make replay-rec)</option>';
  if(d.replays.length) load();
}
async function load(){
  const f=$('#file').value; if(!f)return;
  R=await (await fetch('/replay?file='+encodeURIComponent(f))).json();
  $('#meta').textContent='winner: '+(R.winner==null?'draw':R.names[R.winner]);
  $('#h0').textContent=R.names['0']; $('#h1').textContent=R.names['1'];
  $('#slider').max=R.steps.length-1; idx=0;
  $('#log').innerHTML=R.steps.map((s,i)=>`<div class=li id=li${i} onclick=goto(${i})>
    <span class=p${s.player}>${s.n}. P${s.player}</span> ${esc(s.chosen_text||'(skip)')}</div>`).join('');
  goto(0);
}
function pane(p){return `<div class=row><b>Active:</b> ${p.active}</div>
  <div class=row><b>Bench:</b> ${(p.bench||[]).join(', ')||'—'}</div>
  <div class=row class=muted>Prizes left: ${p.prizes} · Deck: ${p.deck}</div>
  <div class=row><b>Hand:</b> ${Array.isArray(p.hand)?p.hand.join(', '):p.hand+' cards'}</div>
  ${(p.status||[]).length?`<div class=row style=color:#E8503A>${p.status.join(', ')}</div>`:''}`;}
function esc(s){return (s||'').replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
function goto(i){
  if(!R)return; idx=Math.max(0,Math.min(R.steps.length-1,i)); const s=R.steps[idx];
  $('#slider').value=idx; $('#pos').textContent=`step ${idx+1}/${R.steps.length}`;
  $('#ctx').textContent=`P${s.player} decision — ${s.context} (turn ${s.turn})`;
  $('#p0').innerHTML=pane(s.player===0?s.you:s.opp);
  $('#p1').innerHTML=pane(s.player===1?s.you:s.opp);
  $('#opts').innerHTML=s.options.map(o=>`<div class="opt ${s.chosen.includes(o.i)?'chosen':''}">${esc(o.text)}</div>`).join('');
  document.querySelectorAll('.li').forEach(e=>e.classList.remove('cur'));
  const li=$('#li'+idx); if(li){li.classList.add('cur');li.scrollIntoView({block:'nearest'});}
}
function step(d){goto(idx+d);}
document.onkeydown=e=>{if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1);};
init();
</script>"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print(f"PTCG Replay -> http://localhost:{port}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
