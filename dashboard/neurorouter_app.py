#!/usr/bin/env python3
"""
NEURO ROUTER: standalone desktop/browser app (no Streamlit)
===========================================================
Run:  python dashboard/neurorouter_app.py        (optional native window: pip install pywebview)

Python runs the simulation and serves a small local JSON API on 127.0.0.1; one HTML page draws
everything on canvas.

What is real and what is decorative (also stated in the UI):
  REAL        node queue load, link up/down state and utilisation, KPIs, telemetry charts, the heat-map
              and the alert feed. All of it is read from the recorded simulation history.
  DECORATIVE  the small moving "comet" particles and the sound effects. Their density follows link
              utilisation, but they are NOT traces of individual packets or of actual routes.
The simulation runs to completion first; the animation is a replay of what was recorded.

All numbers come from dashboard_core, shared with the Streamlit dashboard.

LIVE Q-LEARNING BRAIN panel (live_panel.py + routing_rl/simulation/live.py):
  A separate, controllable session of the SAME simulator with the REAL tabular Q-learning agent. It is
  stepped in a background thread; the HTTP handlers only read snapshots or apply validated controls
  (start/pause/step/reset, traffic, link outage via the simulator's event system, exploration, alpha/gamma).
  Endpoints: POST /api/live/start, POST /api/live/control, GET /api/live/state, GET /api/live/qtable.
"""
from __future__ import annotations

import atexit
import json
import re
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from dashboard_core import (
    LIMITS, MAX_PACKETS, METHOD_NAMES, METHODS, RUN_LOCK, TOPOLOGY_CATALOG, TRAFFIC_TYPES, RECOVERY_MARGIN,
    Scenario, ValidationError, build_config, describe_run, get_layout_positions, link_info, run_battle, run_scenario,
    to_jsonable, undirected_edges,
)

from live_panel import inject as inject_live_panel
from routing_rl.simulation.live import (
    HISTORY_DEFAULT, HISTORY_MAX, OUTAGE_DURATION, SPEEDS, TRAFFIC_MULT, ControlError, LiveSession,
)

MAX_BODY_BYTES = 16 * 1024   # request bodies are tiny JSON objects
MAX_STORED_RUNS = 12         # oldest recorded runs are evicted so memory cannot grow without bound
RUNS: "OrderedDict[str, dict]" = OrderedDict()
RUNS_LOCK = threading.Lock()


def js(o) -> str:
    return json.dumps(to_jsonable(o), allow_nan=False)


def galaxy_payload(sim, cfg, max_frames: int = 240) -> Dict:
    """Recorded history, normalised and down-sampled for the canvas replay."""
    pos = get_layout_positions(cfg.network.topology, sim.routers)
    ids = sorted(sim.routers.keys())
    xs = np.array([pos.get(i, (0, 0))[0] for i in ids], float)
    ys = np.array([pos.get(i, (0, 0))[1] for i in ids], float)
    nx = (xs - xs.min()) / (np.ptp(xs) or 1.0)
    ny = (ys - ys.min()) / (np.ptp(ys) or 1.0)
    nodes = [{"id": str(i), "x": float(nx[k]), "y": float(ny[k]), "mx": int(max(1, sim.routers[i].max_queue_size))}
             for k, i in enumerate(ids)]
    index = {i: k for k, i in enumerate(ids)}
    edges = undirected_edges(sim)

    hist = list(sim.history)
    step = max(1, len(hist) // max_frames)
    hist = hist[::step] + ([hist[-1]] if (len(hist) - 1) % step else [])
    frames = []
    for h in hist:
        links = []
        for u, v in edges:
            info = link_info(h, u, v)
            links.append([1 if info.get("up", True) else 0, round(float(info.get("util", 0) or 0), 3)])
        frames.append({"t": round(float(h["t"]), 2), "q": [float(h["queues"].get(i, 0)) for i in ids], "l": links})
    srcs = {str(s) for s in (cfg.traffic.sources or [])}
    dsts = {str(d) for d in (cfg.traffic.destinations or [])}
    for n in nodes:
        n["role"] = "source" if n["id"] in srcs else "destination" if n["id"] in dsts else "relay"
    return {"nodes": nodes, "edges": [[index[u], index[v]] for u, v in edges], "frames": frames,
            "label": f"{cfg.network.topology} · {len(ids)} nodes · {METHOD_NAMES[cfg.routing.method]}"}


GALAXY_HTML = r"""
<!doctype html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@300;500&family=Manrope:wght@500;700&display=swap');
html,body{margin:0;height:100%;background:#03030a;overflow:hidden;font-family:Manrope,sans-serif;color:#cfd6ff}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.hud{position:absolute;left:22px;top:18px;pointer-events:none}
.hud b{font-family:Unbounded,sans-serif;font-weight:500;font-size:15px;color:#fff}
.hud span{display:block;font-size:12px;color:#7d86b8;margin-top:3px}
#alert{position:absolute;right:22px;top:18px;font-family:Unbounded,sans-serif;font-size:12px;color:#ff4d5e;
 text-shadow:0 0 14px #ff4d5e;opacity:0;transition:opacity .4s}
.note{position:absolute;left:22px;bottom:78px;font-size:11px;color:#6b739f;max-width:560px;line-height:1.4;pointer-events:none}
.bar{position:absolute;left:18px;right:18px;bottom:16px;display:flex;gap:14px;align-items:center;padding:10px 16px;
 background:rgba(10,8,34,.62);border:1px solid rgba(140,130,255,.22);border-radius:16px;backdrop-filter:blur(10px)}
button,select{background:rgba(106,76,255,.25);border:1px solid rgba(140,130,255,.4);color:#fff;border-radius:10px;
 padding:6px 12px;font-family:Manrope,sans-serif;font-weight:700;cursor:pointer}
button:hover{background:rgba(255,61,154,.35)}
input[type=range]{flex:1;accent-color:#ff3d9a}
#tm{font-family:Unbounded,sans-serif;font-size:12px;min-width:92px;text-align:right;color:#3de8ff}
#tip{position:absolute;pointer-events:none;padding:8px 12px;border-radius:10px;font-size:12px;opacity:0;
 background:rgba(10,8,34,.88);border:1px solid rgba(61,232,255,.4);transition:opacity .15s;line-height:1.5}
</style></head><body>
<canvas id="c"></canvas>
<div class="hud"><b>Replay of the recorded run</b><span id="sub"></span></div>
<div id="alert"></div><div id="tip"></div>
<div class="note">Node colour, queue ring, link colour and link-lost alerts are recorded simulation state.
The moving comet particles and sounds are decorative: they are not packet-level traces of real routes.</div>
<div class="bar">
 <button id="pp">Pause</button><button id="cm">Cinematic</button><button id="sd">Sound off</button>
 <select id="sp"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select>
 <input id="sc" type="range" min="0" step="any" value="0"><span id="tm">0.00 s</span>
</div>
<script>
const D=__DATA__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const N=D.nodes.length,F=D.frames,nF=F.length;
let W,H,dpr,neb,mx=0,my=0,playing=true,speed=1,pos=0,last=performance.now(),T=0;
const $=id=>document.getElementById(id);
$('sub').textContent=D.label;$('sc').max=nF-1;

const stars=Array.from({length:520},()=>({x:Math.random(),y:Math.random(),z:Math.random()*.9+.1,p:Math.random()*6.28}));
function resize(){
  dpr=devicePixelRatio||1;W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  neb=document.createElement('canvas');neb.width=W;neb.height=H;const n=neb.getContext('2d');
  [[.2,.25,.5,'106,76,255'],[.8,.2,.4,'255,61,154'],[.55,.85,.5,'61,232,255'],[.1,.8,.35,'106,76,255'],[.9,.75,.3,'255,179,71']].forEach(b=>{
    const g=n.createRadialGradient(b[0]*W,b[1]*H,0,b[0]*W,b[1]*H,b[2]*Math.max(W,H));
    g.addColorStop(0,`rgba(${b[3]},.30)`);g.addColorStop(1,`rgba(${b[3]},0)`);n.fillStyle=g;n.fillRect(0,0,W,H)});
}
addEventListener('resize',resize);resize();
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();mx=(e.clientX-r.left)/W-.5;my=(e.clientY-r.top)/H-.5;hover(e.clientX-r.left,e.clientY-r.top)});

const padX=110,padTop=90,padBot=110,nodeScale=N>22?0.62:N>16?0.78:1;
const P=i=>({x:padX+D.nodes[i].x*(W-2*padX)+Math.sin(T*.7+i*1.9)*7+mx*-14,
             y:padTop+D.nodes[i].y*(H-padTop-padBot)+Math.cos(T*.6+i*2.3)*7+my*-14});
const E=D.edges.map((e,k)=>({a:e[0],b:e[1],c:((k%2)?1:-1)*(.08+.05*(k%3))}));
function ctl(e,A,B){const mx_=(A.x+B.x)/2,my_=(A.y+B.y)/2,dx=B.x-A.x,dy=B.y-A.y;return{x:mx_-dy*e.c,y:my_+dx*e.c}}
function qp(A,C,B,s){const u=1-s;return{x:u*u*A.x+2*u*s*C.x+s*s*B.x,y:u*u*A.y+2*u*s*C.y+s*s*B.y}}
const load=f=>f<.35?[61,232,255]:f<.7?[255,179,71]:[255,77,94];

let parts=[],pulses=[],lastUp=null,hot=-1;
let ac=null,snd=false,lastPing=0,cam={x:0,y:0,z:1},cine=false;
function tone(f,d,g,type,f2){if(!snd)return;const o=ac.createOscillator(),v=ac.createGain(),t=ac.currentTime;o.type=type||'sine';o.frequency.setValueAtTime(f,t);if(f2)o.frequency.exponentialRampToValueAtTime(f2,t+d);v.gain.setValueAtTime(g,t);v.gain.exponentialRampToValueAtTime(.0001,t+d);o.connect(v);v.connect(ac.destination);o.start();o.stop(t+d)}
function arrive(p){if(T-lastPing>.14){lastPing=T;tone(320+(p.h-190)*3,.15,.03)}}
function alarm(u){u?tone(500,.5,.08,'sine',900):tone(300,.7,.1,'sawtooth',110)}
function state(){
  const i=Math.min(nF-1,Math.floor(pos)),j=Math.min(nF-1,i+1),f=pos-i,a=F[i],b=F[j];
  return{q:a.q.map((v,k)=>v+(b.q[k]-v)*f),l:a.l.map((v,k)=>[v[0],v[1]+(b.l[k][1]-v[1])*f]),t:a.t+(b.t-a.t)*f}
}
function frame(now){
  const dt=Math.min(.05,(now-last)/1000);last=now;T+=dt;
  if(playing){pos+=dt*speed*(nF-1)/22;if(pos>=nF-1)pos=0}
  const S=state();$('tm').textContent=S.t.toFixed(2)+' s';$('sc').value=pos;

  ctx.fillStyle='#03030a';ctx.fillRect(0,0,W,H);
  ctx.globalCompositeOperation='lighter';ctx.globalAlpha=.8+.2*Math.sin(T*.4);
  ctx.save();ctx.translate(mx*-30,my*-30);ctx.drawImage(neb,-30,-30,W+60,H+60);ctx.restore();ctx.globalAlpha=1;
  stars.forEach(s=>{const x=((s.x*W+T*4*s.z+mx*-60*s.z)%W+W)%W,y=s.y*H+my*-60*s.z,tw=.5+.5*Math.sin(T*2*s.z+s.p);
    ctx.fillStyle=`rgba(210,220,255,${.15+.7*tw*s.z})`;ctx.beginPath();ctx.arc(x,y,s.z*1.3,0,6.283);ctx.fill()});

  const pts=D.nodes.map((_,i)=>P(i));
  const up=S.l.map(v=>v[0]>=.5?1:0);
  if(lastUp)up.forEach((u,k)=>{if(u!==lastUp[k]){alarm(u);const A=pts[E[k].a],B=pts[E[k].b];
    pulses.push({x:(A.x+B.x)/2,y:(A.y+B.y)/2,r:4,a:1,c:u?'93,255,180':'255,77,94'})}});
  lastUp=up;
  if(!cam.i){cam.x=W/2;cam.y=H/2;cam.i=1}
  let tz=1,tx=W/2,ty=H/2;
  if(cine){const dk=E.findIndex((e,k)=>!up[k]);
    if(dk>=0){const A=pts[E[dk].a],B=pts[E[dk].b];tx=(A.x+B.x)/2;ty=(A.y+B.y)/2;tz=1.7}
    else{let bi=-1,bf=.55;D.nodes.forEach((n,i)=>{const f=S.q[i]/n.mx;if(f>bf){bf=f;bi=i}});
      if(bi>=0){tx=pts[bi].x;ty=pts[bi].y;tz=1.55}else{tx=W/2+Math.sin(T*.15)*W*.12;ty=H/2+Math.cos(T*.12)*H*.08;tz=1.08}}}
  const kk=Math.min(1,dt*1.6);cam.x+=(tx-cam.x)*kk;cam.y+=(ty-cam.y)*kk;cam.z+=(tz-cam.z)*kk;
  ctx.save();ctx.translate(W/2,H/2);ctx.scale(cam.z,cam.z);ctx.translate(-cam.x,-cam.y);
  const downs=E.filter((e,k)=>!up[k]);
  const al=$('alert');al.style.opacity=downs.length?1:0;
  al.textContent=downs.length?'Link lost: '+downs.map(e=>D.nodes[e.a].id+' / '+D.nodes[e.b].id).join(', '):'';

  E.forEach((e,k)=>{
    const A=pts[e.a],B=pts[e.b],C=ctl(e,A,B),u=S.l[k][1],ok=up[k];
    const hue=190+u*150;
    ctx.beginPath();ctx.moveTo(A.x,A.y);ctx.quadraticCurveTo(C.x,C.y,B.x,B.y);
    if(ok){ctx.strokeStyle=`hsla(${hue},100%,65%,${.25+.5*u})`;ctx.lineWidth=1+5*u;ctx.shadowColor=`hsl(${hue},100%,60%)`;ctx.shadowBlur=8+22*u;ctx.setLineDash([])}
    else{const fl=Math.random()<.6;ctx.strokeStyle=`rgba(255,77,94,${fl?.7:.15})`;ctx.lineWidth=1.5;ctx.shadowColor='#ff4d5e';ctx.shadowBlur=14;ctx.setLineDash([3,9])}
    ctx.stroke();ctx.shadowBlur=0;ctx.setLineDash([]);
    if(ok){
      if(Math.random()<(.05+u*.9)*dt*60*speed*(playing?1:.2))parts.push({k,d:Math.random()<.5?0:1,s:0,v:.35+Math.random()*.4+u*.4,h:hue,tr:[]});
    }else if(Math.random()<.5){const p=qp(A,C,B,Math.random());parts.push({sp:1,x:p.x,y:p.y,vx:(Math.random()-.5)*60,vy:(Math.random()-.5)*60,life:1})}
  });
  parts=parts.filter(p=>{
    if(p.sp){p.x+=p.vx*dt;p.y+=p.vy*dt;p.life-=dt*2.2;if(p.life<=0)return false;
      ctx.fillStyle=`rgba(255,120,100,${p.life})`;ctx.fillRect(p.x,p.y,2,2);return true}
    p.s+=p.v*dt;if(p.s>=1&&up[p.k])arrive(p);if(p.s>=1||!up[p.k])return false;
    const e=E[p.k],A=pts[p.d?e.b:e.a],B=pts[p.d?e.a:e.b],C=ctl(e,pts[e.a],pts[e.b]),q=qp(A,C,B,p.s);
    p.tr.push(q);if(p.tr.length>14)p.tr.shift();
    for(let i=1;i<p.tr.length;i++){ctx.strokeStyle=`hsla(${p.h},100%,75%,${i/p.tr.length*.7})`;ctx.lineWidth=i/p.tr.length*3;
      ctx.beginPath();ctx.moveTo(p.tr[i-1].x,p.tr[i-1].y);ctx.lineTo(p.tr[i].x,p.tr[i].y);ctx.stroke()}
    ctx.fillStyle='#fff';ctx.shadowColor=`hsl(${p.h},100%,70%)`;ctx.shadowBlur=14;ctx.beginPath();ctx.arc(q.x,q.y,2.6,0,6.283);ctx.fill();ctx.shadowBlur=0;
    return true});
  if(parts.length>900)parts.splice(0,parts.length-900);

  pulses=pulses.filter(p=>{p.r+=dt*220;p.a-=dt*.8;if(p.a<=0)return false;
    ctx.strokeStyle=`rgba(${p.c},${p.a})`;ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,6.283);ctx.stroke();return true});

  D.nodes.forEach((n,i)=>{
    const {x,y}=pts[i],fr=Math.min(1,S.q[i]/n.mx),c=load(fr),R=(20+14*fr)*nodeScale,col=`${c[0]},${c[1]},${c[2]}`;
    const halo=ctx.createRadialGradient(x,y,R*.6,x,y,R*3.4);halo.addColorStop(0,`rgba(${col},${.35+.3*fr})`);halo.addColorStop(1,`rgba(${col},0)`);
    ctx.fillStyle=halo;ctx.beginPath();ctx.arc(x,y,R*3.4,0,6.283);ctx.fill();
    const g=ctx.createRadialGradient(x-R*.35,y-R*.4,R*.1,x,y,R);g.addColorStop(0,'#fff');g.addColorStop(.35,`rgb(${col})`);g.addColorStop(1,`rgba(${c[0]*.25|0},${c[1]*.25|0},${c[2]*.4|0},1)`);
    ctx.fillStyle=g;ctx.shadowColor=`rgb(${col})`;ctx.shadowBlur=26;ctx.beginPath();ctx.arc(x,y,R,0,6.283);ctx.fill();ctx.shadowBlur=0;
    ctx.strokeStyle='rgba(255,255,255,.12)';ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,R+9,0,6.283);ctx.stroke();
    ctx.strokeStyle=`rgb(${col})`;ctx.beginPath();ctx.arc(x,y,R+9,-1.5708,-1.5708+fr*6.283);ctx.stroke();
    const m=Math.min(10,Math.round(S.q[i]));
    for(let k=0;k<m;k++){const an=T*(.9+.07*k)+k*6.283/Math.max(m,1);
      ctx.fillStyle='rgba(255,255,255,.85)';ctx.beginPath();ctx.arc(x+Math.cos(an)*(R+20),y+Math.sin(an)*(R+20)*.55,2.2,0,6.283);ctx.fill()}
    ctx.fillStyle='#fff';ctx.font='500 12px Unbounded, sans-serif';ctx.textAlign='center';ctx.fillText(n.id,x,y+R+42);
    if(n.role==='source'){ctx.fillStyle='#5dffb4';ctx.font='600 10px Unbounded, sans-serif';ctx.fillText('SRC',x,y+R+56)}
    if(n.role==='destination'){ctx.fillStyle='#ff3d9a';ctx.font='600 10px Unbounded, sans-serif';ctx.fillText('DST',x,y+R+56)}
    if(i===hot){ctx.strokeStyle='#3de8ff';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,R+16,0,6.283);ctx.stroke()}
  });
  ctx.restore();ctx.globalCompositeOperation='source-over';
  requestAnimationFrame(frame);
}
function hover(px,py){
  let best=-1,bd=40;D.nodes.forEach((_,i)=>{const p=P(i),d=Math.hypot(p.x-px,p.y-py);if(d<bd){bd=d;best=i}});
  hot=best;const tip=$('tip');
  if(best<0){tip.style.opacity=0;return}
  const S=state(),n=D.nodes[best];
  tip.innerHTML=`<b>${n.id}</b> · ${n.role}<br>Queue ${S.q[best].toFixed(0)} of ${n.mx}`;
  tip.style.left=(px+16)+'px';tip.style.top=(py+16)+'px';tip.style.opacity=1;
}
$('pp').onclick=()=>{playing=!playing;$('pp').textContent=playing?'Pause':'Play'};
$('cm').onclick=()=>{cine=!cine;$('cm').textContent=cine?'Cinematic on':'Cinematic'};
$('sd').onclick=()=>{snd=!snd;if(snd&&!ac)ac=new (window.AudioContext||window.webkitAudioContext)();$('sd').textContent=snd?'Sound on':'Sound off'};
$('sp').onchange=e=>speed=+e.target.value;
$('sc').oninput=e=>{pos=+e.target.value};
requestAnimationFrame(frame);
</script></body></html>
"""


# ---------------------------------------------------------------------------
# Run handlers
# ---------------------------------------------------------------------------
def do_run(p: dict) -> dict:
    sc = Scenario.from_payload(p).with_seed()
    rr = run_scenario(sc)
    d = describe_run(rr)
    ids = sorted(rr.sim.routers)
    step = max(1, len(rr.sim.history) // 140)
    hist = rr.sim.history[::step]
    d["ids"] = [str(i) for i in ids]
    d["heat"] = [[min(1, h["queues"].get(i, 0) / max(1, rr.sim.routers[i].max_queue_size)) for h in hist] for i in ids]
    d["scenario"] = sc.__dict__
    payload = galaxy_payload(rr.sim, rr.cfg)
    d["label"] = payload["label"]
    rid = uuid.uuid4().hex[:10]
    with RUNS_LOCK:
        RUNS[rid] = payload
        while len(RUNS) > MAX_STORED_RUNS:
            RUNS.popitem(last=False)
    d["id"] = rid
    return d


def do_battle(p: dict) -> dict:
    sc, rows = run_battle(Scenario.from_payload(p))
    return {"rankings": rows, "seed": sc.seed, "n_nodes": rows[0]["n_nodes"], "scenario": sc.__dict__}


class Busy(Exception):
    pass


# ---------------------------------------------------------------------------
# Live Q-learning session (one at a time; it runs in its own thread, not on the HTTP threads)
# ---------------------------------------------------------------------------
LIVE_LOCK = threading.Lock()
LIVE: Dict[str, Any] = {"session": None}
LIVE_ALLOWED_FIELDS = {
    "play": set(), "pause": set(), "follow": set(), "step": set(), "speed": {"value"}, "reset": {"keep_learning"},
    "traffic": {"multiplier"}, "fail_link": {"a", "b", "duration"}, "restore_link": {"a", "b"},
    "explore": {"enabled"}, "params": {"alpha", "gamma"},
}


def _close_live() -> None:
    with LIVE_LOCK:
        if LIVE["session"] is not None:
            LIVE["session"].close()
            LIVE["session"] = None


atexit.register(_close_live)


def _live_session() -> LiveSession:
    s = LIVE["session"]
    if s is None:
        raise ValidationError("No live session yet. Press Start first.")
    return s


def do_live_start(p: dict) -> dict:
    hist = p.get("history", HISTORY_DEFAULT)
    if isinstance(hist, bool) or not isinstance(hist, int) or not 10 <= hist <= HISTORY_MAX:
        raise ValidationError(f"history must be an integer between 10 and {HISTORY_MAX}")
    # The live brain is always tabular Q-learning and never uses the scheduled failure: outages are user-triggered.
    sc = Scenario.from_payload(dict(p, method="tabular_q", enable_fail=False)).with_seed()
    cfg = build_config(sc, "tabular_q")
    with LIVE_LOCK:
        old = LIVE["session"]
        LIVE["session"] = None
        if old is not None:
            old.close()
        LIVE["session"] = LiveSession(cfg, history=hist)
        return LIVE["session"].snapshot()


def do_live_control(p: dict) -> dict:
    action = p.get("action")
    if not isinstance(action, str) or action not in LIVE_ALLOWED_FIELDS:
        raise ValidationError(f"action must be one of {sorted(LIVE_ALLOWED_FIELDS)}")
    kw = {k: v for k, v in p.items() if k != "action"}
    extra = set(kw) - LIVE_ALLOWED_FIELDS[action]
    if extra:
        raise ValidationError(f"unexpected field(s) for {action}: {sorted(extra)}")
    result = _live_session().control(action, **kw)
    return {"ok": True, **result}


def _int_param(q: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = q.get(name, [None])[0]
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValidationError(f"{name} must be an integer")
    if not lo <= v <= hi:
        raise ValidationError(f"{name} must be between {lo} and {hi}")
    return v


def do_live_state(q: dict) -> dict:
    s = LIVE["session"]
    if s is None:
        return {"active": False}
    focus = _int_param(q, "focus", -1, -1, 2**31)
    ufocus = _int_param(q, "ufocus", -1, -1, 2**31)
    return s.snapshot(focus=None if focus < 0 else focus, update_focus=None if ufocus < 0 else ufocus,
                      limit=_int_param(q, "limit", 40, 1, 200))


def do_live_qtable(q: dict) -> dict:
    s = _live_session()
    ident = re.compile(r"[A-Za-z0-9_.\-]{1,32}")
    out = {}
    for k in ("router", "dst"):
        v = q.get(k, [""])[0]
        if v and not ident.fullmatch(v):
            raise ValidationError(f"{k} is not a valid router id")
        out[k] = v or None
    return s.qtable_page(out["router"], out["dst"], _int_param(q, "offset", 0, 0, 10**7), _int_param(q, "limit", 50, 1, 200))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ct="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ct + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _err(self, code, msg):
        self._send(code, js({"error": msg}))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.replace("__LIMITS__", js({
                "rate": LIMITS["rate"], "duration": LIMITS["duration"], "priority": LIMITS["priority"],
                "fail_dur": LIMITS["fail_dur"], "margin": RECOVERY_MARGIN, "max_packets": MAX_PACKETS,
                "topologies": list(TOPOLOGY_CATALOG.keys()), "traffic": list(TRAFFIC_TYPES),
                "methods": [[m, METHOD_NAMES[m]] for m in METHODS],
                "live": {"speeds": list(SPEEDS), "traffic": list(TRAFFIC_MULT), "outage": list(OUTAGE_DURATION)}})), "text/html")
        elif u.path in ("/api/live/state", "/api/live/qtable"):
            try:
                q = parse_qs(u.query)
                self._send(200, js(do_live_state(q) if u.path.endswith("state") else do_live_qtable(q)))
            except (ValidationError, ControlError) as e:
                self._err(400, str(e))
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                self._err(500, f"Live view failed: {type(e).__name__}: {e}")
        elif u.path == "/galaxy":
            rid = parse_qs(u.query).get("id", [""])[0]
            with RUNS_LOCK:
                d = RUNS.get(rid) if re.fullmatch(r"[0-9a-f]{10}", rid) else None
            # "</" is escaped so run data can never close the <script> element
            self._send(200, GALAXY_HTML.replace("__DATA__", js(d).replace("</", "<\\/")) if d else "This run has expired. Launch again.", "text/html")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/run", "/api/battle", "/api/live/start", "/api/live/control"):
            return self._err(404, "not found")
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._err(400, "bad Content-Length")
        if n <= 0 or n > MAX_BODY_BYTES:
            return self._err(413 if n > MAX_BODY_BYTES else 400, "request body missing or too large")
        try:
            p = json.loads(self.rfile.read(n))
        except (ValueError, UnicodeDecodeError):
            return self._err(400, "request body is not valid JSON")
        if self.path.startswith("/api/live/"):
            try:
                result = do_live_start(p) if self.path.endswith("start") else do_live_control(p)
                return self._send(200, js(result))
            except (ValidationError, ControlError) as e:
                return self._err(400, str(e))
            except Exception as e:
                traceback.print_exc(file=sys.stderr)   # full traceback stays in the terminal
                return self._err(500, f"Live session failed: {type(e).__name__}: {e}")
        if not RUN_LOCK.acquire(blocking=False):
            return self._err(429, "A simulation is already running. Wait for it to finish.")
        try:
            result = do_run(p) if self.path == "/api/run" else do_battle(p)
            self._send(200, js(result))
        except ValidationError as e:
            self._err(400, str(e))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)  # full traceback stays in the terminal
            self._err(500, f"Simulation failed: {type(e).__name__}: {e}")
        finally:
            RUN_LOCK.release()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)   # loopback only, random free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    try:
        import webview
        webview.create_window("NeuroRouter", url, width=1560, height=960, background_color="#03030a")
        webview.start()
    except ImportError:
        print("Opening in your browser:", url, "(pip install pywebview for a native window)")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>NeuroRouter</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700&family=Manrope:wght@400;600;700&display=swap');
:root{--ion:#3de8ff;--vio:#6a4cff;--mag:#ff3d9a;--mint:#5dffb4;--ember:#ffb347;--lost:#ff4d5e;--dim:#7d86b8;--ink:#cfd6ff}
*{box-sizing:border-box}body{margin:0;background:#03030a;color:var(--ink);font-family:Manrope,sans-serif;min-height:100vh}
body::before{content:"";position:fixed;inset:-20%;z-index:0;pointer-events:none;filter:blur(30px);animation:d 40s ease-in-out infinite alternate;
 background:radial-gradient(40% 35% at 18% 22%,rgba(106,76,255,.28),transparent 70%),radial-gradient(35% 30% at 82% 18%,rgba(255,61,154,.18),transparent 70%),radial-gradient(45% 40% at 60% 90%,rgba(61,232,255,.14),transparent 70%)}
@keyframes d{to{transform:translate3d(4%,3%,0) rotate(6deg) scale(1.08)}}
.wrap{position:relative;z-index:1;max-width:1560px;margin:auto;padding:20px}
h1{font-family:Unbounded,sans-serif;font-size:2rem;margin:0;background:linear-gradient(95deg,#fff,var(--ion) 35%,var(--vio) 65%,var(--mag));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--dim);margin:4px 0 8px}
.disc{font-size:.8rem;color:#9aa4d8;border-left:3px solid var(--ion);padding:4px 10px;margin:0 0 14px;background:rgba(61,232,255,.06);border-radius:0 8px 8px 0}
.glass{background:linear-gradient(160deg,rgba(24,20,60,.55),rgba(8,8,24,.6));border:1px solid rgba(140,130,255,.18);border-radius:18px;padding:14px 18px;backdrop-filter:blur(14px)}
.ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;align-items:end}
label{display:block;font-size:.78rem;color:var(--dim)}output{color:var(--ion);float:right}
select,input[type=number]{width:100%;background:rgba(14,12,40,.8);border:1px solid rgba(140,130,255,.3);color:var(--ink);border-radius:9px;padding:7px}
input[type=range]{width:100%;accent-color:var(--mag)}
button{border:0;border-radius:12px;padding:12px 16px;font:700 .8rem Unbounded,sans-serif;color:#fff;cursor:pointer;background:linear-gradient(95deg,var(--vio),var(--mag));box-shadow:0 0 30px rgba(255,61,154,.35)}
button:hover{filter:brightness(1.2)}button:disabled{opacity:.5;cursor:wait}button.alt{background:linear-gradient(95deg,#0b6a8a,var(--vio))}
.orbs{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0}
.orb{display:flex;gap:12px;align-items:center;padding:10px 14px}
.ring{--p:0;--c:var(--ion);width:56px;height:56px;border-radius:50%;flex:none;background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(255,255,255,.07) 0);-webkit-mask:radial-gradient(circle,transparent 19px,#000 20px);mask:radial-gradient(circle,transparent 19px,#000 20px);filter:drop-shadow(0 0 8px var(--c))}
.good{--c:var(--mint)}.warn{--c:var(--ember)}.bad{--c:var(--lost)}.ion{--c:var(--ion)}
.v{font:500 1.15rem Unbounded,sans-serif;color:#fff}.l{font-size:.76rem;color:var(--dim)}
iframe{width:100%;height:680px;border:1px solid rgba(140,130,255,.22);border-radius:22px;box-shadow:0 0 80px rgba(80,60,255,.25);background:#03030a}
.grid{display:grid;grid-template-columns:1.6fr 1fr;gap:14px;margin-top:14px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.pt{font:500 .78rem Unbounded,sans-serif;margin-bottom:6px}canvas.ch{width:100%;height:170px;display:block}
.log{max-height:230px;overflow:auto;font:.78rem ui-monospace,monospace}.log div{margin:3px 0}
.warnc{color:var(--ember)}.goodc{color:var(--mint)}.badc{color:var(--lost)}.ionc{color:var(--ion)}
#busy{color:var(--ion);font:.8rem Unbounded,sans-serif;margin-left:12px}
#err{display:none;margin-top:10px;padding:8px 12px;border-radius:10px;border:1px solid var(--lost);color:#ffb3bb;background:rgba(255,77,94,.1);font-size:.85rem}
.row{display:flex;gap:12px;align-items:center;margin-top:10px;flex-wrap:wrap}
.bar{height:8px;border-radius:6px;background:linear-gradient(90deg,var(--vio),var(--mag));margin-top:6px}
details{margin-top:8px}summary{cursor:pointer;color:var(--dim);font-size:.78rem}
</style></head><body><div class="wrap">
<h1>NeuroRouter</h1><div class="sub">Deep-RL routing in a simulated network: random geometric S→D topology with 2-D coordinates</div>
<div class="disc"><b>Simulation, not a live network.</b> Each run finishes on the server first, then the page replays the recorded history.
Load, link state, KPIs and alerts are real simulation output; the flowing particles in the animation are decorative and are not packet traces.</div>
<div class="glass"><div class="ctl">
 <div><label>Routing policy</label><select id="method"></select></div>
 <div><label>Topology</label><select id="topo"></select></div>
 <div><label>Traffic</label><select id="traffic"></select></div>
 <div><label>Rate (pps) <output id="ro"></output></label><input id="rate" type="range" step="2.5"></div>
 <div><label>Duration (s) <output id="dro"></output></label><input id="dur" type="range" step="5"></div>
 <div><label>Priority share <output id="po"></output></label><input id="prio" type="range" step="0.02"></div>
 <div><label><input id="fail" type="checkbox" checked> Inject link failure</label><input id="fs" type="number" step="2" title="Failure start (s)"></div>
 <div><label>Outage (s)</label><input id="fd" type="number" step="1"></div>
 <div><label>Seed</label><input id="seed" type="number" value="42" min="1"><label style="margin-top:6px"><input id="lockseed" type="checkbox" checked> Lock seed (reproducible topology)</label></div>
</div><div class="row"><button id="run">Launch simulation</button><button id="battle" class="alt">Battle mode: all 4 policies</button><button id="shuffle" class="alt">Shuffle topology</button><span id="busy"></span></div>
<div id="err"></div></div>
<div id="kpi" class="orbs"></div>
<iframe id="gal" style="display:none" allow="autoplay"></iframe>
<div class="grid"><div class="glass"><div class="pt">Telemetry (red band = injected link failure)</div><div class="charts">
 <div><label>Average delay (s)</label><canvas class="ch" id="c1"></canvas></div><div><label>Delivery ratio (instantaneous)</label><canvas class="ch" id="c2"></canvas></div>
 <div><label>Queue depth (total packets)</label><canvas class="ch" id="c3"></canvas></div><div><label>Throughput (instantaneous)</label><canvas class="ch" id="c4"></canvas></div></div>
 <div class="pt" style="margin-top:14px">Router load over time (queue occupancy)</div><canvas class="ch" id="hm" style="height:230px"></canvas></div>
<div><div class="glass"><div class="pt">Event history (recorded link and queue transitions)</div><div class="log" id="alerts">Launch a simulation to begin.</div><div id="rawlog"></div></div>
<div class="glass" style="margin-top:14px"><div class="pt">Agent brain</div><div id="brain" class="l">No agent stats yet.</div></div>
<div class="glass" style="margin-top:14px"><div class="pt">Battle arena</div><div id="arena" class="l">Press Battle mode to run all four policies on the same seed, topology and failure schedule.</div></div></div></div>
</div><script>
const $=id=>document.getElementById(id);
const L=__LIMITS__;
// build controls from the server's limits so the UI cannot offer values the API will reject
$('method').innerHTML=L.methods.map(m=>`<option value="${m[0]}">${m[1]}</option>`).join('');
$('topo').innerHTML=L.topologies.map(t=>`<option>${t}</option>`).join('');
$('traffic').innerHTML=L.traffic.map(t=>`<option>${t}</option>`).join('');
const rng=(id,out,lim,val)=>{const e=$(id);e.min=lim[0];e.max=lim[1];e.value=val;$(out).value=val;e.oninput=()=>$(out).value=e.value};
rng('rate','ro',L.rate,25);rng('dur','dro',L.duration,60);rng('prio','po',L.priority,0.12);
$('fs').min=1;$('fs').max=L.duration[1];$('fs').value=28;$('fd').min=L.fail_dur[0];$('fd').max=L.fail_dur[1];$('fd').value=15;
const NAMES=Object.fromEntries(L.methods);
const err=m=>{const e=$('err');e.style.display=m?'block':'none';e.textContent=m||''};
async function post(u,b){const r=await fetch(u,{method:'POST',body:JSON.stringify(b)});let j;try{j=await r.json()}catch(e){throw new Error('Server returned an unreadable response')}
 if(!r.ok)throw new Error(j.error||('HTTP '+r.status));return j}
const params=()=>({method:$('method').value,topology:$('topo').value,rate:+$('rate').value,traffic_type:$('traffic').value,priority:+$('prio').value,duration:+$('dur').value,seed:$('lockseed').checked?+$('seed').value:null,enable_fail:$('fail').checked,fail_start:+$('fs').value,fail_dur:+$('fd').value});
const busy=t=>{$('busy').textContent=t;$('run').disabled=$('battle').disabled=$('shuffle').disabled=!!t};
async function go(path,txt,fn){err('');busy(txt);try{fn(await post(path,params()))}catch(e){err(e.message)}busy('')}
$('run').onclick=()=>go('/api/run','Simulating... (the full run finishes before anything is drawn)',render);
$('battle').onclick=()=>go('/api/battle','Four policies in the arena... (this takes a while)',arena);
$('shuffle').onclick=()=>{$('lockseed').checked=false;$('seed').value=1+Math.floor(Math.random()*9998)};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const orb=k=>`<div class="glass orb" title="${esc(k.note)}"><div class="ring ${k.tone}" style="--p:${k.pct}"></div><div><div class="v">${esc(k.value)}<small class="l"> ${k.value==='n/a'||k.value==='none'?'':esc(k.unit)}</small></div><div class="l">${esc(k.label)}</div></div></div>`;
const kget=(r,key)=>{const k=r.kpis.find(x=>x.key===key);return k?(k.value==='n/a'||k.value==='none'?k.value:k.value+(k.unit?' '+k.unit:'')):'n/a'};
function render(d){
 if(d.seed!=null)$('seed').value=d.seed;
 $('kpi').innerHTML=d.kpis.map(orb).join('');
 const g=$('gal');g.style.display='block';g.src='/galaxy?id='+d.id;
 const t=d.ts.t,w=d.window;
 line('c1',t,d.ts.avg_delay,'#ff3d9a',w);line('c2',t,d.ts.delivery_ratio_inst,'#5dffb4',w);line('c3',t,d.ts.queue_total,'#ffb347',w);line('c4',t,d.ts.throughput_inst,'#3de8ff',w);
 $('hm').style.height=Math.max(230,d.ids.length*18)+'px';heat('hm',d.ids,d.heat);
 $('alerts').innerHTML=d.events.length?d.events.map(a=>`<div class="${a.level}c">${a.t.toFixed(1).padStart(6)} s &nbsp; ${esc(a.text)}</div>`).join(''):'No overloads or link changes were recorded in this run.';
 $('rawlog').innerHTML=d.raw_log.length?`<details><summary>Simulator log (${d.raw_log.length} lines)</summary><div class="log">${d.raw_log.map(m=>`<div>${esc(m)}</div>`).join('')}</div></details>`:'';
 const ps=Object.entries(d.pstats);$('brain').innerHTML=ps.length?ps.map(([k,v])=>`<div style="display:flex;justify-content:space-between;padding:3px 0"><span>${esc(k.replace(/_/g,' '))}</span><b style="color:#3de8ff">${Number.isInteger(v)?v.toLocaleString():v.toFixed(3)}</b></div>`).join(''):'This policy has no learning stats.';
 g.scrollIntoView({behavior:'smooth',block:'center'})}
function arena(r){
 if(r.seed!=null)$('seed').value=r.seed;
 $('arena').innerHTML=`<div class="l" style="margin-bottom:8px">${r.n_nodes} nodes · seed ${r.seed} · same topology and failure schedule for all four. The DQN trains online during its run, so its score includes its learning period.</div>`
 +r.rankings.map((x,i)=>`<div style="margin:8px 0"><b style="color:#fff">#${i+1} ${esc(x.name)}</b> <span style="float:right;color:#ff3d9a">${x.score.toFixed(1)}</span><div class="bar" style="width:${Math.max(2,x.score)}%"></div><div>delivery ${kget(x,'delivery')}, loss ${kget(x,'loss')}, delay ${kget(x,'delay')}</div><div>route changes ${kget(x,'route_changes')}, adaptation ${kget(x,'adaptation')}</div></div>`).join('')}
function line(id,t,y,c,win){const cv=$(id),x=cv.getContext('2d'),w=cv.width=cv.clientWidth*2,h=cv.height=cv.clientHeight*2;x.clearRect(0,0,w,h);if(!y||!y.length)return;
 const vals=y.filter(v=>v!=null);if(!vals.length)return;
 const mn=Math.min(...vals,0),mx=Math.max(...vals)||1,T0=t[0],T1=t[t.length-1]||1,X=v=>50+(v-T0)/(T1-T0||1)*(w-60),Y=v=>h-24-(v-mn)/(mx-mn||1)*(h-44);
 if(win){x.fillStyle='rgba(255,77,94,.14)';x.fillRect(X(win[0]),10,Math.max(2,X(win[1])-X(win[0])),h-34)}
 x.strokeStyle='rgba(140,130,255,.12)';x.lineWidth=1;for(let i=0;i<5;i++){const yy=20+i*(h-44)/4;x.beginPath();x.moveTo(50,yy);x.lineTo(w-10,yy);x.stroke()}
 x.beginPath();let pen=false;y.forEach((v,i)=>{if(v==null){pen=false;return}pen?x.lineTo(X(t[i]),Y(v)):x.moveTo(X(t[i]),Y(v));pen=true});
 x.strokeStyle=c;x.lineWidth=3;x.shadowColor=c;x.shadowBlur=12;x.stroke();x.shadowBlur=0;
 x.fillStyle='#7d86b8';x.font='20px Manrope';x.fillText(mx.toFixed(2),2,24);x.fillText(mn.toFixed(2),2,h-26)}
const ramp=v=>{const S=[[3,3,10],[75,43,214],[255,61,154],[255,211,92]],p=Math.min(.999,Math.max(0,v))*3,i=Math.floor(p),f=p-i;return`rgb(${S[i].map((a,k)=>a+(S[i+1][k]-a)*f|0)})`};
function heat(id,ids,z){const cv=$(id),x=cv.getContext('2d'),w=cv.width=cv.clientWidth*2,h=cv.height=cv.clientHeight*2,Lm=90,rh=h/ids.length,cw=(w-Lm)/z[0].length;x.clearRect(0,0,w,h);
 z.forEach((r,i)=>{r.forEach((v,j)=>{x.fillStyle=ramp(v);x.fillRect(Lm+j*cw,i*rh,cw+1,rh-2)});x.fillStyle='#9aa4d8';x.font='22px Manrope';x.fillText(ids[i],6,i*rh+rh/2+8)})}
</script></body></html>"""


PAGE = inject_live_panel(PAGE)   # adds the LIVE Q-LEARNING BRAIN section; everything else is unchanged


if __name__ == "__main__":
    main()