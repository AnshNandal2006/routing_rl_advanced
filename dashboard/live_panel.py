"""HTML / CSS / JS for the "LIVE Q-LEARNING BRAIN" panel of NeuroRouter.

Rendering and user interaction only. Every number shown comes from the /api/live/* endpoints,
which read them from the real TabularQPolicy and Simulator (see routing_rl/simulation/live.py).
No Q-values, decisions or explanations are generated in JavaScript.

`inject(page)` splices the panel into the existing NeuroRouter page without touching its other parts.
"""

LIVE_CSS = r"""
#qbrain{margin-top:14px;padding:16px 18px}
.qb-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.qb-title{font:700 1rem Unbounded,sans-serif;letter-spacing:.02em;background:linear-gradient(95deg,#fff,var(--ion) 60%,var(--mint));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.chip{font:700 .72rem Unbounded,sans-serif;padding:6px 12px;border-radius:999px;border:1px solid rgba(140,130,255,.35);color:var(--dim);white-space:nowrap}
.chip.running{color:var(--mint);border-color:var(--mint)}.chip.paused{color:var(--ember);border-color:var(--ember)}
.chip.finished{color:var(--dim)}.chip.error{color:var(--lost);border-color:var(--lost)}
.qb-ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}
.qb-ctl .sep{width:1px;height:26px;background:rgba(140,130,255,.25);margin:0 4px}
.qb-ctl button{padding:8px 12px;font-size:.68rem;box-shadow:none}
.qb-ctl select,.qb-ctl input[type=number]{width:auto;padding:5px 7px;font-size:.78rem}
.qb-ctl input[type=number]{width:64px}.qb-ctl label{display:inline-flex;gap:5px;align-items:center;font-size:.76rem}
.qb-ctl .grp{font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
button.ghost{background:rgba(106,76,255,.22);border:1px solid rgba(140,130,255,.4)}
button.warn{background:linear-gradient(95deg,#8a4b0b,#c9561f)}button.good{background:linear-gradient(95deg,#0b7a5a,#16a37a)}
#qbMsg{display:none;margin-top:8px;font-size:.82rem;padding:6px 10px;border-radius:8px;border:1px solid var(--ember);color:#ffd9a0;background:rgba(255,179,71,.08)}
#qbMsg.bad{border-color:var(--lost);color:#ffb3bb;background:rgba(255,77,94,.1)}
.qb-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:12px 0}
.qk{padding:8px 10px;border-radius:12px;background:rgba(14,12,40,.7);border:1px solid rgba(140,130,255,.18)}
.qk .v{font:500 1.05rem Unbounded,sans-serif;color:#fff;transition:color .3s}.qk .k{font-size:.68rem;color:var(--dim);margin-top:2px}
.qk.green .v{color:var(--mint)}.qk.amber .v{color:var(--ember)}.qk.red .v{color:var(--lost)}.qk.grey .v{color:var(--dim)}
.qb-grid{display:grid;grid-template-columns:minmax(340px,1fr) minmax(420px,1.25fr);gap:12px;margin-top:6px}
.qb-grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:1180px){.qb-grid,.qb-grid3{grid-template-columns:1fr}}
.qb-card{padding:12px 14px;border-radius:14px;background:rgba(10,8,34,.55);border:1px solid rgba(140,130,255,.16);min-width:0}
.qb-card h4{margin:0 0 8px;font:500 .74rem Unbounded,sans-serif;color:#dfe4ff;letter-spacing:.03em}
.qb-card .sub{font-size:.74rem;color:var(--dim);margin-bottom:8px;line-height:1.4}
svg.qmap{width:100%;height:auto;display:block;background:rgba(3,3,10,.55);border-radius:10px}
.qmap .e{stroke:#3a4470;stroke-width:1.5;transition:stroke .4s,stroke-width .4s}
.qmap .e.cur{stroke:var(--mint);stroke-width:4}.qmap .e.prev{stroke:#7d86b8;stroke-width:2;stroke-dasharray:2 6;opacity:.7}
.qmap .e.alt{stroke:var(--ember);stroke-width:2.5;stroke-dasharray:7 4}.qmap .e.down{stroke:var(--lost);stroke-width:2.5;stroke-dasharray:3 5}
.qmap .e.sel{stroke:var(--mint);stroke-width:5;animation:pulse 1.2s ease-out 1}
@keyframes pulse{0%{stroke-width:12;opacity:.4}100%{stroke-width:5;opacity:1}}
.qmap text{font:600 9px Manrope,sans-serif;fill:#fff;text-anchor:middle;pointer-events:none}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:.68rem;color:var(--dim);margin-top:6px}
.legend i{display:inline-block;width:18px;height:0;border-top:3px solid;vertical-align:middle;margin-right:4px}
.rt{font-size:.8rem;line-height:1.7;margin-top:8px}.rt .path{font:500 .78rem ui-monospace,monospace}
.rt s{color:var(--dim)}
table.qt{width:100%;border-collapse:collapse;font-size:.76rem}
table.qt th{font:500 .66rem Manrope,sans-serif;text-align:left;color:var(--dim);padding:4px 6px;border-bottom:1px solid rgba(140,130,255,.2);white-space:nowrap}
table.qt td{padding:6px;border-bottom:1px solid rgba(140,130,255,.08);vertical-align:top}
table.qt td.num{font-family:ui-monospace,monospace;text-align:right;white-space:nowrap}
table.qt tr.selected{background:rgba(93,255,180,.10)}table.qt tr.explored{background:rgba(255,179,71,.10)}
table.qt tr.down{background:rgba(255,77,94,.08)}table.qt tr.down td{color:#ffb3bb}
table.qt tr.excluded td,table.qt tr.unavailable td{color:#7d86b8}
.qt .un{color:#7d86b8}.qt small{display:block;color:var(--dim);font-size:.66rem}
.flash{animation:fl 1.1s ease-out}@keyframes fl{0%{background:rgba(61,232,255,.35)}100%{background:transparent}}
.bd{display:inline-block;padding:2px 8px;border-radius:999px;font:700 .62rem Manrope,sans-serif;border:1px solid}
.bd.selected{color:var(--mint);border-color:var(--mint)}.bd.explored{color:var(--ember);border-color:var(--ember)}
.bd.alternative{color:#e8c37a;border-color:rgba(255,179,71,.5)}.bd.down{color:var(--lost);border-color:var(--lost)}
.bd.excluded,.bd.unavailable{color:var(--dim);border-color:rgba(125,134,184,.5)}
.why .ln{display:flex;gap:8px;margin:7px 0;font-size:.8rem;line-height:1.45}
.tg{flex:none;font:700 .58rem Manrope,sans-serif;letter-spacing:.05em;padding:2px 6px;border-radius:5px;height:fit-content;margin-top:2px}
.tg.policy{background:rgba(106,76,255,.35);color:#d9d2ff}.tg.learned{background:rgba(61,232,255,.2);color:var(--ion)}
.tg.observed{background:rgba(93,255,180,.18);color:var(--mint)}.tg.note{background:rgba(255,179,71,.2);color:var(--ember)}
.mth{font-size:.78rem}.mth .f{font:500 .86rem ui-monospace,monospace;padding:8px 10px;border-radius:8px;background:rgba(3,3,10,.6);margin:6px 0;overflow-x:auto;white-space:nowrap}
.mth table{width:100%;font-size:.76rem;border-collapse:collapse}.mth td{padding:3px 6px;border-bottom:1px solid rgba(140,130,255,.08)}
.mth td:last-child{text-align:right;font-family:ui-monospace,monospace}
.ok{color:var(--mint)}.bad{color:var(--lost)}
.tlw{max-height:430px;overflow:auto;padding-right:4px}
.tl{padding:8px 10px;margin:6px 0;border-radius:10px;background:rgba(14,12,40,.65);border:1px solid rgba(140,130,255,.14);font-size:.76rem;line-height:1.5;cursor:pointer;transition:border-color .2s}
.tl:hover{border-color:rgba(61,232,255,.5)}.tl.sel{border-color:var(--mint);box-shadow:0 0 0 1px var(--mint) inset}
.tl.chg{border-left:4px solid var(--ember)}.tl.new{animation:fl 1.2s ease-out}
.tl .mono,.mono{font-family:ui-monospace,monospace}.tl s{color:var(--dim)}
.ulist{max-height:150px;overflow:auto;font-size:.72rem;margin-top:8px}
.ulist div{padding:2px 4px;cursor:pointer;border-radius:5px}.ulist div:hover{background:rgba(61,232,255,.1)}.ulist div.sel{background:rgba(93,255,180,.12)}
.qb-foot{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:1180px){.qb-foot{grid-template-columns:1fr}}
.qb-foot .lg{max-height:170px;overflow:auto;font:.72rem ui-monospace,monospace}.qb-foot .lg div{margin:2px 0}
details.qfull summary{cursor:pointer;color:#cfd6ff;font-size:.78rem;margin-top:12px}
.qfull .tools{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
.qfull .tools button{padding:6px 10px;font-size:.66rem;box-shadow:none}
.qempty{color:var(--dim);font-size:.82rem;padding:14px 4px}
"""

LIVE_HTML = r"""
<section id="qbrain" class="glass">
 <div class="qb-head">
  <div><div class="qb-title">LIVE Q-LEARNING BRAIN — ADAPTIVE ROUTE DECISION ENGINE</div>
   <div class="l" style="margin-top:4px;max-width:900px">A real tabular Q-learning agent routing packets in the real simulator. Every Q-value, decision, update and explanation below is read from that agent and its event records. This is a simulation, not a live network. The session uses the topology, traffic, rate, priority and seed settings above; link failures here are triggered by you.</div></div>
  <div id="qbStatus" class="chip">no session</div>
 </div>
 <div class="qb-ctl">
  <button id="qbPlay" class="good">Start</button><button id="qbPause" class="ghost">Pause</button>
  <button id="qbStep" class="ghost">Step 1 decision</button>
  <select id="qbSpeed" title="Simulated seconds per real second"></select>
  <button id="qbNew" class="ghost" title="Build a fresh session from the settings above">New session from settings</button>
  <span class="sep"></span><span class="grp">Reset episode</span>
  <button id="qbResetKeep" class="ghost" title="New network and clock; Q-table is carried over">keep Q-table</button>
  <button id="qbResetWipe" class="warn" title="New network and clock; Q-table is emptied">wipe Q-table</button>
 </div>
 <div class="qb-ctl">
  <span class="grp">Traffic</span>
  <button class="ghost" data-mult="2">×2</button><button class="ghost" data-mult="3">×3</button><button class="ghost" data-mult="1">Normal</button>
  <span class="sep"></span><span class="grp">Link</span>
  <select id="qbLink"></select><label>for <input id="qbDur" type="number" min="1" max="60" step="1" value="15"> s</label>
  <button id="qbFail" class="warn">Fail link</button><button id="qbRestore" class="ghost">Restore link</button>
  <span class="sep"></span><label><input id="qbExplore" type="checkbox" checked> Exploration (ε-greedy)</label>
  <span class="sep"></span><span class="grp">α</span><input id="qbAlpha" type="number" min="0.01" max="1" step="0.01" value="0.15">
  <span class="grp">γ</span><input id="qbGamma" type="number" min="0" max="0.99" step="0.01" value="0.92">
  <button id="qbApply" class="ghost">Apply</button>
 </div>
 <div id="qbMsg"></div>
 <div id="qbReset" class="l" style="margin-top:8px"></div>
 <div id="qbKpi" class="qb-kpis"></div>
 <div class="qb-grid">
  <div class="qb-card"><h4>ROUTE MAP &amp; ROUTE COMPARISON</h4>
   <div id="qbMap"><div class="qempty">Press Start to create a session.</div></div>
   <div class="legend"><span><i style="border-color:#5dffb4"></i>current route</span><span><i style="border-color:#7d86b8;border-top-style:dotted"></i>previous route</span><span><i style="border-color:#ffb347;border-top-style:dashed"></i>alternative at focus router</span><span><i style="border-color:#ff4d5e;border-top-style:dashed"></i>link down</span></div>
   <div id="qbRoute" class="rt"></div></div>
  <div class="qb-card"><h4>LIVE Q-TABLE — DECISION STATE</h4><div id="qbFocusHead" class="sub"></div><div id="qbTable"><div class="qempty">No decisions yet.</div></div></div>
 </div>
 <div class="qb-grid3">
  <div class="qb-card why"><h4>WHY DID THE AI CHOOSE THIS PATH?</h4><div id="qbWhy"><div class="qempty">No decisions yet.</div></div></div>
  <div class="qb-card"><h4>LEARNING UPDATE — MATHEMATICS</h4>
   <details open><summary class="l" style="cursor:pointer">Show / hide calculation</summary><div id="qbMath" class="mth"><div class="qempty">No Q-value update yet.</div></div></details>
   <div class="l" style="margin-top:8px">Recent updates (click to inspect)</div><div id="qbUpd" class="ulist"></div></div>
  <div class="qb-card"><h4>ROUTE DECISION TIMELINE</h4>
   <div class="sub"><label><input id="qbOnlyChg" type="checkbox"> route changes only</label> · <a href="#" id="qbFollow" style="color:var(--ion)">follow latest</a> · click a card to inspect it</div>
   <div id="qbTl" class="tlw"><div class="qempty">No decisions yet.</div></div></div>
 </div>
 <details class="qfull" id="qbFull"><summary>Full Q-table explorer (all learned state/action pairs)</summary>
  <div class="tools"><select id="qfRouter"><option value="">all routers</option></select><select id="qfDst"><option value="">all destinations</option></select>
   <button class="ghost" id="qfPrev">‹ prev</button><button class="ghost" id="qfNext">next ›</button><button class="ghost" id="qfRefresh">refresh</button><span id="qfInfo" class="l"></span></div>
  <div id="qfTable"></div></details>
 <div class="qb-foot"><div class="qb-card"><h4>NETWORK EVENTS (from the simulator's event system)</h4><div id="qbEvents" class="lg"></div></div>
  <div class="qb-card"><h4>YOUR CONTROL ACTIONS</h4><div id="qbCtlLog" class="lg"></div></div></div>
</section>
"""

LIVE_JS = r"""
// ===================== LIVE Q-LEARNING BRAIN =====================
(()=>{
const Q=id=>document.getElementById(id);
const LV=L.live||{speeds:[1],traffic:[0.25,4],outage:[1,60]};
let S=null,focus=null,ufocus=null,inflight=false,lastStr='',prevQ={},lastFocusSeq=0,seenSeq=0,linkTouched=false,edgesKey='',qfOffset=0,starting=false;
const fq=x=>x==null?'—':Number(x).toFixed(2), fq4=x=>x==null?'—':Number(x).toFixed(4);
const ph=p=>(p||[]).map(esc).join(' → ');
async function lget(p){const r=await fetch(p);let j;try{j=await r.json()}catch(e){throw new Error('unreadable server response')}if(!r.ok)throw new Error(j.error||('HTTP '+r.status));return j}
async function lpost(p,b){const r=await fetch(p,{method:'POST',body:JSON.stringify(b)});let j;try{j=await r.json()}catch(e){throw new Error('unreadable server response')}if(!r.ok)throw new Error(j.error||('HTTP '+r.status));return j}
function msg(t,bad){const e=Q('qbMsg');if(!t){e.style.display='none';return}e.className=bad?'bad':'';e.style.display='block';e.textContent=t}
Q('qbSpeed').innerHTML=LV.speeds.map(s=>`<option value="${s}"${s===1?' selected':''}>${s}× speed</option>`).join('');

async function startSession(){
  if(starting)return;starting=true;msg('');
  try{const p=Object.assign(params(),{method:'tabular_q',enable_fail:false});
    S=await lpost('/api/live/start',p);focus=ufocus=null;lastStr='';prevQ={};seenSeq=0;edgesKey='';
    if(S.scenario&&S.scenario.seed!=null)$('seed').value=S.scenario.seed;render()}
  catch(e){msg(e.message,true)}finally{starting=false}
}
async function ctl(action,extra){
  msg('');
  try{if(!S||!S.active){await startSession();if(!S||!S.active)return null}
    const r=await lpost('/api/live/control',Object.assign({action},extra||{}));
    if(r&&r.message)msg(r.message,false);await poll(true);return r}
  catch(e){msg(e.message,true);return null}
}
Q('qbPlay').onclick=()=>ctl('play');
Q('qbPause').onclick=()=>ctl('pause');
Q('qbStep').onclick=()=>{focus=null;return ctl('step')};
Q('qbNew').onclick=startSession;
Q('qbSpeed').onchange=e=>ctl('speed',{value:+e.target.value});
Q('qbResetKeep').onclick=()=>ctl('reset',{keep_learning:true});
Q('qbResetWipe').onclick=()=>ctl('reset',{keep_learning:false});
document.querySelectorAll('#qbrain [data-mult]').forEach(b=>b.onclick=()=>ctl('traffic',{multiplier:+b.dataset.mult}));
const link=()=>{const v=Q('qbLink').value.split('|');return{a:v[0],b:v[1]}};
Q('qbLink').onchange=()=>{linkTouched=true};
Q('qbFail').onclick=()=>{linkTouched=true;return ctl('fail_link',Object.assign(link(),{duration:+Q('qbDur').value}))};   // keep the chosen link selected so Restore targets it
Q('qbRestore').onclick=async()=>{const r=await ctl('restore_link',link());if(r)linkTouched=false};
Q('qbExplore').onchange=e=>ctl('explore',{enabled:e.target.checked});
Q('qbApply').onclick=()=>ctl('params',{alpha:+Q('qbAlpha').value,gamma:+Q('qbGamma').value});
Q('qbFollow').onclick=e=>{e.preventDefault();focus=ufocus=null;ctl('follow')};
Q('qbOnlyChg').onchange=()=>{if(S)renderTimeline()};

async function poll(force){
  if(inflight||(!S||!S.active)&&!force)return;inflight=true;
  try{const q=new URLSearchParams({limit:40});if(focus!=null)q.set('focus',focus);if(ufocus!=null)q.set('ufocus',ufocus);
    const n=await lget('/api/live/state?'+q);if(!n.active){S=null;inflight=false;return}
    const s=JSON.stringify(n);if(s!==lastStr){lastStr=s;S=n;render()}}
  catch(e){msg('Lost contact with the simulator: '+e.message,true)}
  inflight=false;
}
function schedule(){const fast=S&&S.status==='running';setTimeout(async()=>{await poll(false);schedule()},fast?500:1500)}
schedule();

function render(){
  try{header();kpis();renderMap();renderFocus();renderWhy();renderMath();renderTimeline();renderEvents();syncControls()}
  catch(e){console.error(e);msg('Display error: '+e.message,true)}
}
function header(){
  const c=Q('qbStatus');if(!S){c.className='chip';c.textContent='no session';return}
  c.className='chip '+S.status;
  c.textContent=`${S.status.toUpperCase()} · t = ${S.t.toFixed(1)} / ${S.duration} s · episode ${S.episode}`;
  const r=S.reset_info||{},qs=r.q_table==='initial'?'fresh (empty) Q-table':r.q_table==='preserved'?`Q-table PRESERVED from the previous episode (${r.q_entries} entries, ε = ${(+r.epsilon).toFixed(3)})`:'Q-table WIPED (empty, ε restored)';
  Q('qbReset').innerHTML=`Episode ${S.episode}: ${qs}. ${esc(S.scenario.topology)} · ${S.scenario.n_nodes} routers · seed ${S.scenario.seed} · flow ${S.scenario.sources.map(esc).join(',')} → ${S.scenario.destinations.map(esc).join(',')}`+
    (S.error?`<div class="bad mono" style="white-space:pre-wrap;margin-top:6px">${esc(S.error)}</div>`:'');
}
function kcard(k,v,cls,title){return `<div class="qk ${cls||''}" title="${esc(title||'')}"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`}
function kpis(){
  if(!S){Q('qbKpi').innerHTML='';return}
  const k=S.kpis,mode=k.mode,sc=mode==='explore'?'amber':mode?'green':'grey';
  Q('qbKpi').innerHTML=
   kcard('Current router',k.router!=null?esc(k.router):'—',k.router?'':'grey')+
   kcard('Destination',k.dst!=null?esc(k.dst):'—',k.dst?'':'grey')+
   kcard('Selected next hop',k.selected!=null?esc(k.selected)+(mode==='explore'?' <small>explored</small>':''):'—',sc,'Green = greedy/exploit choice, amber = exploration choice')+
   kcard('Best valid Q-value',fq(k.best_valid_q),k.best_valid_q==null?'grey':'','Highest current Q among the valid actions of the decision state (unvisited actions count as 0.00)')+
   kcard('Exploration ε',k.explore_enabled?(+k.epsilon).toFixed(3):'OFF',k.explore_enabled?'amber':'grey','Probability of a random action. OFF = purely greedy (you switched it off).')+
   kcard('Q-value updates',(k.updates).toLocaleString(),'','Total real calls of the Q-learning update (hop, delivery and drop updates)')+
   kcard('Route changes observed',k.route_changes.toLocaleString(),'','Committed decisions whose next hop differs from the previous decision at the same router and destination (includes exploration)')+
   kcard('Q-table entries',`${k.q_entries} <small>/ ${k.q_states} states</small>`,'')+
   kcard('Packets delivered',k.delivered.toLocaleString(),'');
}
function pairs(path,set){for(let i=0;i+1<(path||[]).length;i++){const a=path[i],b=path[i+1];set.add(a<b?a+'|'+b:b+'|'+a)}}
function renderMap(){
  if(!S){return}
  const g=S.graph,W=620,H=330,px=40,py=30,pos={};g.nodes.forEach(n=>pos[n.id]={x:px+n.x*(W-2*px),y:py+n.y*(H-2*py)});
  const key=(a,b)=>a<b?a+'|'+b:b+'|'+a,cur=new Set(),prev=new Set(),alt=new Set(),down=new Set(S.down_links.map(e=>key(e[0],e[1])));
  const r=S.routes[0];if(r){pairs(r.current.path,cur);if(r.previous)pairs(r.previous.path,prev)}
  const fd=S.focus&&S.focus.decision;let sel=null;
  if(fd){sel=key(fd.router,fd.selected);S.focus.rows.forEach(x=>{if(x.status==='alternative')alt.add(key(fd.router,x.hop))})}
  const isNew=fd&&fd.seq!==lastFocusSeq&&focus==null;if(fd)lastFocusSeq=fd.seq;
  let h=`<svg class="qmap" viewBox="0 0 ${W} ${H}">`;
  g.edges.forEach(e=>{const k=key(e[0],e[1]),a=pos[e[0]],b=pos[e[1]];
    let c='e';if(down.has(k))c+=' down';else if(k===sel)c+=' sel';else if(cur.has(k))c+=' cur';else if(alt.has(k))c+=' alt';else if(prev.has(k))c+=' prev';
    h+=`<line class="${isNew&&k===sel?c:c.replace(' sel',' cur')}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`});
  g.nodes.forEach(n=>{const p=pos[n.id],ring=n.role==='source'?'#5dffb4':n.role==='destination'?'#ff3d9a':'#6a4cff',f=fd&&fd.router===n.id;
    if(f)h+=`<circle cx="${p.x}" cy="${p.y}" r="19" fill="none" stroke="#3de8ff" stroke-width="1.5" opacity=".8"/>`;
    h+=`<circle cx="${p.x}" cy="${p.y}" r="13" fill="#16213e" stroke="${ring}" stroke-width="2.5"/><text x="${p.x}" y="${p.y+3}">${esc(n.id)}</text>`});
  Q('qbMap').innerHTML=h+'</svg>';
  let t='';
  if(!r)t='<span class="l">No packet has been delivered yet, so no route is known.</span>';
  else{t=`<div>Latest delivered route (${esc(r.flow)}): <span class="path" style="color:#5dffb4">${ph(r.current.path)}</span> <span class="l">since t = ${r.current.since} s</span></div>`;
    t+=r.previous?`<div>Previous route: <s class="path">${ph(r.previous.path)}</s> <span class="l">(replaced)</span></div>`:'<div class="l">No route change observed yet.</div>';
    if(r.dominant)t+=`<div class="l">Most used route over the last ${r.window} deliveries: <span class="path">${ph(r.dominant.path)}</span> (${(r.dominant.share*100).toFixed(0)}%)</div>`}
  if(S.down_links.length)t+=`<div class="bad">Unavailable: ${S.down_links.map(e=>esc(e[0])+'↔'+esc(e[1])).join(', ')}</div>`;
  Q('qbRoute').innerHTML=t;
}
function renderFocus(){
  const f=S&&S.focus;if(!f){Q('qbFocusHead').innerHTML='';Q('qbTable').innerHTML='<div class="qempty">No routing decision has been made yet. Press Start or Step 1 decision.</div>';return}
  const d=f.decision;
  Q('qbFocusHead').innerHTML=`Decision #${d.seq} at t = ${d.t.toFixed(2)} s · packet #${d.packet_id}${d.priority?' (priority)':''} · path so far <span class="mono">${ph(d.path.concat([d.router]))}</span> · current state <span class="mono">(${esc(d.router)}, ${esc(d.dst)}, level ${d.queue_level})</span> · destination <b>${esc(d.dst)}</b>`+(focus!=null?' · <span style="color:var(--ember)">pinned</span>':'')+(S.kpis.pending_decisions>0?` · <span class="l">(+${S.kpis.pending_decisions} more decision(s) in this simulator tick not revealed yet)</span>`:'');
  const lab={selected:'SELECTED',explored:'EXPLORED',alternative:'alternative',down:'LINK DOWN',excluded:'loop-excluded',unavailable:'unavailable'};
  let h='<table class="qt"><tr><th>State / router</th><th>Dest</th><th>Next hop</th><th style="text-align:right">Q-value (live)</th><th>Link cost (observed)</th><th>Status</th></tr>';
  f.rows.forEach(r=>{const k=r.router+'|'+r.dst+'|'+r.level+'|'+r.hop,old=prevQ[k],ch=old!==undefined&&old!==r.q;prevQ[k]=r.q;
    const q=r.visited?fq(r.q):'<span class="un">Unvisited</span><small>treated as 0.00</small>';
    const was=r.visited&&r.q_at_decision!=null&&r.q!==r.q_at_decision?`<small>${fq(r.q_at_decision)} at decision</small>`:'';
    const lu=r.last_update?`<small>last update ${fq(r.last_update.q_old)} → ${fq(r.last_update.q_new)} (t=${r.last_update.t==null?'—':r.last_update.t})</small>`:'';
    const lc=r.link_up?`${r.link_delay_ms} ms · util ${(r.link_util*100).toFixed(0)}%`:'<span class="bad">DOWN</span>';
    h+=`<tr class="${r.status}"><td class="mono">${esc(r.router)} <small>lvl ${r.level}</small></td><td>${esc(r.dst)}</td><td class="mono"><b>${esc(r.hop)}</b></td><td class="num ${ch?'flash':''}">${q}${was}${lu}</td><td>${lc}</td><td><span class="bd ${r.status}">${lab[r.status]||r.status}</span></td></tr>`});
  Q('qbTable').innerHTML=h+'</table><div class="l" style="margin-top:6px">Q-values are read from the agent\'s table now. Link cost = propagation + queued transmit delay, observed from the simulator; the Q-learning agent does not use it directly (it only learns from rewards).</div>';
}
function renderWhy(){
  const f=S&&S.focus;if(!f){Q('qbWhy').innerHTML='<div class="qempty">No decisions yet.</div>';return}
  const tg={policy:'POLICY',learned:'LEARNED',observed:'OBSERVED',note:'NOTE'};
  Q('qbWhy').innerHTML=f.decision.explanation.map(e=>`<div class="ln"><span class="tg ${e.kind}">${tg[e.kind]}</span><span>${esc(e.text)}</span></div>`).join('')+
   '<div class="l" style="margin-top:8px"><b>POLICY</b> = selection rule · <b>LEARNED</b> = Q-value estimate · <b>OBSERVED</b> = direct simulator observation</div>';
}
function renderMath(){
  const m=S&&S.math;if(!m){Q('qbMath').innerHTML='<div class="qempty">No Q-value update yet.</div>';Q('qbUpd').innerHTML='';return}
  const br=m.target-m.q_old,rp=m.reward_parts||{};
  let parts='';
  if(m.trigger==='hop_forward')parts=`hop ${fq(rp.hop)} + queue ${fq(rp.queue)} (next router ${esc(rp.next_router)}, level ${rp.next_queue_level})`+(rp.priority_multiplier!==1?` × ${rp.priority_multiplier}`:'');
  else if(m.trigger==='delivered')parts=`delivery bonus ${fq(rp.delivery_bonus)}`;
  else parts=`drop penalty ${fq(rp.drop_penalty)} (${esc(rp.reason)})`;
  Q('qbMath').innerHTML=`<div class="l">Update #${m.seq} · t = ${m.t==null?'—':m.t} s · trigger: ${esc(m.trigger)} · packet #${m.packet_id}</div>
   <div class="f">${m.formula}</div>
   <div class="f">Q_new = Q_old + α × [ r + γ × maxQ(s′) − Q_old ]</div>
   <div class="f">= ${fq4(m.q_old)} + ${m.alpha} × [ ${fq4(m.reward)} + ${m.gamma} × ${fq4(m.max_next)} − ${fq4(m.q_old)} ]</div>
   <div class="f">= ${fq4(m.q_old)} + ${m.alpha} × ${fq4(br)} = <b>${fq4(m.recomputed)}</b></div>
   <table><tr><td>state s = (router, dest, queue level)</td><td>(${m.state.map(esc).join(', ')})</td></tr><tr><td>action a (next hop)</td><td>${esc(m.action)}</td></tr>
   <tr><td>previous Q(s,a)${m.first_visit?' <span class="l">(first visit, initial 0)</span>':''}</td><td>${fq4(m.q_old)}</td></tr><tr><td>learning rate α</td><td>${m.alpha}</td></tr>
   <tr><td>reward r = ${parts}</td><td>${fq4(m.reward)}</td></tr><tr><td>discount γ</td><td>${m.gamma}</td></tr>
   <tr><td>max Q(s′,a′) over next-state actions</td><td>${fq4(m.max_next)}</td></tr><tr><td>updated Q(s,a) written by the agent</td><td>${fq4(m.q_new)}</td></tr>
   <tr><td>Q-table value right now</td><td>${fq4(m.table_now)}</td></tr></table>
   <div style="margin-top:6px">${m.verified?'<span class="ok">✔ The formula reproduces the agent\'s stored value (tolerance 1e-9).</span>':'<span class="bad">✘ Mismatch between the formula and the stored value.</span>'}</div>
   ${m.terminal?`<div class="l" style="margin-top:4px">${esc(m.note)}</div>`:''}
   <div class="l" style="margin-top:4px">If the table value now differs from the written value, a later update changed it.</div>`;
  Q('qbUpd').innerHTML=S.updates.slice(0,25).map(u=>`<div data-u="${u.seq}" class="${u.seq===m.seq?'sel':''}">#${u.seq} t=${u.t==null?'—':u.t} ${esc(u.state[0])}→${esc(u.action)} ${fq(u.q_old)} → ${fq(u.q_new)} <span class="l">${esc(u.trigger)}</span></div>`).join('');
  Q('qbUpd').querySelectorAll('[data-u]').forEach(d=>d.onclick=()=>{ufocus=+d.dataset.u;poll(true)});
}
function renderTimeline(){
  if(!S){return}
  const only=Q('qbOnlyChg').checked,top=S.decisions.length?S.decisions[0].seq:0,fseq=S.focus?S.focus.decision.seq:null;
  const list=S.decisions.filter(d=>!only||d.changed);
  if(!list.length){Q('qbTl').innerHTML=`<div class="qempty">${S.decisions.length?'No route changes in the recent decisions.':'No decisions yet.'}</div>`;return}
  Q('qbTl').innerHTML=list.map(d=>{
    const qs=d.candidates.map(c=>`Q(${esc(d.router)},${esc(c.hop)}) = ${c.visited?fq(c.q):'Unvisited'}`).join(' · ');
    const nh=d.changed?`Previous next hop: <s class="mono">${esc(d.previous_hop)}</s> → New next hop: <b class="mono" style="color:var(--mint)">${esc(d.selected)}</b>`:`Next hop: <b class="mono">${esc(d.selected)}</b>${d.previous_hop==null?' <span class="l">(first at this router)</span>':' <span class="l">(unchanged)</span>'}`;
    return `<div class="tl ${d.changed?'chg':''} ${d.seq===fseq?'sel':''} ${d.seq>seenSeq&&seenSeq?'new':''}" data-s="${d.seq}">
     <b class="mono">T = ${d.t.toFixed(2)} s</b> | Router ${esc(d.router)} | Destination ${esc(d.dst)} <span class="bd ${d.mode==='explore'?'explored':'selected'}" style="float:right">${d.mode==='explore'?'explore':'greedy'}</span>
     <div class="l">packet #${d.packet_id} · path so far ${ph(d.path.concat([d.router]))}</div><div>${nh}</div><div class="l mono">${qs}</div><div class="l">Trigger: ${esc(d.trigger)}</div></div>`}).join('');
  seenSeq=top;
  Q('qbTl').querySelectorAll('[data-s]').forEach(d=>d.onclick=()=>{focus=+d.dataset.s;ufocus=null;poll(true)});
}
function renderEvents(){
  if(!S)return;
  Q('qbEvents').innerHTML=S.events.length?S.events.map(e=>`<div>${esc(e.text)}</div>`).join(''):'<div class="l">No network events yet.</div>';
  Q('qbCtlLog').innerHTML=S.controls.length?S.controls.map(c=>`<div>t=${c.t} s · ${esc(c.text)}</div>`).join(''):'<div class="l">No controls used yet.</div>';
}
function syncControls(){
  const ek=JSON.stringify(S.graph.edges);
  if(ek!==edgesKey){edgesKey=ek;const cur=Q('qbLink').value;Q('qbLink').innerHTML=S.graph.edges.map(e=>`<option value="${esc(e[0])}|${esc(e[1])}">${esc(e[0])} ↔ ${esc(e[1])}</option>`).join('');if(cur)Q('qbLink').value=cur;linkTouched=false}
  if(!linkTouched&&S.down_links.length){const d=S.down_links[0],o=[...Q('qbLink').options].find(x=>x.value===d[0]+'|'+d[1]||x.value===d[1]+'|'+d[0]);if(o)Q('qbLink').value=o.value}   // an outage is active: point Restore at it
  else if(!linkTouched&&S.routes[0]){const p=S.routes[0].current.path;if(p.length>1){const v=[p[0],p[1]],o=[...Q('qbLink').options].find(x=>x.value===v[0]+'|'+v[1]||x.value===v[1]+'|'+v[0]);if(o)Q('qbLink').value=o.value}}
  if(document.activeElement!==Q('qbExplore'))Q('qbExplore').checked=S.params.explore_enabled;
  if(document.activeElement!==Q('qbAlpha'))Q('qbAlpha').value=S.params.alpha;
  if(document.activeElement!==Q('qbGamma'))Q('qbGamma').value=S.params.gamma;
  const sp=Q('qbSpeed');if(document.activeElement!==sp)sp.value=String(S.speed);
}
// ---- full Q-table explorer (loaded on demand, never auto-refreshed) ----
async function loadFull(reset){
  if(reset)qfOffset=0;
  try{const q=new URLSearchParams({offset:qfOffset,limit:50});if(Q('qfRouter').value)q.set('router',Q('qfRouter').value);if(Q('qfDst').value)q.set('dst',Q('qfDst').value);
    const r=await lget('/api/live/qtable?'+q);
    const fill=(id,vals)=>{const cur=Q(id).value;Q(id).innerHTML=Q(id).options[0].outerHTML+vals.map(v=>`<option>${esc(v)}</option>`).join('');Q(id).value=cur};fill('qfRouter',r.routers);fill('qfDst',r.dsts);
    Q('qfInfo').textContent=r.total?`${qfOffset+1}–${Math.min(qfOffset+r.limit,r.total)} of ${r.total} entries`:'Q-table is empty';
    Q('qfTable').innerHTML=r.rows.length?`<table class="qt"><tr><th>Router</th><th>Dest</th><th>Queue lvl</th><th>Next hop</th><th style="text-align:right">Q-value</th><th>Link</th><th>Last update (s)</th></tr>`+
      r.rows.map(x=>`<tr><td class="mono">${esc(x.router)}</td><td>${esc(x.dst)}</td><td>${x.level}</td><td class="mono">${esc(x.hop)}</td><td class="num">${fq(x.q)}</td><td>${x.link_up===false?'<span class="bad">down</span>':'up'}</td><td>${x.last_update_t==null?'—':x.last_update_t}</td></tr>`).join('')+'</table>':'<div class="qempty">No learned entries match.</div>'}
  catch(e){Q('qfTable').innerHTML=`<div class="bad">${esc(e.message)}</div>`}
}
Q('qbFull').addEventListener('toggle',()=>{if(Q('qbFull').open)loadFull(true)});
Q('qfRefresh').onclick=()=>loadFull(false);Q('qfRouter').onchange=Q('qfDst').onchange=()=>loadFull(true);
Q('qfPrev').onclick=()=>{qfOffset=Math.max(0,qfOffset-50);loadFull(false)};Q('qfNext').onclick=()=>{qfOffset+=50;loadFull(false)};
poll(true);
})();
"""


def inject(page: str) -> str:
    """Splice the panel into the NeuroRouter page (CSS before </style>, HTML after the control box, JS last)."""
    css_marker = "</style></head><body>"
    html_marker = '<div id="err"></div></div>\n'
    js_marker = "</script></body></html>"
    for m in (css_marker, html_marker, js_marker):
        if page.count(m) != 1:
            raise RuntimeError(f"NeuroRouter page layout changed: marker {m!r} found {page.count(m)} times")
    page = page.replace(css_marker, LIVE_CSS + css_marker)
    page = page.replace(html_marker, html_marker + LIVE_HTML)
    return page.replace(js_marker, LIVE_JS + js_marker)
