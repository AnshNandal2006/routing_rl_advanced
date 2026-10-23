// Loads the real NeuroRouter page in jsdom against a running server and drives the live panel.
const { JSDOM } = require('jsdom');
const base = process.argv[2];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fails = [];
const check = (name, ok, extra='') => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (extra ? '  ' + extra : '')); if (!ok) fails.push(name); };
(async () => {
  const errors = [];
  const dom = await JSDOM.fromURL(base + '/', {
    runScripts: 'dangerously', pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = (u, o) => fetch(new URL(u, base), o);
      w.scrollTo = () => {}; w.HTMLElement.prototype.scrollIntoView = () => {};
      w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: () => () => ({ addColorStop(){} }), set: () => true });
      w.addEventListener('error', e => errors.push(String(e.error || e.message)));
      const ce = w.console.error; w.console.error = (...a) => { errors.push(a.join(' ')); };
    },
  });
  const w = dom.window, d = w.document, Q = id => d.getElementById(id);
  await sleep(600);
  // ---- empty states (nothing started)
  check('panel present', !!Q('qbrain') && /LIVE Q-LEARNING BRAIN/.test(Q('qbrain').textContent));
  check('empty: status chip says no session', Q('qbStatus').textContent === 'no session');
  check('empty: map/table placeholders', /Press Start/.test(Q('qbMap').textContent) && /No decisions yet/.test(Q('qbTable').textContent));
  check('original controls still there', ['run', 'battle', 'shuffle', 'method', 'topo'].every(i => !!Q(i)));
  check('method list intact (4 policies)', Q('method').options.length === 4);
  check('topology list populated', Q('topo').options.length >= 4, `(${Q('topo').options.length})`);
  // ---- step one decision from nothing (auto-creates the session)
  Q('qbStep').click(); await sleep(1500);
  check('session created by Step', /PAUSED|RUNNING/.test(Q('qbStatus').textContent), Q('qbStatus').textContent);
  check('table has rows after one step', Q('qbTable').querySelectorAll('tr').length >= 2);
  check('why panel has tagged lines', Q('qbWhy').querySelectorAll('.tg').length >= 2);
  check('math panel shows verified', /reproduces the agent/.test(Q('qbMath').textContent));
  check('timeline has one decision', Q('qbTl').querySelectorAll('.tl').length === 1);
  // ---- run for a while
  Q('qbSpeed').value = '25'; Q('qbSpeed').dispatchEvent(new w.Event('change')); Q('qbPlay').click(); await sleep(3500);
  check('running: status chip', /RUNNING/.test(Q('qbStatus').textContent), Q('qbStatus').textContent);
  check('timeline grows', Q('qbTl').querySelectorAll('.tl').length > 5, `(${Q('qbTl').querySelectorAll('.tl').length})`);
  check('KPI cards rendered', Q('qbKpi').querySelectorAll('.qk').length === 9);
  check('map drawn with edges', Q('qbMap').querySelectorAll('line').length > 5 && Q('qbMap').querySelectorAll('circle').length > 5);
  check('route comparison text', /Latest delivered route|No packet/.test(Q('qbRoute').textContent));
  Q('qbPause').click(); await sleep(1200);
  check('paused', /PAUSED/.test(Q('qbStatus').textContent));
  // ---- fail the link on current route
  const nOpts = Q('qbLink').options.length; check('link list populated', nOpts > 3, `(${nOpts})`);
  Q('qbDur').value = '10'; Q('qbFail').click(); await sleep(800);
  Q('qbStep').click(); await sleep(800); Q('qbStep').click(); await sleep(800);
  check('failed link shown red on map', Q('qbMap').querySelectorAll('line.down').length === 1);
  check('control log recorded outage', /Outage injected/.test(Q('qbCtlLog').textContent));
  check('sim event log shows FAILED', /FAILED/.test(Q('qbEvents').textContent), Q('qbEvents').textContent.slice(0, 60));
  const failedBefore = Q('qbMap').querySelectorAll('line.down').length;
  Q('qbRestore').click(); await sleep(700); Q('qbStep').click(); await sleep(900); Q('qbStep').click(); await sleep(900);
  check('restore accepted (banner hidden)', Q('qbMsg').style.display !== 'block', Q('qbMsg').style.display === 'block' ? Q('qbMsg').textContent : '');
  check('restore actually brought the link back', failedBefore === 1 && Q('qbMap').querySelectorAll('line.down').length === 0);
  check('sim event log shows RECOVERED', /RECOVERED/.test(Q('qbEvents').textContent));
  // ---- controls: exploration, params, traffic, bad inputs surface errors
  Q('qbExplore').checked = false; Q('qbExplore').dispatchEvent(new w.Event('change')); await sleep(900);
  check('exploration OFF shown in KPI', /OFF/.test(Q('qbKpi').textContent));
  Q('qbAlpha').value = '5'; Q('qbApply').click(); await sleep(700);
  check('invalid alpha shows an error', Q('qbMsg').style.display === 'block' && /α/.test(Q('qbMsg').textContent), Q('qbMsg').textContent);
  Q('qbAlpha').value = '0.3'; Q('qbGamma').value = '0.8'; Q('qbApply').click(); await sleep(900);
  check('valid alpha/gamma accepted and reflected', Q('qbMsg').style.display !== 'block' && Q('qbAlpha').value === '0.3');
  d.querySelector('[data-mult="3"]').click(); await sleep(700);
  check('traffic control logged', /×3/.test(Q('qbCtlLog').textContent));
  // ---- inspect a timeline card + an update
  const card = Q('qbTl').querySelectorAll('.tl')[3]; card.click(); await sleep(1000);
  check('click pins a decision', /pinned/.test(Q('qbFocusHead').textContent));
  Q('qbFollow').click(); await sleep(1000);
  check('follow latest unpins', !/pinned/.test(Q('qbFocusHead').textContent));
  const u = Q('qbUpd').querySelectorAll('[data-u]')[2]; u.click(); await sleep(900);
  check('update list click changes math panel', new RegExp('Update #' + u.dataset.u).test(Q('qbMath').textContent));
  // ---- reset behaviours
  Q('qbResetKeep').click(); await sleep(900);
  check('reset keep: message says PRESERVED', /PRESERVED/.test(Q('qbReset').textContent), Q('qbReset').textContent.slice(0, 80));
  check('after reset: no decisions yet in timeline', /No decisions yet/.test(Q('qbTl').textContent));
  check('after reset: math panel empty message', /No Q-value update yet/.test(Q('qbMath').textContent));
  Q('qbResetWipe').click(); await sleep(900);
  check('reset wipe: message says WIPED', /WIPED/.test(Q('qbReset').textContent));
  check('wiped: Q-table entries 0', /0 <small>/.test(Q('qbKpi').innerHTML) || /Q-table entries[\s\S]*?0/.test(Q('qbKpi').textContent), Q('qbKpi').textContent);
  // ---- full explorer on an empty table, then with data
  Q('qbFull').open = true; Q('qbFull').dispatchEvent(new w.Event('toggle')); await sleep(800);
  check('explorer handles empty table', /empty|No learned/.test(Q('qfInfo').textContent + Q('qfTable').textContent));
  Q('qbPlay').click(); await sleep(2500); Q('qbPause').click(); await sleep(800);
  Q('qfRefresh').click(); await sleep(800);
  check('explorer lists entries', Q('qfTable').querySelectorAll('tr').length > 2, Q('qfInfo').textContent);
  check('no JS errors during whole session', errors.length === 0, errors.slice(0, 3).join(' | '));
  console.log(fails.length ? `\n${fails.length} UI CHECK(S) FAILED` : '\nALL UI CHECKS PASSED');
  dom.window.close(); process.exit(fails.length ? 1 : 0);
})().catch(e => { console.error('HARNESS ERROR', e); process.exit(2); });
