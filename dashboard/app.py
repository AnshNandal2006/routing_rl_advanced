#!/usr/bin/env python3
"""
Adaptive Routing RL: Streamlit dashboard
========================================
Run:  streamlit run dashboard/app.py

How it works (read this before presenting):
  * This is a SIMULATION-FIRST dashboard. When you press Launch, the whole run executes to
    completion on the server; only then are the results drawn. The time slider REPLAYS the
    recorded history (one snapshot every 0.25 simulated seconds). It is not live packet streaming.
  * Every number comes from dashboard_core, the same module the NeuroRouter app and the
    saved experiment summaries use, so definitions never drift apart.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find dashboard_core however we are launched

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard_core import (
    LIMITS, METHODS, METHOD_NAMES, RUN_LOCK, SCORE_WEIGHTS, TOPOLOGY_CATALOG, TRAFFIC_TYPES,
    Scenario, ValidationError, describe_run, get_layout_positions, link_info, run_battle,
    run_scenario, to_jsonable, undirected_edges,
)

# ---------------------------------------------------------------------------
# Streamlit API compatibility: `use_container_width` is being replaced by `width="stretch"`
# ---------------------------------------------------------------------------
CHART_W = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}
BTN_W = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}

st.set_page_config(page_title="Adaptive Routing RL - Command Center", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
<style>
    .stApp { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
    h1, h2, h3 { color: #e0e0ff !important; }
    div[data-testid="stMetric"] { background:#16213e; border-radius:10px; padding:.6rem .9rem; border:1px solid #0f3460; }
    .stSelectbox label, .stSlider label, .stCheckbox label { color:#a0a0c0 !important; }
    .banner { background:#101a33; border:1px solid #2a3f7a; border-left:4px solid #3de8ff; border-radius:8px;
              padding:.6rem 1rem; color:#c8d0f0; font-size:.9rem; margin-bottom:.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: scenario designer
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.title("🛰️ Control Center")

sb.header("Routing policy")
method = sb.selectbox("Algorithm", METHODS, format_func=lambda m: METHOD_NAMES[m])

sb.header("Network")
topology = sb.selectbox("Topology", list(TOPOLOGY_CATALOG.keys()),
                        format_func=lambda x: f"{x}: {str(TOPOLOGY_CATALOG[x])[:40]}…")

sb.header("Traffic")
rate = sb.slider("Rate (pps)", LIMITS["rate"][0], LIMITS["rate"][1], 25.0, 2.5)
traffic_type = sb.selectbox("Pattern", list(TRAFFIC_TYPES))
priority_ratio = sb.slider("Priority traffic share", LIMITS["priority"][0], LIMITS["priority"][1], 0.12, 0.02)

sb.header("Simulation")
duration = sb.slider("Duration (s)", LIMITS["duration"][0], LIMITS["duration"][1], 70.0, 5.0)
lock_seed = sb.checkbox("Lock seed (reproducible topology)", value=True)
seed = sb.number_input("Seed", LIMITS["seed"][0], 9999, 42, disabled=not lock_seed)
sb.caption("Unlocked: every launch draws a new random geometric graph. The graph is frozen for the duration of a run.")

sb.header("Link failure")
enable_fail = sb.checkbox("Inject link failure", value=True)
fail_dur_max = float(min(LIMITS["fail_dur"][1], duration - 15))
fail_dur = sb.slider("Failure duration (s)", LIMITS["fail_dur"][0], fail_dur_max, min(15.0, fail_dur_max), 5.0,
                     disabled=not enable_fail)
fail_start_max = float(duration - fail_dur - 5)
fail_start = sb.slider("Failure start (s)", 5.0, fail_start_max, min(30.0, fail_start_max), 5.0,
                       disabled=not enable_fail)

run_btn = sb.button("🚀 Launch simulation", type="primary", **BTN_W)
battle_btn = sb.button("⚔️ Compare all 4 policies", **BTN_W)


def current_scenario() -> Scenario:
    return Scenario.from_payload(dict(
        method=method, topology=topology, rate=rate, traffic_type=traffic_type, priority=priority_ratio,
        duration=duration, seed=int(seed) if lock_seed else None, enable_fail=enable_fail,
        fail_start=fail_start, fail_dur=fail_dur)).with_seed()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def network_figure(state: dict, rr, title: str) -> go.Figure:
    sim, cfg = rr.sim, rr.cfg
    pos = get_layout_positions(cfg.network.topology, sim.routers)
    srcs = {str(s) for s in (cfg.traffic.sources or [])}
    dsts = {str(d) for d in (cfg.traffic.destinations or [])}
    fig = go.Figure()

    for (u, v) in undirected_edges(sim):
        info = link_info(state, u, v) or {"up": True, "util": 0}
        up, util = info.get("up", True), float(info.get("util", 0) or 0)
        (x0, y0), (x1, y1) = pos.get(u, (0, 0)), pos.get(v, (0, 0))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False, hoverinfo="text",
            text=f"{u}↔{v} | {'UP' if up else 'DOWN'} | utilisation {util:.2f}",
            line=dict(color="#2ecc71" if up else "#e74c3c", width=2 + 4 * util, dash="solid" if up else "dash")))

    xs, ys, cols, texts, sizes = [], [], [], [], []
    for rid, q in state["queues"].items():
        x, y = pos.get(rid, (0, 0))
        frac = min(1.0, q / max(1, sim.routers[rid].max_queue_size))
        cols.append(f"rgb({int(frac * 255)},{int((0.3 + 0.4 * (1 - frac)) * 255)},{int((1 - frac) * 255)})")
        role = " SRC" if str(rid) in srcs else " DST" if str(rid) in dsts else ""
        xs.append(x); ys.append(y); sizes.append(28 + 20 * frac)
        texts.append(f"{rid}{role}<br>Q:{q:.0f}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="middle center", showlegend=False, hoverinfo="text",
        marker=dict(size=sizes, color=cols, line=dict(width=2, color="white")),
        textfont=dict(color="white", size=11, family="Arial Black")))
    fig.update_layout(title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      margin=dict(l=10, r=10, t=40, b=10), height=420)
    return fig


def metrics_figure(ts: dict, up_to: float, window, t_end: float) -> go.Figure:
    t = np.array(ts["t"], dtype=float)
    mask = t <= up_to
    series = [("Average delay (s)", "avg_delay", "#e74c3c"), ("Delivery ratio (instantaneous)", "delivery_ratio_inst", "#2ecc71"),
              ("Total queue length (packets)", "queue_total", "#f39c12"), ("Throughput (inst.)", "throughput_inst", "#3498db")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[0] for s in series], vertical_spacing=0.14, horizontal_spacing=0.08)
    for n, (_, key, color) in enumerate(series):
        y = np.array(ts[key], dtype=float)[mask]  # None -> NaN -> gap in the line
        fig.add_trace(go.Scatter(x=t[mask], y=y, line=dict(color=color, width=2), connectgaps=False), row=n // 2 + 1, col=n % 2 + 1)
    if window:
        fig.add_vrect(x0=window[0], x1=window[1], fillcolor="#e74c3c", opacity=0.14, line_width=0, row="all", col="all")
    fig.add_vline(x=up_to, line_dash="dot", line_color="#9aa4d8", row="all", col="all")
    fig.update_xaxes(range=[0, t_end], gridcolor="#333", title_text="")
    fig.update_yaxes(gridcolor="#333")
    fig.update_layout(height=460, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,33,62,0.4)",
                      font=dict(color="#ccc"), margin=dict(l=40, r=20, t=40, b=30))
    return fig


def kpi_grid(kpis: list) -> None:
    for i in range(0, len(kpis), 5):
        cols = st.columns(5)
        for c, k in zip(cols, kpis[i:i + 5]):
            c.metric(k["label"], f'{k["value"]}{" " + k["unit"] if k["unit"] and k["value"] not in ("n/a", "none") else ""}',
                     help=k["note"] or None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🛰️ Adaptive Network Routing: Deep RL Command Center")
st.caption("Discrete-time network simulator · Double Dueling DQN vs. classical baselines · scheduled link failures · QoS traffic")
st.markdown(
    '<div class="banner"><b>How to read this page.</b> The simulation runs to completion first; the results below are '
    'then <i>replayed</i> from recorded history. This is not live packet streaming and not a real network. '
    'Red-shaded areas on the charts mark the injected link failure.</div>', unsafe_allow_html=True)

st.session_state.setdefault("run", None)
st.session_state.setdefault("battle", None)

if run_btn or battle_btn:
    try:
        sc = current_scenario()
    except ValidationError as e:
        st.error(f"Invalid scenario: {e}")
        st.stop()
    try:
        with RUN_LOCK:
            if run_btn:
                with st.spinner(f"Running {METHOD_NAMES[sc.method]} to completion…"):
                    rr = run_scenario(sc)
                st.session_state.run = {"rr": rr, "d": describe_run(rr)}
                st.session_state.battle = None
            else:
                with st.spinner("Running all four policies on the same seed, topology and failure schedule…"):
                    sc2, rows = run_battle(sc)
                st.session_state.battle = {"sc": sc2, "rows": rows}
                st.session_state.run = None
    except Exception as e:  # surface simulator errors instead of a raw traceback page
        st.error(f"Simulation failed: {type(e).__name__}: {e}")
        st.stop()

run, battle = st.session_state.run, st.session_state.battle

if battle:
    sc2, rows = battle["sc"], battle["rows"]
    st.subheader("⚔️ Policy comparison")
    st.caption(f"{rows[0]['n_nodes']} routers · seed {sc2.seed} · same topology, failure schedule and traffic settings for all four. "
               "The DQN trains online during its run, so its score includes its own learning period.")
    table = []
    for rank, r in enumerate(rows, 1):
        km = {k["key"]: k["raw"] for k in r["kpis"]}
        table.append({"Rank": rank, "Policy": r["name"], "Score": round(r["score"], 1),
                      "Delivery %": None if km["delivery"] is None else round(km["delivery"] * 100, 2),
                      "Loss %": None if km["loss"] is None else round(km["loss"] * 100, 2),
                      "Avg delay (ms)": None if km["delay"] is None else round(km["delay"] * 1000),
                      "P95 delay (ms)": None if km["p95"] is None else round(km["p95"] * 1000),
                      "Route changes": km["route_changes"], "Adaptation (s)": None if km["adaptation"] is None else round(km["adaptation"], 1)})
    df = pd.DataFrame(table)
    st.dataframe(df, hide_index=True, **({"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}))
    fig = px.bar(df, x="Policy", y="Score", color="Policy", title="Composite score (0-100)")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,33,62,0.4)", font=dict(color="#ccc"), height=320)
    st.plotly_chart(fig, **CHART_W)
    with st.expander("How the composite score is defined"):
        st.write(f"Score = 100 × ({SCORE_WEIGHTS['delivery']:.0%} delivery ratio + {SCORE_WEIGHTS['delay']:.0%} delay credit "
                 f"+ {SCORE_WEIGHTS['loss']:.0%} (1 − loss) + {SCORE_WEIGHTS['priority']:.0%} priority delivery). "
                 "Delay credit falls linearly from 1 to 0 as average delay goes from 0 to 2 s.")
elif run is None:
    st.info("👈 Pick a scenario in the sidebar, then **Launch simulation** (one policy) or **Compare all 4 policies**.")
    st.markdown("""
| Feature | What it does |
|---|---|
| **Double Dueling DQN** | Value + advantage streams, Double-DQN targets, experience replay, ε-decay |
| **Baselines** | Shortest path, congestion-aware heuristic, tabular Q-learning |
| **Dynamic events** | A scheduled link failure with recovery, so you can measure adaptation |
| **QoS classes** | Priority vs. best-effort packets |
| **Replay slider** | Scrub the recorded history of the finished run |
""")
else:
    rr, d = run["rr"], run["d"]
    sim, cfg = rr.sim, rr.cfg

    kpi_grid(d["kpis"])
    st.caption(f"{cfg.network.topology} · {d['n_nodes']} routers · {d['n_links']} directed links · seed {d['seed']} · "
               f"{METHOD_NAMES[rr.method]} · simulated in {d['seconds']} s of wall-clock time")
    st.markdown("---")

    times = [h["t"] for h in sim.history]
    selected_t = st.select_slider("⏱️ Replay: scrub through the recorded history", options=times, value=times[-1],
                                  format_func=lambda x: f"{x:.2f} s")
    idx = min(range(len(times)), key=lambda i: abs(times[i] - selected_t))
    t_end = float(cfg.simulation.duration)

    col_net, col_met = st.columns([1, 1.15])
    with col_net:
        st.plotly_chart(network_figure(sim.history[idx], rr, f"Network state at t = {selected_t:.2f} s"), **CHART_W)
    with col_met:
        st.plotly_chart(metrics_figure(rr.ts, selected_t, d["window"], t_end), **CHART_W)

    pstats = d["pstats"]
    if pstats:
        st.subheader("🧠 Learning diagnostics")
        c = st.columns(4)
        c[0].metric("ε (exploration)", f"{pstats.get('epsilon', 0):.3f}")
        c[1].metric("Replay buffer", f"{pstats.get('buffer_size', 0):,.0f}")
        c[2].metric("Train steps", f"{pstats.get('train_steps', 0):,.0f}")
        c[3].metric("Avg loss (recent)", f"{pstats.get('avg_loss', 0):.4f}")
        hist = getattr(sim.policy, "loss_history", None)
        if rr.method == "dqn" and hist:
            fig = px.line(pd.DataFrame({"step": range(len(hist)), "loss": list(hist)}), x="step", y="loss", title="DQN training loss")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,33,62,0.4)", font=dict(color="#ccc"), height=280)
            st.plotly_chart(fig, **CHART_W)

    with st.expander(f"📋 Event history ({len(d['events'])} recorded transitions)", expanded=bool(d["events"])):
        st.caption("Derived from the recorded link and queue state: link down/up and routers crossing 70% queue occupancy.")
        if d["events"]:
            st.dataframe(pd.DataFrame(d["events"]).rename(columns={"t": "time (s)", "level": "type", "text": "event"}),
                         hide_index=True)
        else:
            st.write("No overloads or link changes occurred in this run.")
        if d["raw_log"]:
            st.text("\n".join(d["raw_log"]))

    with st.expander("📊 Full numerical summary"):
        st.json(d["summary"])
        st.download_button("Download this run as JSON", json.dumps(to_jsonable({"scenario": rr.scenario.__dict__, "kpis": d["kpis"], "summary": d["summary"]}), indent=1),
                           file_name=f"run_{rr.method}_{rr.scenario.seed}.json", mime="application/json")