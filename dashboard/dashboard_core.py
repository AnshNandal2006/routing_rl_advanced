#!/usr/bin/env python3
"""
Shared core for both dashboards (Streamlit and NeuroRouter).

Everything that defines *what a number means* lives here, so the two UIs can
never disagree with each other (or with the saved experiment summaries):

  * input validation and hard limits       -> Scenario.from_payload
  * config construction                    -> build_config
  * running one policy / all four policies -> run_scenario / run_battle
  * KPI definitions and formatting         -> build_kpis
  * composite score                        -> composite_score
  * adaptation-time estimate               -> estimate_adaptation
  * event history from recorded link/queue state -> derive_events

This is a SIMULATION. Nothing here touches a real network.

CLI (handy for the README / reproducing a run without any UI):
    python dashboard/dashboard_core.py --method dqn --topology <name> --seed 42
"""
from __future__ import annotations

import math
import random
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from routing_rl.config import Config
from routing_rl.network.topology import TOPOLOGY_CATALOG, build_topology, get_layout_positions, traffic_defaults
from routing_rl.simulation.simulator import Simulator

__all__ = [
    "METHODS", "METHOD_NAMES", "TRAFFIC_TYPES", "LIMITS", "ValidationError", "Scenario", "RunResult",
    "RUN_LOCK", "build_config", "run_scenario", "run_battle", "describe_run", "build_kpis",
    "composite_score", "estimate_adaptation", "derive_events", "to_jsonable", "failure_window",
    "get_layout_positions", "TOPOLOGY_CATALOG", "undirected_edges", "link_info", "SCORE_WEIGHTS",
]

# ---------------------------------------------------------------------------
# Constants and hard limits
# ---------------------------------------------------------------------------
METHODS = ["dqn", "tabular_q", "congestion_aware", "shortest_path"]
METHOD_NAMES = {
    "dqn": "Double Dueling DQN",
    "tabular_q": "Tabular Q-learning",
    "congestion_aware": "Congestion-aware heuristic",
    "shortest_path": "Shortest path (baseline)",
}
TRAFFIC_TYPES = ("poisson", "constant", "bursty")

LIMITS = {
    "rate": (1.0, 60.0),        # packets per second
    "duration": (30.0, 120.0),  # simulated seconds
    "priority": (0.0, 0.40),    # fraction of priority packets
    "fail_dur": (5.0, 30.0),    # outage length, simulated seconds
    "seed": (1, 2**31 - 1),
}
MAX_NODES = 30            # scenario-size cap (random geometric graphs are 10-30 nodes)
MAX_PACKETS = 7200        # rate * duration safety cap (= 60 pps * 120 s)
RECOVERY_MARGIN = 5.0     # failure must end this long before the run does

# One simulation at a time per process: the simulator and DQN are CPU heavy and
# the web server / Streamlit may serve several browser sessions concurrently.
RUN_LOCK = threading.Lock()

# Weights of the composite score (documented in README, shown in the UIs).
SCORE_WEIGHTS = {"delivery": 0.40, "delay": 0.25, "loss": 0.20, "priority": 0.15}
DELAY_SATURATION_S = 2.0  # an average delay of >= 2 s earns zero delay credit


class ValidationError(ValueError):
    """Raised for any user-supplied value that is missing, malformed or out of range."""


# ---------------------------------------------------------------------------
# Scenario (validated user input)
# ---------------------------------------------------------------------------
def _num(p: dict, name: str, lo: float, hi: float, default: float) -> float:
    v = p.get(name, default)
    if isinstance(v, bool) or v is None:
        raise ValidationError(f"{name} must be a number")
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number (got {v!r})")
    if not math.isfinite(v):
        raise ValidationError(f"{name} must be finite")
    if not lo <= v <= hi:
        raise ValidationError(f"{name} must be between {lo:g} and {hi:g} (got {v:g})")
    return v


@dataclass(frozen=True)
class Scenario:
    method: str = "dqn"
    topology: str = ""
    rate: float = 25.0
    traffic_type: str = "poisson"
    priority: float = 0.12
    duration: float = 60.0
    seed: Optional[int] = 42          # None -> a fresh seed is drawn once per launch
    enable_fail: bool = True
    fail_start: float = 28.0
    fail_dur: float = 15.0

    @classmethod
    def from_payload(cls, p: Dict[str, Any]) -> "Scenario":
        if not isinstance(p, dict):
            raise ValidationError("request body must be a JSON object")
        method = p.get("method", "dqn")
        if method not in METHODS:
            raise ValidationError(f"method must be one of {METHODS}")
        topology = p.get("topology")
        if topology not in TOPOLOGY_CATALOG:
            raise ValidationError(f"topology must be one of {list(TOPOLOGY_CATALOG)}")
        traffic = p.get("traffic_type", "poisson")
        if traffic not in TRAFFIC_TYPES:
            raise ValidationError(f"traffic_type must be one of {list(TRAFFIC_TYPES)}")

        rate = _num(p, "rate", *LIMITS["rate"], 25.0)
        duration = _num(p, "duration", *LIMITS["duration"], 60.0)
        priority = _num(p, "priority", *LIMITS["priority"], 0.12)
        if rate * duration > MAX_PACKETS:
            raise ValidationError(
                f"scenario too large: rate x duration = {rate * duration:.0f} packets (limit {MAX_PACKETS})")

        seed = p.get("seed")
        if seed is not None:
            if isinstance(seed, bool):
                raise ValidationError("seed must be an integer or null")
            try:
                f = float(seed)
                if f != int(f):
                    raise ValueError
                seed = int(f)
            except (TypeError, ValueError):
                raise ValidationError("seed must be an integer or null")
            if not LIMITS["seed"][0] <= seed <= LIMITS["seed"][1]:
                raise ValidationError(f"seed must be between {LIMITS['seed'][0]} and {LIMITS['seed'][1]}")

        ef = p.get("enable_fail", True)
        if ef not in (True, False, 0, 1):
            raise ValidationError("enable_fail must be true or false")
        enable_fail = bool(ef)

        fail_dur = _num(p, "fail_dur", *LIMITS["fail_dur"], 15.0)
        fail_start = _num(p, "fail_start", 1.0, duration, 28.0)
        if enable_fail and fail_start + fail_dur > duration - RECOVERY_MARGIN:
            raise ValidationError(
                f"failure window {fail_start:g}-{fail_start + fail_dur:g} s must end at least "
                f"{RECOVERY_MARGIN:g} s before the run ends ({duration:g} s) so recovery can be observed")

        return cls(method, topology, rate, traffic, priority, duration, seed, enable_fail, fail_start, fail_dur)

    def with_seed(self) -> "Scenario":
        """Draw a seed ONCE per launch so every policy in a battle sees one frozen topology."""
        if self.seed is not None:
            return self
        return replace(self, seed=random.SystemRandom().randint(*LIMITS["seed"]))


def failure_window(sc: Scenario) -> Optional[Tuple[float, float]]:
    return (sc.fail_start, sc.fail_start + sc.fail_dur) if sc.enable_fail else None


# ---------------------------------------------------------------------------
# Config construction and running
# ---------------------------------------------------------------------------
def build_config(sc: Scenario, method: Optional[str] = None) -> Config:
    if sc.seed is None:
        raise ValueError("call Scenario.with_seed() before build_config")
    method = method or sc.method
    cfg = Config.from_yaml(ROOT / "config" / "default.yaml")
    cfg.routing.method = method
    cfg.network.topology = sc.topology
    cfg.traffic.rate_pps = sc.rate
    cfg.traffic.type = sc.traffic_type
    cfg.traffic.priority_ratio = sc.priority
    cfg.simulation.duration = sc.duration
    cfg.simulation.seed = int(sc.seed)
    cfg.network.topology_seed = int(sc.seed)
    cfg.network.enable_failures = sc.enable_fail
    # scenario-size cap
    cfg.network.n_nodes_max = min(int(cfg.network.n_nodes_max), MAX_NODES)
    cfg.network.n_nodes_min = min(int(cfg.network.n_nodes_min), int(cfg.network.n_nodes_max))

    routers, _, _ = build_topology(
        sc.topology,
        queue_size=cfg.network.default_queue_size,
        capacity=cfg.network.default_capacity,
        prop_delay=cfg.network.propagation_delay,
        seed=cfg.network.topology_seed,
        n_nodes_min=cfg.network.n_nodes_min,
        n_nodes_max=cfg.network.n_nodes_max,
    )
    sources, destinations, fail_link = traffic_defaults(sc.topology, routers)
    cfg.traffic.sources, cfg.traffic.destinations = sources, destinations
    win = failure_window(sc)
    cfg.network.failure_schedule = (
        [{"t_start": win[0], "t_end": win[1], "link": fail_link}] if win else [])
    return cfg


@dataclass
class RunResult:
    scenario: Scenario
    method: str
    cfg: Any
    sim: Any
    summary: Dict[str, Any]
    ts: Dict[str, List[Any]]
    seconds: float

    @property
    def n_nodes(self) -> int:
        return len(self.sim.routers)


def run_scenario(sc: Scenario, method: Optional[str] = None) -> RunResult:
    """Run one policy to completion. Callers hold RUN_LOCK."""
    sc = sc.with_seed()
    method = method or sc.method
    cfg = build_config(sc, method)
    t0 = time.perf_counter()
    sim = Simulator(cfg, train_rl=(method == "dqn"))
    metrics = sim.run(progress=False)
    if not getattr(sim, "history", None):
        raise RuntimeError("simulation produced no recorded history")
    return RunResult(sc, method, cfg, sim, dict(metrics.summary()), clean_series(metrics.time_series),
                     time.perf_counter() - t0)


def run_battle(sc: Scenario) -> Tuple[Scenario, List[Dict[str, Any]]]:
    """All four policies on the same seed / topology / failure schedule, ranked by composite score."""
    sc = sc.with_seed()
    rows = []
    for m in METHODS:
        rr = run_scenario(sc, m)
        d = describe_run(rr, light=True)
        rows.append({"method": m, "name": METHOD_NAMES[m], "score": d["score"], "kpis": d["kpis"],
                     "n_nodes": d["n_nodes"], "seconds": rr.seconds})
    rows.sort(key=lambda r: -r["score"])
    return sc, rows


# ---------------------------------------------------------------------------
# Metric definitions (single source of truth)
# ---------------------------------------------------------------------------
# The metrics collector may name a statistic differently across versions; these
# aliases are tried in order. Add the real key here if yours differs.
ALIASES = {
    "delivery_ratio": ("delivery_ratio", "pdr"),
    "loss_ratio": ("loss_ratio", "packet_loss_ratio", "drop_ratio"),
    "avg_delay": ("avg_delay", "mean_delay"),
    "p95_delay": ("p95_delay", "delay_p95"),
    "avg_hops": ("avg_hops", "mean_hops"),
    "priority_delivery_ratio": ("priority_delivery_ratio", "priority_pdr"),
    "route_changes": ("route_changes", "n_route_changes", "path_changes", "route_change_count"),
    "adaptation_time": ("adaptation_time", "adaptation_time_s", "recovery_time"),
}


def pick(summary: Dict[str, Any], key: str) -> Optional[float]:
    for name in ALIASES[key]:
        v = summary.get(name)
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def composite_score(summary: Dict[str, Any]) -> float:
    """0-100. 40% delivery + 25% delay (linear, 0 at >=2 s) + 20% (1-loss) + 15% priority delivery."""
    dr = pick(summary, "delivery_ratio") or 0.0
    delay = pick(summary, "avg_delay")
    loss = pick(summary, "loss_ratio")
    prio = pick(summary, "priority_delivery_ratio")
    w = SCORE_WEIGHTS
    s = (w["delivery"] * dr
         + w["delay"] * (1 - min((delay if delay is not None else DELAY_SATURATION_S) / DELAY_SATURATION_S, 1))
         + w["loss"] * (1 - (loss if loss is not None else 1.0))
         + w["priority"] * (prio if prio is not None else dr))
    return float(max(0.0, min(100.0, 100 * s)))


def estimate_adaptation(ts: Dict[str, List[Any]], t_fail: float, threshold: float = 0.9,
                        baseline_window: float = 10.0, hold: int = 3) -> Tuple[Optional[float], str]:
    """
    Seconds from failure onset until instantaneous delivery ratio is back to >= 90% of its
    pre-failure mean (held for 3 consecutive samples). 0.0 means delivery never dropped below
    that level; None means it could not be measured / never recovered.
    Used only when the simulator's own summary does not report adaptation time.
    """
    t = np.asarray(ts.get("t") or [], dtype=float)
    y = np.asarray(ts.get("delivery_ratio_inst") or [], dtype=float)
    if t.size == 0 or y.size != t.size:
        return None, "no time series"
    ok = np.isfinite(y)
    pre = ok & (t < t_fail) & (t >= t_fail - baseline_window)
    if pre.sum() < 3:
        return None, "too little pre-failure data"
    base = float(np.mean(y[pre]))
    if base <= 0:
        return None, "pre-failure delivery was zero"
    level = threshold * base
    post = np.where(ok & (t >= t_fail))[0]
    below = [i for i in post if y[i] < level]
    if not below:
        return 0.0, "delivery never fell below 90% of baseline"
    cand = [i for i in post if i > below[0]]
    for n, i in enumerate(cand):
        w = cand[n:n + hold]
        if len(w) == hold and all(y[j] >= level for j in w):
            return float(t[i] - t_fail), "estimated from delivery-ratio time series"
    return None, "did not recover before the run ended"


def _tone(v: Optional[float], good: float, warn: float, higher_is_better: bool = True) -> str:
    if v is None:
        return "ion"
    if higher_is_better:
        return "good" if v >= good else "warn" if v >= warn else "bad"
    return "good" if v <= good else "warn" if v <= warn else "bad"


def build_kpis(summary: Dict[str, Any], ts: Dict[str, List[Any]], sc: Scenario, n_nodes: int,
               score: Optional[float] = None) -> List[Dict[str, Any]]:
    """Ordered KPI cards. Both dashboards render exactly this list."""
    dr, loss = pick(summary, "delivery_ratio"), pick(summary, "loss_ratio")
    delay, p95, hops = pick(summary, "avg_delay"), pick(summary, "p95_delay"), pick(summary, "avg_hops")
    rc = pick(summary, "route_changes")
    score = composite_score(summary) if score is None else score

    def k(key, label, raw, fmt, unit="", pct=0.0, tone="ion", note=""):
        return {"key": key, "label": label, "raw": raw, "unit": unit, "tone": tone, "note": note,
                "pct": float(max(3, min(100, pct))),
                "value": fmt(raw) if raw is not None else "n/a"}

    # adaptation time: prefer the simulator's own number, else estimate from the time series
    win = failure_window(sc)
    if win is None:
        ad_raw, ad_note, ad_val = None, "No link failure was injected in this run.", "none"
    else:
        ad_raw = pick(summary, "adaptation_time")
        if ad_raw is not None:
            ad_note, ad_val = "Reported by the simulator's metrics collector.", None
        else:
            ad_raw, why = estimate_adaptation(ts, win[0])
            ad_note, ad_val = why.capitalize() + ".", None
    ad_card = k("adaptation", "Adaptation time", ad_raw, lambda v: f"{v:.1f}", "s",
                100 - (ad_raw or 0) * 5 if ad_raw is not None else 3,
                _tone(ad_raw, 3, 10, False), ad_note + " Time from failure onset until delivery recovers to 90% of its pre-failure level.")
    if ad_val:
        ad_card["value"] = ad_val

    return [
        k("nodes", "Routers", float(n_nodes), lambda v: f"{int(v)}", "", n_nodes / MAX_NODES * 100, "ion",
          "Nodes in the frozen topology for this run."),
        k("delivery", "Delivery ratio", dr, lambda v: f"{v * 100:.1f}", "%", (dr or 0) * 100,
          _tone(dr, .9, .6), "Packets delivered / packets generated."),
        k("loss", "Packet loss", loss, lambda v: f"{v * 100:.2f}", "%", 100 - (loss or 0) * 500,
          _tone(loss, .05, .2, False), "Packets dropped (queue overflow, TTL expiry, dead link) / packets generated. "
          "Packets still in flight at the end of the run count as neither delivered nor lost."),
        k("delay", "Average delay", delay, lambda v: f"{v * 1000:.0f}", "ms", 100 - (delay or 0) * 50, "ion",
          "Mean end-to-end delay of delivered packets."),
        k("p95", "P95 delay", p95, lambda v: f"{v * 1000:.0f}", "ms", 100 - (p95 or 0) * 33, "ion",
          "95th-percentile end-to-end delay of delivered packets."),
        k("hops", "Average hops", hops, lambda v: f"{v:.2f}", "", 100 - abs((hops or 0) - 2) * 20, "ion",
          "Mean hop count of delivered packets."),
        k("route_changes", "Route changes", rc, lambda v: f"{int(round(v))}", "", 100 - (rc or 0) * 2, "ion",
          "How often a flow's next-hop choice changed." if rc is not None else
          "Not reported by the simulator's metrics collector; add 'route_changes' to Metrics.summary() "
          "(or its real key name to ALIASES in dashboard_core.py)."),
        ad_card,
        k("score", "Composite score", score, lambda v: f"{v:.0f}", "/100", score, "ion",
          "40% delivery + 25% delay + 20% (1 - loss) + 15% priority delivery. Weights in dashboard_core.SCORE_WEIGHTS."),
    ]


# ---------------------------------------------------------------------------
# Recorded-history helpers (events, edges)
# ---------------------------------------------------------------------------
def undirected_edges(sim) -> List[Tuple[str, str]]:
    out, seen = [], set()
    for (u, v) in sim.links:
        key = tuple(sorted((str(u), str(v))))
        if key not in seen:
            seen.add(key)
            out.append((u, v))
    return out


def link_info(h: Dict, u, v) -> Dict:
    return h["links"].get(f"{u}->{v}") or h["links"].get(f"{v}->{u}") or {}


def derive_events(sim, overload: float = 0.7, keep: int = 80) -> List[Dict[str, Any]]:
    """Real state transitions found in the recorded history (not decoration)."""
    ids, edges = sorted(sim.routers), undirected_edges(sim)
    ev, hot0, down0 = [], set(), set()
    for h in sim.history:
        t = round(float(h["t"]), 2)
        hot = {i for i in ids if h["queues"].get(i, 0) / max(1, sim.routers[i].max_queue_size) >= overload}
        down = {(u, v) for u, v in edges if not link_info(h, u, v).get("up", True)}
        ev += [{"t": t, "level": "warn", "text": f"Router {i} overloaded (queue >= {overload:.0%})"} for i in sorted(hot - hot0)]
        ev += [{"t": t, "level": "good", "text": f"Router {i} back to normal"} for i in sorted(hot0 - hot)]
        ev += [{"t": t, "level": "bad", "text": f"Link {u}-{v} went down"} for u, v in sorted(down - down0)]
        ev += [{"t": t, "level": "ion", "text": f"Link {u}-{v} restored"} for u, v in sorted(down0 - down)]
        hot0, down0 = hot, down
    return ev[-keep:]


def clean_series(ts: Dict[str, Any]) -> Dict[str, List[Any]]:
    return {k: to_jsonable(list(v)) if hasattr(v, "__iter__") and not isinstance(v, (str, dict)) else v
            for k, v in dict(ts).items()}


def to_jsonable(o: Any) -> Any:
    """JSON-safe copy: numpy -> python, NaN/inf -> None (browsers reject bare NaN)."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        return to_jsonable(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if o is None or isinstance(o, (int, str, bool)):
        return o
    return str(o)


def describe_run(rr: RunResult, light: bool = False) -> Dict[str, Any]:
    """Everything a UI needs about a finished run, computed once, with shared definitions."""
    score = composite_score(rr.summary)
    out = {
        "method": rr.method,
        "name": METHOD_NAMES[rr.method],
        "summary": to_jsonable(rr.summary),
        "score": score,
        "kpis": build_kpis(rr.summary, rr.ts, rr.scenario, rr.n_nodes, score),
        "n_nodes": rr.n_nodes,
        "n_links": len(rr.sim.links),
        "seed": rr.scenario.seed,
        "seconds": round(rr.seconds, 2),
        "window": failure_window(rr.scenario),
    }
    if light:
        return out
    stats = rr.sim.policy.get_stats() if hasattr(rr.sim.policy, "get_stats") else {}
    out.update({
        "ts": rr.ts,
        "events": derive_events(rr.sim),
        "raw_log": [str(m) for m in (getattr(rr.sim, "event_log", None) or [])][-200:],
        "pstats": {k: v for k, v in (stats or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)},
    })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run one routing policy and print its metrics (no UI).")
    ap.add_argument("--method", choices=METHODS, default="dqn")
    ap.add_argument("--topology", default=next(iter(TOPOLOGY_CATALOG)), choices=list(TOPOLOGY_CATALOG))
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--traffic", default="poisson", choices=TRAFFIC_TYPES)
    ap.add_argument("--priority", type=float, default=0.12)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-failure", action="store_true")
    ap.add_argument("--fail-start", type=float, default=28.0)
    ap.add_argument("--fail-dur", type=float, default=15.0)
    ap.add_argument("--battle", action="store_true", help="run all four policies")
    a = ap.parse_args()
    try:
        sc = Scenario.from_payload(dict(method=a.method, topology=a.topology, rate=a.rate, traffic_type=a.traffic,
                                        priority=a.priority, duration=a.duration, seed=a.seed,
                                        enable_fail=not a.no_failure, fail_start=a.fail_start, fail_dur=a.fail_dur))
    except ValidationError as e:
        ap.error(str(e))
    results = []
    if a.battle:
        sc, rows = run_battle(sc)
        results = rows
    else:
        results = [describe_run(run_scenario(sc), light=True)]
    for r in results:
        print(f"\n== {r.get('name')}  (score {r['score']:.1f}, {r['n_nodes']} nodes)")
        for k in r["kpis"]:
            print(f"  {k['label']:<18} {k['value']} {k['unit']}")


if __name__ == "__main__":
    _cli()
