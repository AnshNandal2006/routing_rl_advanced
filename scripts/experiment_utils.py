"""Shared helpers for main.py, scripts/run_comparison.py and the tests.

* dotted-path config overrides that fail loudly on typos,
* one function (run_one) that runs a single method and returns everything worth saving,
* statistics helpers (mean, std, 95% CI, paired differences),
* run metadata so every result file says how it was produced.
"""
from __future__ import annotations
import math, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from routing_rl.config import Config

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("shortest_path", "congestion_aware", "tabular_q", "dqn")
LABELS = {"shortest_path": "Shortest path", "congestion_aware": "Congestion-aware",
          "tabular_q": "Tabular Q-learning", "dqn": "Double Dueling DQN"}
COLORS = {"shortest_path": "#e74c3c", "congestion_aware": "#3498db",
          "tabular_q": "#9b59b6", "dqn": "#2ecc71"}
# key, axis label, which direction is better
METRICS: List[Tuple[str, str, str]] = [
    ("delivery_ratio", "Delivery ratio", "higher"),
    ("loss_ratio", "Packet loss ratio", "lower"),
    ("avg_delay", "Average delay (s)", "lower"),
    ("p95_delay", "P95 delay (s)", "lower"),
    ("throughput_pps", "Throughput (pkt/s)", "higher"),
    ("avg_hops", "Average hops", "lower"),
    ("route_changes_per_delivered", "Route changes per delivered packet", "lower"),
    ("adaptation_time_mean", "Mean adaptation time (s)", "lower"),
    ("priority_delivery_ratio", "Priority delivery ratio", "higher"),
]


def json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------------------------------------------------------- config overrides
def parse_override(text: str) -> Tuple[str, Any]:
    """'traffic.rate_pps=40' -> ('traffic.rate_pps', 40.0). Values are parsed as YAML."""
    if "=" not in text:
        raise ValueError(f"Override {text!r} must look like section.field=value")
    path, raw = text.split("=", 1)
    return path.strip(), yaml.safe_load(raw)


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if isinstance(cur, dict):
            if p not in cur:
                raise ValueError(f"Unknown config field {path!r} (no {p!r})")
            cur = cur[p]
        elif hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            raise ValueError(f"Unknown config field {path!r} (no {p!r})")
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    elif hasattr(cur, last):
        setattr(cur, last, value)
    else:
        raise ValueError(f"Unknown config field {path!r}")


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    for path, value in (overrides or {}).items():
        set_path(cfg, path, value)
    return cfg


def clone_cfg(cfg: Config) -> Config:
    return Config.from_dict(cfg.to_dict())


def resolve_cfg(cfg: Config) -> Config:
    """Make endpoints and the failure link valid for the chosen topology.

    * traffic sources/destinations missing from the topology are replaced by its defaults;
    * a failure-schedule link written as "auto" becomes the topology's representative link.
    """
    from routing_rl.network.topology import build_topology, traffic_defaults
    seed = cfg.network.topology_seed if cfg.network.topology_seed is not None else cfg.simulation.seed
    routers, _, _ = build_topology(
        cfg.network.topology, queue_size=cfg.network.default_queue_size, capacity=cfg.network.default_capacity,
        prop_delay=cfg.network.propagation_delay, seed=int(seed),
        n_nodes_min=cfg.network.n_nodes_min, n_nodes_max=cfg.network.n_nodes_max)
    srcs, dsts, fail = traffic_defaults(cfg.network.topology, routers)
    t = cfg.traffic
    if not t.sources or not t.destinations or any(x not in routers for x in list(t.sources) + list(t.destinations)):
        t.sources, t.destinations = list(srcs), list(dsts)
    cfg.network.failure_schedule = [
        {**ev, "link": list(fail)} if ev.get("link") == "auto" else ev for ev in (cfg.network.failure_schedule or [])]
    return cfg


# ---------------------------------------------------------------- single run
def run_one(cfg: Config, method: str, *, progress: bool = False, strict_accounting: bool = False,
            keep_timeseries: bool = True, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run one routing method on an unmodified copy of cfg. Raises on failure (callers report it)."""
    from routing_rl.simulation.simulator import Simulator
    cfg = clone_cfg(cfg)
    cfg.routing.method = method
    t0 = time.perf_counter()
    sim = Simulator(cfg, train_rl=(method == "dqn"), strict_accounting=strict_accounting)
    metrics = sim.run(progress=progress)
    summary = metrics.summary()
    ts = metrics.time_series
    summary["throughput_pps"] = float(np.mean(ts["throughput_inst"])) if ts.get("throughput_inst") else 0.0
    policy = sim.policy.get_stats() if hasattr(sim.policy, "get_stats") else {}
    loss = getattr(sim.policy, "loss_history", None)
    training = None
    if loss:
        step = max(1, len(loss) // 2000)
        training = {"loss_history": [float(x) for x in loss[::step]], "stride": step, "total_steps": len(loss)}
    ckpt = None
    if checkpoint_path is not None and hasattr(sim.policy, "save"):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        sim.policy.save(str(checkpoint_path))
        ckpt = str(checkpoint_path)
    return {"summary": summary, "time_series": ts if keep_timeseries else None, "policy": policy,
            "training": training, "accounting": sim.accounting_report, "event_log": sim.event_log[-50:],
            "runtime_s": round(time.perf_counter() - t0, 2), "config": cfg.to_dict(), "checkpoint": ckpt}


# ---------------------------------------------------------------- metadata
def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def collect_metadata(**extra) -> Dict[str, Any]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("networkx", "torch", "matplotlib", "yaml"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv), "platform": platform.platform(), "versions": versions,
            "git_commit": _git_commit(), **extra}


# ---------------------------------------------------------------- statistics
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t_crit(df: int) -> float:
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def _clean(values) -> List[float]:
    return [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]


def describe(values) -> Dict[str, Any]:
    v = _clean(values)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None, "min": None, "max": None, "values": []}
    a = np.asarray(v)
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": float(a.mean()), "std": std,
            "ci95": float(t_crit(n - 1) * std / math.sqrt(n)) if n > 1 else None,
            "min": float(a.min()), "max": float(a.max()), "values": v}


def paired(method_vals: Dict[int, Optional[float]], base_vals: Dict[int, Optional[float]], direction: str) -> Dict[str, Any]:
    """Per-seed differences (method - baseline) over seeds where both runs produced a value."""
    diffs = [method_vals[s] - base_vals[s] for s in method_vals
             if s in base_vals and method_vals[s] is not None and base_vals[s] is not None]
    d = describe(diffs)
    better = sum(1 for x in diffs if (x > 0 if direction == "higher" else x < 0))
    return {"n": d["n"], "mean_diff": d["mean"], "ci95": d["ci95"], "better_in": better,
            "ci_excludes_zero": bool(d["ci95"] is not None and abs(d["mean"]) > d["ci95"])}
