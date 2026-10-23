"""Metrics collection with exact packet accounting.

Definitions (all of them are fixed, so every number can be audited)
-------------------------------------------------------------------
Every packet the traffic generator emits is a *generated attempt*, including packets rejected
by a full source queue. Each generated packet ends in exactly one terminal outcome:
    delivered | dropped (any reason, incl. src_queue_full, queue_full, link_failure, loop) | expired
or is still unresolved (queued or in flight) when the run ends. A guard ignores a second terminal
outcome for the same packet and counts it in ``duplicate_terminal`` (should always stay 0).

Warm-up: the summary is a *cohort* measured over packets CREATED at or after ``warmup``. Packets
created earlier are excluded from every cohort counter, delay and hop sample, so
    delivery_ratio + loss_ratio + unresolved_ratio == 1   and   loss_ratio <= 1.
The time series is measured by OBSERVATION time (events at t >= warmup in a sliding window).
Instantaneous ratios there are shares of packets that reached a terminal outcome in the window:
    delivery_ratio_inst = delivered / (delivered + dropped + expired),  loss_inst = 1 - that.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MetricsCollector:
    warmup: float = 0.0
    window: float = 5.0

    # cohort counters (packets created at/after warm-up)
    generated: int = 0
    delivered: int = 0
    dropped: int = 0
    expired: int = 0
    loops: int = 0                     # loop drops (already included in ``dropped``)
    priority_delivered: int = 0
    priority_generated: int = 0
    route_changes: int = 0
    duplicate_terminal: int = 0

    delays: List[float] = field(default_factory=list)
    hop_counts: List[int] = field(default_factory=list)
    priority_delays: List[float] = field(default_factory=list)
    drop_reasons: Dict[str, int] = field(default_factory=dict)

    # all-time counters (warm-up included) used for packet-conservation checks
    raw: Dict[str, int] = field(default_factory=lambda: {"generated": 0, "delivered": 0, "dropped": 0, "expired": 0})
    events: List[Dict[str, Any]] = field(default_factory=list)

    time_series: Dict[str, List[float]] = field(
        default_factory=lambda: {k: [] for k in (
            "t", "avg_delay", "p95_delay", "queue_total", "queue_max", "throughput_inst",
            "delivery_ratio_inst", "loss_inst", "route_changes_inst")}
    )

    _delivered_window: Deque[float] = field(default_factory=deque)
    _lost_window: Deque[float] = field(default_factory=deque)
    _route_window: Deque[float] = field(default_factory=deque)
    _terminal_ids: set = field(default_factory=set)
    _last_hop: Dict[Tuple[str, str], str] = field(default_factory=dict)
    _last_t: float = 0.0
    _last_delivery_inst: float = 1.0

    # ------------------------------------------------------------------ helpers
    def _cohort(self, packet) -> bool:
        return packet.creation_time >= self.warmup

    def _first_terminal(self, packet) -> bool:
        pid = getattr(packet, "packet_id", id(packet))
        if pid in self._terminal_ids:
            self.duplicate_terminal += 1
            return False
        self._terminal_ids.add(pid)
        return True

    # ------------------------------------------------------------------ recording
    def record_generation(self, packet, t: float, priority: bool = False) -> None:
        """Record one generated attempt. (Legacy form ``record_generation(n, t)`` is still accepted.)"""
        if isinstance(packet, int):
            self.raw["generated"] += packet
            if t >= self.warmup:
                self.generated += packet
                self.priority_generated += packet if priority else 0
            return
        self.raw["generated"] += 1
        if self._cohort(packet):
            self.generated += 1
            if packet.is_priority:
                self.priority_generated += 1

    def record_delivery(self, packet, t: float) -> None:
        if not self._first_terminal(packet):
            return
        self.raw["delivered"] += 1
        if t >= self.warmup:
            self._delivered_window.append(t)
        if not self._cohort(packet):
            return
        self.delivered += 1
        delay = t - packet.creation_time
        self.delays.append(delay)
        self.hop_counts.append(packet.hop_count)
        if packet.is_priority:
            self.priority_delivered += 1
            self.priority_delays.append(delay)

    def record_drop(self, packet, reason: str, t: float) -> None:
        if not self._first_terminal(packet):
            return
        self.raw["dropped"] += 1
        if t >= self.warmup:
            self._lost_window.append(t)
        if self._cohort(packet):
            self.dropped += 1
            self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1
            if reason == "loop":
                self.loops += 1

    def record_loop(self, packet, t: float) -> None:
        self.record_drop(packet, "loop", t)

    def record_expiry(self, packet, t: float) -> None:
        if not self._first_terminal(packet):
            return
        self.raw["expired"] += 1
        if t >= self.warmup:
            self._lost_window.append(t)
        if self._cohort(packet):
            self.expired += 1
            self.drop_reasons["expired"] = self.drop_reasons.get("expired", 0) + 1

    def record_route_choice(self, router_id: str, dst: str, next_hop: str, t: float) -> None:
        """Count a route change when a router sends traffic for ``dst`` via a different next hop."""
        key = (router_id, dst)
        prev = self._last_hop.get(key)
        self._last_hop[key] = next_hop
        if prev is not None and prev != next_hop and t >= self.warmup:
            self.route_changes += 1
            self._route_window.append(t)

    def mark_event(self, t: float, kind: str, detail: str = "") -> None:
        """Mark a network disturbance (kind: 'disruption' or 'recovery') for adaptation analysis."""
        self.events.append({"t": float(t), "kind": kind, "detail": detail})

    # ------------------------------------------------------------------ sampling
    def snapshot(self, t: float, routers: Dict) -> None:
        if t < self.warmup:
            return
        self._last_t = t
        cutoff = t - self.window
        for dq in (self._delivered_window, self._lost_window, self._route_window):
            while dq and dq[0] < cutoff:
                dq.popleft()
        d, l = len(self._delivered_window), len(self._lost_window)
        if d + l > 0:
            self._last_delivery_inst = d / (d + l)  # carried forward while nothing resolves
        dr = self._last_delivery_inst
        span = max(1.0, min(self.window, t - self.warmup))
        total_q = sum(len(r) for r in routers.values())
        max_q = max((len(r) for r in routers.values()), default=0)
        recent = self.delays[-400:] or [0.0]
        ts = self.time_series
        ts["t"].append(t)
        ts["avg_delay"].append(float(np.mean(recent)))
        ts["p95_delay"].append(float(np.percentile(recent, 95)) if len(recent) > 5 else float(np.mean(recent)))
        ts["queue_total"].append(total_q)
        ts["queue_max"].append(max_q)
        ts["throughput_inst"].append(d / span)
        ts["delivery_ratio_inst"].append(dr)
        ts["loss_inst"].append(1.0 - dr)
        ts["route_changes_inst"].append(len(self._route_window) / span)

    # ------------------------------------------------------------------ adaptation
    def adaptation_report(self, band: float = 0.05, delay_factor: float = 1.25,
                          hold: float = 2.0, baseline_window: float = 10.0) -> List[Dict[str, Any]]:
        """Time for performance to return to its pre-disruption band after each marked event.

        Baseline = mean delivery ratio / delay over ``baseline_window`` seconds before the
        disruption (a recovery event reuses the baseline of the disruption it follows). Performance
        is "back in band" when delivery >= baseline - ``band`` and delay <= max(delay_factor x
        baseline, baseline + 20 ms), and it stays there for ``hold`` seconds. Events too close
        to the end of the run to be confirmed report ``recovered=False``.
        """
        ts = self.time_series
        t = np.asarray(ts["t"], float)
        dr = np.asarray(ts["delivery_ratio_inst"], float)
        dl = np.asarray(ts["avg_delay"], float)
        report, base = [], None
        for ev in sorted(self.events, key=lambda e: e["t"]):
            te = ev["t"]
            entry = {"t": round(te, 3), "kind": ev["kind"], "detail": ev.get("detail", ""),
                     "baseline_delivery": None, "baseline_delay": None, "adaptation_time": None,
                     "recovered": False, "delivery_drop": None}
            if ev["kind"] == "disruption" and len(t):
                m = (t >= te - baseline_window) & (t < te)
                base = (float(dr[m].mean()), float(dl[m].mean())) if m.sum() >= 3 else None
            if base is not None and len(t):
                entry["baseline_delivery"], entry["baseline_delay"] = base
                ok = (dr >= base[0] - band) & (dl <= max(base[1] * delay_factor, base[1] + 0.02))
                end_t = None
                for i in np.where(t >= te)[0]:
                    if t[i] + hold > t[-1] + 1e-9:
                        break
                    w = (t >= t[i]) & (t < t[i] + hold)
                    if ok[w].all():
                        end_t = float(t[i])
                        break
                span = (t >= te) & (t <= end_t) if end_t is not None else (t >= te)
                if span.any():
                    entry["delivery_drop"] = round(max(0.0, base[0] - float(dr[span].min())), 4)
                if end_t is not None:
                    entry["adaptation_time"], entry["recovered"] = round(end_t - te, 3), True
            report.append(entry)
        return report

    # ------------------------------------------------------------------ summary
    def summary(self) -> Dict[str, Any]:
        total = max(1, self.generated)
        resolved = self.delivered + self.dropped + self.expired
        unresolved = max(0, self.generated - resolved)
        adapt = self.adaptation_report()
        times = [a["adaptation_time"] for a in adapt if a["recovered"]]
        observed = max(1e-9, self._last_t - self.warmup)
        return {
            "generated": self.generated,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "expired": self.expired,
            "loops": self.loops,
            "unresolved": unresolved,
            "delivery_ratio": self.delivered / total,
            "loss_ratio": (self.dropped + self.expired) / total,
            "unresolved_ratio": unresolved / total,
            "avg_delay": float(np.mean(self.delays)) if self.delays else 0.0,
            "p50_delay": float(np.median(self.delays)) if self.delays else 0.0,
            "p95_delay": float(np.percentile(self.delays, 95)) if self.delays else 0.0,
            "p99_delay": float(np.percentile(self.delays, 99)) if len(self.delays) > 20 else 0.0,
            "avg_hops": float(np.mean(self.hop_counts)) if self.hop_counts else 0.0,
            "priority_delivery_ratio": self.priority_delivered / max(1, self.priority_generated),
            "priority_avg_delay": float(np.mean(self.priority_delays)) if self.priority_delays else 0.0,
            "route_changes": self.route_changes,
            "route_changes_per_delivered": self.route_changes / max(1, self.delivered),
            "route_change_rate": self.route_changes / observed,
            "adaptation": adapt,
            "adaptation_time_mean": float(np.mean(times)) if times else None,
            "unrecovered_events": sum(1 for a in adapt if a["baseline_delivery"] is not None and not a["recovered"]),
            "duplicate_terminal": self.duplicate_terminal,
            "drop_reasons": dict(self.drop_reasons),
        }