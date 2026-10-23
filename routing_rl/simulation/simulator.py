"""Time-stepped network simulator with exact packet accounting and event injection."""
from __future__ import annotations
import random
from typing import Dict, Any, List
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn

from routing_rl.config import Config
from routing_rl.network.router import Router
from routing_rl.network.topology import build_topology, traffic_defaults
from routing_rl.simulation.traffic import TrafficGenerator
from routing_rl.simulation.events import EventScheduler, LinkFailureEvent
from routing_rl.metrics.collector import MetricsCollector


def _seed_everything(seed: int) -> None:
    """Seed the process-wide RNGs that policies use (tabular exploration uses ``random``)."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


class Simulator:
    """Traffic uses its own generator seeded only by ``cfg.simulation.seed``, so every routing
    method sees the identical packet arrivals for a given seed."""

    def __init__(self, cfg: Config, verbose: bool = False, train_rl: bool = True,
                 strict_accounting: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.train_rl = train_rl
        self.strict_accounting = strict_accounting
        seed = int(cfg.simulation.seed)
        _seed_everything(seed)
        self.rng = np.random.default_rng(seed)

        topo_seed = cfg.network.topology_seed
        if topo_seed is None:
            topo_seed = seed
        self.topology_seed = int(topo_seed)
        self.routers, self.links, self.graph = build_topology(
            cfg.network.topology,
            queue_size=cfg.network.default_queue_size,
            capacity=cfg.network.default_capacity,
            prop_delay=cfg.network.propagation_delay,
            seed=self.topology_seed,
            n_nodes_min=cfg.network.n_nodes_min,
            n_nodes_max=cfg.network.n_nodes_max,
        )
        self.n_nodes = len(self.routers)
        cfg.network.topology_seed = self.graph.graph.get("topology_seed", self.topology_seed)
        self.policy = self._build_policy(seed)
        self.traffic = TrafficGenerator(cfg.traffic, self.rng)
        self.metrics = MetricsCollector(warmup=cfg.simulation.warmup)
        self.events = EventScheduler()
        if cfg.network.enable_failures and cfg.network.failure_schedule:
            schedule = []
            for ev in cfg.network.failure_schedule:
                a, b = ev["link"][0], ev["link"][1]
                if (a, b) not in self.links and (b, a) not in self.links:
                    _, _, fallback = traffic_defaults(cfg.network.topology, self.routers)
                    ev = {**ev, "link": fallback}
                schedule.append(ev)
            self.events.from_config(schedule)

        self.t = 0.0
        self.dt = cfg.simulation.dt
        self.duration = cfg.simulation.duration
        self.steps = int(self.duration / self.dt)
        self.sample_every = cfg.simulation.sample_every
        self.event_log: List[str] = []
        self.accounting_report: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.history_every = max(1, int(0.25 / self.dt))  # ~0.25 s
        self.step_index = 0           # next tick for advance(); run() does not use it
        self.finalized = False

    def _build_policy(self, seed: int):
        m = self.cfg.routing.method
        rc = {
            "hop_penalty": self.cfg.routing.hop_penalty,
            "queue_penalty_scale": self.cfg.routing.queue_penalty_scale,
            "delivery_bonus": self.cfg.routing.delivery_bonus,
            "drop_penalty": self.cfg.routing.drop_penalty,
            "latency_scale": self.cfg.routing.latency_scale,
        }
        if m == "shortest_path":
            from routing_rl.routing.shortest_path import ShortestPathPolicy
            return ShortestPathPolicy(self.routers, self.links, rc)
        if m == "congestion_aware":
            from routing_rl.routing.congestion_aware import CongestionAwarePolicy
            return CongestionAwarePolicy(self.routers, self.links, alpha=self.cfg.routing.alpha, reward_cfg=rc)
        if m in ("tabular_q", "q_learning"):
            from routing_rl.routing.tabular_q import TabularQPolicy
            return TabularQPolicy(self.routers, self.links, reward_cfg=rc)
        if m == "dqn":
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise ImportError("Routing method 'dqn' requires PyTorch (pip install torch).") from exc
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            from routing_rl.routing.dqn.agent import DQNPolicy
            return DQNPolicy(self.routers, self.links, cfg=self.cfg.routing.dqn, reward_cfg=rc, train=self.train_rl)
        raise ValueError(f"Unknown routing method: {m}")

    # ------------------------------------------------------------------
    def run(self, progress: bool = True) -> MetricsCollector:
        if progress:
            with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(bar_width=40),
                          "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn()) as prog:
                task = prog.add_task(f"Sim [{self.cfg.routing.method}]", total=self.steps)
                self._run_loop(lambda: prog.update(task, advance=1))
        else:
            self._run_loop(None)
        return self.metrics

    def _run_loop(self, tick) -> None:
        for step in range(self.steps):
            self._tick(step)
            if tick:
                tick()
        self._finalize()

    def _tick(self, step: int) -> None:
        """One time step plus its sampling/recording bookkeeping (shared by run() and advance())."""
        self.t += self.dt
        self._step()
        if step % self.sample_every == 0:
            self.metrics.snapshot(self.t, self.routers)
        if step % self.history_every == 0:
            self._record_history()

    def _finalize(self) -> None:
        self._record_history()  # final
        self.accounting_report = self.accounting()
        if not self.accounting_report["balanced"]:
            msg = f"ACCOUNTING MISMATCH: {self.accounting_report}"
            self.event_log.append(msg)
            if self.strict_accounting:
                raise AssertionError(msg)
        if hasattr(self.policy, "on_episode_end"):
            self.policy.on_episode_end()
        self.finalized = True

    # ------------------------------------------------------------------
    # Incremental control (used by the live demo; run() above is unchanged in behaviour)
    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.step_index >= self.steps

    def advance(self, n: int = 1) -> int:
        """Run up to ``n`` more time steps. Returns how many were executed. Finalises once at the end."""
        done = 0
        while done < n and self.step_index < self.steps:
            self._tick(self.step_index)
            self.step_index += 1
            done += 1
        if self.step_index >= self.steps and not self.finalized:
            self._finalize()
        return done

    def _link_key(self, a: str, b: str):
        if (a, b) not in self.links and (b, a) not in self.links:
            raise ValueError(f"No link between {a!r} and {b!r} in this topology")
        return (a, b) if (a, b) in self.links else (b, a)

    def inject_link_outage(self, a: str, b: str, duration: float, capacity=None) -> LinkFailureEvent:
        """Schedule a real outage on link a<->b starting now. It is applied (and logged to event_log and
        the metrics event marks) by the normal event system on the next time step."""
        a, b = self._link_key(a, b)
        ev = LinkFailureEvent(self.t, self.t + float(duration), (a, b), capacity)
        self.events.add(ev)
        return ev

    def end_link_outages(self, a: str, b: str) -> int:
        """End every outage currently affecting link a<->b (the event system then restores the link)."""
        a, b = self._link_key(a, b)
        key = tuple(sorted((a, b)))
        n = 0
        for ev in self.events.events:
            if tuple(sorted(ev.link)) == key and ev.t_start <= self.t + self.dt < ev.t_end:
                ev.t_end = self.t   # inactive from the next step on
                n += 1
        return n

    def accounting(self) -> Dict[str, Any]:
        """Packet conservation: generated == delivered + dropped + expired + queued + in transit."""
        raw = self.metrics.raw
        queued = sum(len(r) for r in self.routers.values())
        in_transit = sum(l.in_transit_count for l in self.links.values())
        resolved = raw["delivered"] + raw["dropped"] + raw["expired"]
        return {**raw, "queued": queued, "in_transit": in_transit,
                "balanced": raw["generated"] == resolved + queued + in_transit,
                "duplicate_terminal": self.metrics.duplicate_terminal}

    def _record_history(self) -> None:
        self.history.append({
            "t": self.t,
            "queues": {rid: len(r) for rid, r in self.routers.items()},
            "links": {f"{s}->{d}": {"up": l.is_up, "util": l.utilization_ema} for (s, d), l in self.links.items()},
        })

    # ------------------------------------------------------------------
    def _step(self) -> None:
        now, dt = self.t, self.dt

        # 1. Scheduled events (overlap-safe) and bookkeeping for adaptation analysis
        self.event_log.extend(self.events.apply(now, self.links, self.policy))
        for ch in self.events.last_changes:
            label = f"{ch['link'][0]}-{ch['link'][1]}"
            if ch["prev_normal"] and not ch["normal"]:
                self.metrics.mark_event(now, "disruption", label)
            elif ch["normal"] and not ch["prev_normal"]:
                self.metrics.mark_event(now, "recovery", label)

        # 2. Packets that were in flight on a link that just failed are lost: count them once
        for link in self.links.values():
            for pkt in link.pop_lost():
                self.metrics.record_drop(pkt, "link_failure", now)
                self.policy.on_packet_dropped(pkt, "link_failure", now)

        # 3. Link capacity tokens
        for link in self.links.values():
            link.step(dt)

        # 4. Traffic: every emitted packet is a generated attempt
        for src, dst, pkt in self.traffic.generate(now, dt, self.cfg.network.default_ttl):
            self.metrics.record_generation(pkt, now)
            if not self.routers[src].enqueue(pkt):
                self.metrics.record_drop(pkt, "src_queue_full", now)
                self.policy.on_packet_dropped(pkt, "src_queue_full", now)

        # 5. Arrivals (TTL is enforced on arrival too)
        for link in self.links.values():
            for itp in link.deliver_ready(now):
                pkt = itp.packet
                if pkt.expiry_time <= now:
                    self.metrics.record_expiry(pkt, now)
                    self.policy.on_packet_dropped(pkt, "expired", now)
                    continue
                pkt.current_router = link.dst
                if link.dst == pkt.dst:
                    pkt.path.append(link.dst)
                    self.metrics.record_delivery(pkt, now)
                    self.policy.on_packet_delivered(pkt, now)
                elif not self.routers[link.dst].enqueue(pkt):
                    self.metrics.record_drop(pkt, "queue_full", now)
                    self.policy.on_packet_dropped(pkt, "queue_full", now)

        # 6. Forward + expire
        for router in self.routers.values():
            self._forward_router(router, now)
            self._expire_router(router, now)

    def _forward_router(self, router: Router, now: float) -> None:
        budget = 8  # bounded work per step to avoid starvation under overload
        deferred = []
        while router.has_packets() and budget > 0:
            budget -= 1
            pkt = router.peek()
            if pkt is None:
                break
            if pkt.expiry_time <= now:
                router.pop()
                self.metrics.record_expiry(pkt, now)
                self.policy.on_packet_dropped(pkt, "expired", now)
                continue
            if pkt.path and router.id in pkt.path:  # loop detection
                router.pop()
                self.metrics.record_loop(pkt, now)
                self.policy.on_packet_dropped(pkt, "loop", now)
                continue
            nxt = self.policy.select_next_hop(pkt, router, now)
            link = self.links.get((router.id, nxt)) if nxt is not None else None
            if link is None or not link.is_up:
                # No usable route right now: park the packet so it cannot block other flows.
                deferred.append(router.pop())
                continue
            if not link.can_accept():
                break  # out of capacity on this link this step
            router.pop()
            pkt.path.append(router.id)
            pkt.hop_count += 1
            link.send(pkt, now)
            self.metrics.record_route_choice(router.id, pkt.dst, nxt, now)
            self.policy.on_hop_forward(pkt, router.id, nxt, now)
            if self.verbose:
                log = self.policy.verbose_decision(pkt, router, now)
                if log:
                    print(log)
        for pkt in deferred:
            if not router.enqueue(pkt):
                self.metrics.record_drop(pkt, "queue_full", now)
                self.policy.on_packet_dropped(pkt, "queue_full", now)

    def _expire_router(self, router: Router, now: float) -> None:
        keep = []
        while router.has_packets():
            pkt = router.pop()
            if pkt.expiry_time <= now:
                self.metrics.record_expiry(pkt, now)
                self.policy.on_packet_dropped(pkt, "expired", now)
            else:
                keep.append(pkt)
        for pkt in keep:  # re-enqueue preserving priority and order
            if not router.enqueue(pkt):
                self.metrics.record_drop(pkt, "queue_full", now)
                self.policy.on_packet_dropped(pkt, "queue_full", now)