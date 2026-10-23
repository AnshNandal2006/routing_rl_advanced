"""Live, controllable Q-learning session built on the REAL Simulator and the REAL TabularQPolicy.

Nothing in here simulates routing by itself. It
  * builds a normal ``Simulator`` with ``routing.method = tabular_q``,
  * attaches an observer (``policy.tracer``) that records what the agent really does:
      - a *decision* each time a packet is actually forwarded (the selection snapshot the agent used),
      - an *update* each time ``_update_q`` really changes the Q-table (with every input of the formula),
      - delivered / dropped packets,
  * advances the simulator in small chunks from a background thread (or one decision at a time),
  * applies controls through the simulator's own event system (link outages) and traffic generator,
  * produces JSON-safe snapshots whose Q-values are read from ``policy.q_table`` at snapshot time.

Everything shown to the user is derived from those records. Explanations only state what the records
support, and say so when the cause cannot be determined.

No PyTorch is imported anywhere on this path.
"""
from __future__ import annotations

import copy
import math
import random
import threading
import time
import traceback
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from routing_rl.config import Config
from routing_rl.network.topology import get_layout_positions
from routing_rl.routing.tabular_q import TabularQPolicy
from routing_rl.simulation.simulator import Simulator

# ---------------------------------------------------------------------------
# Limits (everything user-controllable is bounded)
# ---------------------------------------------------------------------------
HISTORY_DEFAULT = 300            # decisions and updates kept (each); older ones are dropped
HISTORY_MAX = 1000
SPEEDS = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)   # simulated seconds per wall-clock second
TRAFFIC_MULT = (0.25, 4.0)
OUTAGE_DURATION = (1.0, 60.0)
STEP_MAX_TICKS = 400             # "step one decision" gives up after this many ticks (4 s of sim time)
CHUNK_WALL_S = 0.05              # the runner advances the simulation in 50 ms wall-clock slices
QTABLE_PAGE_MAX = 200
CONTROL_LOG_MAX = 40


class ControlError(ValueError):
    """A control request that is invalid or not possible in the current state."""


def _f(x: Any) -> Any:
    """JSON-safe float (NaN/inf -> None)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _state_key(state: Tuple) -> str:
    return f"{state[0]}|{state[1]}|{state[2]}"


class LiveSession:
    def __init__(self, cfg: Config, history: int = HISTORY_DEFAULT):
        if cfg.routing.method not in ("tabular_q", "q_learning"):
            raise ControlError("The live Q-learning brain needs routing.method = tabular_q")
        if not 10 <= int(history) <= HISTORY_MAX:
            raise ControlError(f"history must be between 10 and {HISTORY_MAX}")
        self._cfg0 = copy.deepcopy(cfg)
        self.history_len = int(history)
        self.lock = threading.RLock()
        self.speed = 1.0
        self.episode = 0
        self.traffic_multiplier = 1.0
        self.error: Optional[str] = None
        self._running = False
        self._closed = False
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reset_info: Dict[str, Any] = {}
        self.controls_log: Deque[Dict[str, Any]] = deque(maxlen=CONTROL_LOG_MAX)
        self._build(keep_from=None)

    # ------------------------------------------------------------------ build / reset
    def _build(self, keep_from: Optional[TabularQPolicy]) -> None:
        cfg = copy.deepcopy(self._cfg0)
        sim = Simulator(cfg, train_rl=False)
        policy = sim.policy
        if not isinstance(policy, TabularQPolicy):
            raise ControlError("tabular_q policy expected")
        seed = int(cfg.simulation.seed)
        # Private RNG: episode 0 reproduces exactly what a plain Simulator.run() with this seed does
        # (the module-level `random` and Random(seed) are the same generator), and a running live
        # session is unaffected by any other simulation in the process.
        policy.rng = random.Random(seed + 1_000_003 * self.episode)
        kept = 0
        if keep_from is not None:
            policy.q_table = copy.deepcopy(keep_from.q_table)
            policy.epsilon = keep_from.epsilon
            policy.updates = keep_from.updates
            policy.explorations = keep_from.explorations
            policy.alpha, policy.gamma = keep_from.alpha, keep_from.gamma
            policy.explore_enabled = keep_from.explore_enabled
            kept = sum(len(v) for v in policy.q_table.values())
        policy.tracer = self._on_event
        self.cfg, self.sim, self.policy = cfg, sim, policy
        self.traffic_multiplier = 1.0

        n = self.history_len
        self.decisions: Deque[Dict[str, Any]] = deque(maxlen=n)
        self.updates_log: Deque[Dict[str, Any]] = deque(maxlen=n)
        self.decision_seq = 0
        self.update_seq = 0
        self.route_changes_traced = 0
        self.drops = Counter()
        self.delivered = 0
        self._last_hop: Dict[Tuple[str, str], str] = {}
        self._last_level: Dict[Tuple[str, str], int] = {}
        self._upd_since: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = {}
        self._last_upd: Dict[Tuple, Dict[str, Any]] = {}
        self._decision_by_packet: Dict[int, Dict[str, Any]] = {}
        self._route_cur: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._route_prev: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._route_recent: Dict[Tuple[str, str], Deque[Tuple[str, ...]]] = {}
        self._edges = self._undirected_edges()
        self._cursor: Optional[int] = None   # step mode: last decision revealed to the user (None = follow latest)
        self.controls_log.clear()
        self.reset_info = {
            "episode": self.episode,
            "q_table": "initial" if self.episode == 0 else ("preserved" if keep_from is not None else "reset"),
            "q_entries": kept, "epsilon": policy.epsilon,
        }

    def reset(self, keep_learning: bool) -> None:
        """New episode (fresh network, traffic and clock). The Q-table is either carried over or wiped."""
        with self.lock:
            self._running = False
            old = self.policy
            self.episode += 1
            self._build(keep_from=old if keep_learning else None)
            self.error = None
            self._log("reset", f"Episode {self.episode} started; Q-table "
                      f"{'PRESERVED (' + str(self.reset_info['q_entries']) + ' entries, epsilon ' + format(self.policy.epsilon, '.3f') + ')' if keep_learning else 'WIPED (empty, epsilon back to ' + format(self.policy.initial_epsilon, '.2f') + ')'}")

    def close(self) -> None:
        self._closed = True
        self._running = False
        self._wake.set()

    # ------------------------------------------------------------------ observer
    def _on_event(self, ev: Dict[str, Any]) -> None:
        """Called by the policy while the simulator steps (the session lock is already held)."""
        kind = ev["type"]
        if kind == "decision":
            self._record_decision(ev)
        elif kind == "update":
            self._record_update(ev)
        elif kind == "delivered":
            self._record_delivery(ev)
        elif kind == "dropped":
            self.drops[ev["reason"]] += 1

    def _link_obs(self, a: str, b: str, now: float) -> Dict[str, Any]:
        link = self.sim.links.get((a, b))
        if link is None:
            return {"up": False, "delay_ms": None, "util": None}
        return {"up": bool(link.is_up),
                "delay_ms": round((link.propagation_delay + link.backlog_delay(now)) * 1000.0, 2),
                "util": round(float(link.utilization_ema), 3)}

    def _record_decision(self, ev: Dict[str, Any]) -> None:
        self.decision_seq += 1
        router, dst = ev["router"], ev["dst"]
        flow = (router, dst)
        sel = ev["forwarded_to"]
        prev = self._last_hop.get(flow)
        prev_level = self._last_level.get(flow)
        changed = prev is not None and prev != sel
        if changed:
            self.route_changes_traced += 1
        recent = list(self._upd_since.pop(flow, []))
        t = ev["forwarded_t"]
        cands = []
        for c in ev["candidates"]:
            obs = self._link_obs(router, c["hop"], t)
            nr = self.sim.routers.get(c["hop"])
            cands.append({**c, "link_delay_ms": obs["delay_ms"], "link_util": obs["util"],
                          "next_queue_len": len(nr) if nr is not None else None,
                          "next_queue_level": nr.queue_level() if nr is not None else None})
        rec = {
            "seq": self.decision_seq, "t": round(t, 3), "packet_id": ev["packet_id"], "priority": ev["priority"],
            "src": ev["src"], "dst": dst, "router": router, "path": ev["path"], "state": list(ev["state"]),
            "queue_level": ev["queue_level"], "mode": ev["mode"], "epsilon": ev["epsilon"],
            "explore_enabled": ev["explore_enabled"], "selected": sel, "selection_matches_forward": ev["selected"] == sel,
            "greedy_set": ev["greedy_set"], "candidates": cands, "loop_excluded": ev["loop_excluded"],
            "down": ev["down"], "previous_hop": prev, "previous_level": prev_level, "changed": changed,
            "next_queue_level": ev["next_queue_level"], "update_seq": None,
        }
        rec["explanation"] = self._explain(rec, recent)
        rec["trigger"] = self._trigger(rec)
        self.decisions.append(rec)
        self._decision_by_packet[ev["packet_id"]] = rec
        if len(self._decision_by_packet) > 4 * self.history_len:       # bounded
            for k in list(self._decision_by_packet)[: self.history_len]:
                self._decision_by_packet.pop(k, None)
        self._last_hop[flow] = sel
        self._last_level[flow] = ev["queue_level"]

    def _record_update(self, ev: Dict[str, Any]) -> None:
        self.update_seq += 1
        s, a = ev["state"], ev["action"]
        recomputed = ev["q_old"] + ev["alpha"] * (ev["reward"] + ev["gamma"] * ev["max_next"] - ev["q_old"])
        rec = {
            "seq": self.update_seq, "t": _f(round(ev["t"], 3)) if ev["t"] is not None else None, "trigger": ev["trigger"],
            "packet_id": ev["packet_id"], "state": list(s), "action": a, "first_visit": ev["first_visit"],
            "q_old": ev["q_old"], "alpha": ev["alpha"], "gamma": ev["gamma"], "reward": ev["reward"],
            "max_next": ev["max_next"], "next_state": list(ev["next_state"]) if ev["next_state"] else None,
            "next_q": ev["next_q"], "target": ev["target"], "q_new": ev["q_new"],
            "recomputed": recomputed, "verified": abs(recomputed - ev["q_new"]) <= 1e-9 * (1.0 + abs(recomputed)),
            "reward_parts": ev["reward_parts"], "changed": ev["q_new"] != ev["q_old"],
        }
        self.updates_log.append(rec)
        self._last_upd[(tuple(s), a)] = rec
        self._upd_since.setdefault((s[0], s[1]), deque(maxlen=20)).append(rec)
        d = self._decision_by_packet.get(ev["packet_id"])
        if d is not None and ev["trigger"] == "hop_forward" and d["update_seq"] is None:
            d["update_seq"] = rec["seq"]

    def _record_delivery(self, ev: Dict[str, Any]) -> None:
        self.delivered += 1
        flow = (ev["src"], ev["dst"])
        path = tuple(ev["path"])
        cur = self._route_cur.get(flow)
        if cur is None or tuple(cur["path"]) != path:
            if cur is not None:
                self._route_prev[flow] = cur
            self._route_cur[flow] = {"path": list(path), "since": round(ev["t"], 3), "delay": round(ev["delay"], 4)}
        recent = self._route_recent.setdefault(flow, deque(maxlen=100))
        recent.append(path)

    # ------------------------------------------------------------------ explanations
    @staticmethod
    def _fq(x: float) -> str:
        return f"{x:.2f}"

    def _explain(self, d: Dict[str, Any], recent_updates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Only statements supported by the recorded decision. kind: policy | learned | observed | note."""
        out: List[Dict[str, str]] = []
        r, dst, sel, cands = d["router"], d["dst"], d["selected"], d["candidates"]
        q = {c["hop"]: c["q"] for c in cands}
        n = len(cands)
        greedy = d["greedy_set"]
        eps_txt = f"ε = {d['epsilon']:.3f}"
        if d["mode"] == "explore":
            out.append({"kind": "policy", "text": f"EXPLORATION ({eps_txt}): {sel} was drawn at random from {n} valid action(s). "
                        "It was not chosen because of its Q-value."})
            if sel in greedy:
                out.append({"kind": "learned", "text": f"It happens to be a greedy action too (Q = {self._fq(q[sel])})."})
            else:
                g = greedy[0]
                out.append({"kind": "learned", "text": f"The greedy choice would have been {g} (Q = {self._fq(q[g])}); "
                            f"the explored action {sel} has Q = {self._fq(q[sel])}."})
        elif d["mode"] == "greedy":
            others = sorted(((v, k) for k, v in q.items() if k != sel), reverse=True)
            tail = f"; next best is {others[0][1]} (Q = {self._fq(others[0][0])})" if others else "; it was the only valid action"
            off = " Exploration is switched off." if not d["explore_enabled"] else f" The exploration roll did not trigger ({eps_txt})."
            out.append({"kind": "policy", "text": f"GREEDY: {sel} has the highest Q-value among the valid actions "
                        f"(Q = {self._fq(q[sel])}{tail}).{off}"})
        else:
            unv = all(not c["visited"] for c in cands if c["hop"] in greedy)
            out.append({"kind": "policy", "text": f"GREEDY with a tie: {len(greedy)} actions share the top Q-value "
                        f"({self._fq(q[sel])}); the agent breaks ties at random, picking {sel}."
                        + (" All tied actions are unvisited (initial value 0.0)." if unv else "")})
        if d["loop_excluded"]:
            out.append({"kind": "observed", "text": f"Loop avoidance removed {', '.join(d['loop_excluded'])} "
                        "(already on this packet's path)."})
        if d["down"]:
            out.append({"kind": "observed", "text": f"Link(s) to {', '.join(d['down'])} were down, so those actions were unavailable."})
        out.append({"kind": "observed", "text": f"Agent state = (router {r}, destination {dst}, own queue level {d['queue_level']}). "
                    "The Q-table row is specific to this queue level."})
        if d["changed"]:
            prev = d["previous_hop"]
            out.append({"kind": "note", "text": f"ROUTE CHANGE at {r} for destination {dst}: previous next hop {prev} → {sel}."})
            if d["previous_level"] is not None and d["previous_level"] != d["queue_level"]:
                out.append({"kind": "observed", "text": f"Router {r}'s queue level changed from {d['previous_level']} to "
                            f"{d['queue_level']}, so a different Q-table row was consulted."})
            plink = self.sim.links.get((r, prev))
            if prev in d["down"] or (plink is not None and not plink.is_up):
                out.append({"kind": "observed", "text": f"Link {r}→{prev} is down, so {prev} is no longer a valid action."})
            if prev in d["loop_excluded"]:
                out.append({"kind": "observed", "text": f"{prev} is already on this packet's path (loop avoidance)."})
            if prev in q and sel in q:
                out.append({"kind": "learned", "text": f"Learned estimates now: Q({r}→{prev}) = {self._fq(q[prev])} "
                            f"vs Q({r}→{sel}) = {self._fq(q[sel])}."})
            moved = [u for u in recent_updates if u["action"] == prev and u["changed"]]
            for u in moved[-2:]:
                rp = u["reward_parts"]
                why = self._reward_phrase(u["trigger"], rp)
                out.append({"kind": "learned", "text": f"Since the last decision here, Q({r}→{prev}) was updated "
                            f"{self._fq(u['q_old'])} → {self._fq(u['q_new'])} at t = {u['t']:.2f} s ({why})."})
            if d["mode"] == "explore":
                out.append({"kind": "note", "text": "This change is an exploration choice, not a learned preference."})
            elif not (prev in d["down"] or prev in d["loop_excluded"] or (prev in q and sel in q) or moved):
                out.append({"kind": "note", "text": "Action selected by the current policy; detailed cause unavailable."})
        return out

    @staticmethod
    def _reward_phrase(trigger: str, rp: Dict[str, Any]) -> str:
        if trigger == "hop_forward":
            return (f"reward {rp.get('hop', 0):+.2f} hop cost + {rp.get('queue', 0):+.2f} queue penalty "
                    f"(next router {rp.get('next_router')} at queue level {rp.get('next_queue_level')})"
                    + (f", ×{rp['priority_multiplier']}" if rp.get("priority_multiplier", 1) != 1 else ""))
        if trigger == "delivered":
            return f"delivery bonus {rp.get('delivery_bonus', 0):+.2f}"
        if trigger.startswith("dropped"):
            return f"drop penalty {rp.get('drop_penalty', 0):+.2f}, reason: {rp.get('reason')}"
        return trigger

    def _trigger(self, d: Dict[str, Any]) -> str:
        if d["mode"] == "explore":
            return "Exploration choice"
        if d["changed"]:
            return "Greedy choice changed after learning / state change"
        return "Greedy choice (unchanged)" if d["previous_hop"] is not None else "First decision for this router and destination"

    # ------------------------------------------------------------------ runner
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._runner, name="live-q-runner", daemon=True)
            self._thread.start()

    def _runner(self) -> None:
        while not self._closed:
            if not self._running:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            t0 = time.perf_counter()
            try:
                with self.lock:
                    if not self._running:
                        continue
                    if self.sim.done:
                        self._running = False
                        continue
                    ticks = max(1, int(self.speed * CHUNK_WALL_S / self.sim.dt))
                    self.sim.advance(ticks)
                    if self.sim.done:
                        self._running = False
            except Exception:                     # never swallow: keep the traceback and show it in the UI
                self.error = traceback.format_exc()
                self._running = False
                traceback.print_exc()
            self._wake.clear()   # a stale 'play' signal must not defeat the speed throttle
            self._wake.wait(max(0.0, CHUNK_WALL_S - (time.perf_counter() - t0)))

    # ------------------------------------------------------------------ controls
    def _log(self, kind: str, text: str) -> None:
        self.controls_log.append({"t": round(self.sim.t, 2), "kind": kind, "text": text})

    def control(self, action: str, **kw: Any) -> Dict[str, Any]:
        """Single entry point for every UI control; validates everything."""
        with self.lock:
            fn = getattr(self, f"_c_{action}", None) if action in CONTROL_ACTIONS else None
            if fn is None:
                raise ControlError(f"unknown control {action!r}")
            return fn(**kw) or {}

    def _c_play(self) -> Dict[str, Any]:
        if self.sim.done:
            raise ControlError("This episode has finished. Reset the episode to continue.")
        self.error = None
        self._cursor = None
        self._running = True
        self._ensure_thread()
        self._wake.set()
        return {}

    def _c_pause(self) -> Dict[str, Any]:
        self._running = False
        return {}

    def _c_speed(self, value: Any = None) -> Dict[str, Any]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ControlError("speed must be a number")
        if v not in SPEEDS:
            raise ControlError(f"speed must be one of {list(SPEEDS)}")
        self.speed = v
        return {}

    def _c_follow(self) -> Dict[str, Any]:
        self._cursor = None
        return {}

    def _c_step(self) -> Dict[str, Any]:
        """Reveal exactly one more routing decision.

        The simulator advances in whole time steps and one step can forward several packets, so decisions
        are revealed one at a time from the recorded stream (the cursor); the simulator itself only
        advances when there is no unrevealed decision left."""
        self._running = False
        cur = self._cursor if self._cursor is not None else self.decision_seq
        target = cur + 1
        ticks = 0
        while self.decision_seq < target and ticks < STEP_MAX_TICKS and not self.sim.done:
            ticks += self.sim.advance(1)
        if self.decision_seq >= target:
            self._cursor = target
            return {"found": True, "ticks": ticks, "message": None}
        if self.sim.done:
            raise ControlError("This episode has finished. Reset the episode to continue.")
        return {"found": False, "ticks": ticks,
                "message": f"No routing decision occurred within {ticks * self.sim.dt:.1f} s of simulated time."}

    def _c_reset(self, keep_learning: Any = None) -> Dict[str, Any]:
        if not isinstance(keep_learning, bool):
            raise ControlError("keep_learning must be true or false")
        self.reset(keep_learning)
        return {}

    def _c_traffic(self, multiplier: Any = None) -> Dict[str, Any]:
        try:
            m = float(multiplier)
        except (TypeError, ValueError):
            raise ControlError("multiplier must be a number")
        if not (TRAFFIC_MULT[0] <= m <= TRAFFIC_MULT[1]) or not math.isfinite(m):
            raise ControlError(f"multiplier must be between {TRAFFIC_MULT[0]} and {TRAFFIC_MULT[1]}")
        self.sim.traffic.rate_multiplier = m
        self.traffic_multiplier = m
        self._log("traffic", "Traffic restored to normal (×1.0)" if m == 1.0 else f"Offered load set to ×{m:g}")
        return {}

    def _link_args(self, a: Any, b: Any) -> Tuple[str, str]:
        if not isinstance(a, str) or not isinstance(b, str):
            raise ControlError("link endpoints must be router ids")
        if (a, b) not in self.sim.links and (b, a) not in self.sim.links:
            raise ControlError(f"No link between {a!r} and {b!r} in this topology")
        return a, b

    def _c_fail_link(self, a: Any = None, b: Any = None, duration: Any = None) -> Dict[str, Any]:
        a, b = self._link_args(a, b)
        try:
            dur = float(duration)
        except (TypeError, ValueError):
            raise ControlError("duration must be a number")
        if not (OUTAGE_DURATION[0] <= dur <= OUTAGE_DURATION[1]):
            raise ControlError(f"duration must be between {OUTAGE_DURATION[0]:g} and {OUTAGE_DURATION[1]:g} seconds")
        if self.sim.done:
            raise ControlError("This episode has finished. Reset the episode to continue.")
        self.sim.inject_link_outage(a, b, dur)
        self._log("fail", f"Outage injected on link {a}↔{b} for {dur:g} s (applied by the event system on the next step)")
        return {}

    def _c_restore_link(self, a: Any = None, b: Any = None) -> Dict[str, Any]:
        a, b = self._link_args(a, b)
        n = self.sim.end_link_outages(a, b)
        if n == 0:
            raise ControlError(f"No active outage on link {a}↔{b}")
        self._log("restore", f"Outage on link {a}↔{b} ended early; the event system restores it on the next step")
        return {}

    def _c_explore(self, enabled: Any = None) -> Dict[str, Any]:
        if not isinstance(enabled, bool):
            raise ControlError("enabled must be true or false")
        self.policy.explore_enabled = enabled
        self._log("explore", "Exploration ON (ε-greedy)" if enabled else "Exploration OFF (purely greedy)")
        return {}

    def _c_params(self, alpha: Any = None, gamma: Any = None) -> Dict[str, Any]:
        try:
            al, ga = float(alpha), float(gamma)
        except (TypeError, ValueError):
            raise ControlError("alpha and gamma must be numbers")
        if not (math.isfinite(al) and 0.0 < al <= 1.0):
            raise ControlError("learning rate α must be in (0, 1]")
        if not (math.isfinite(ga) and 0.0 <= ga < 1.0):
            raise ControlError("discount γ must be in [0, 1)")
        self.policy.alpha, self.policy.gamma = al, ga
        self._log("params", f"α = {al:g}, γ = {ga:g} (applies to future updates only)")
        return {}

    # ------------------------------------------------------------------ snapshots
    def _undirected_edges(self) -> List[Tuple[str, str]]:
        out, seen = [], set()
        for (u, v) in self.sim.links:
            k = tuple(sorted((u, v)))
            if k not in seen:
                seen.add(k)
                out.append((u, v))
        return out

    def _graph(self) -> Dict[str, Any]:
        pos = get_layout_positions(self.cfg.network.topology, self.sim.routers)
        ids = sorted(self.sim.routers)
        xs = [pos.get(i, (0, 0))[0] for i in ids]
        ys = [pos.get(i, (0, 0))[1] for i in ids]
        sx, sy = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
        srcs, dsts = set(map(str, self.cfg.traffic.sources)), set(map(str, self.cfg.traffic.destinations))
        return {"nodes": [{"id": i, "x": (xs[k] - min(xs)) / sx, "y": (ys[k] - min(ys)) / sy,
                           "role": "source" if i in srcs else "destination" if i in dsts else "relay"}
                          for k, i in enumerate(ids)],
                "edges": [[u, v] for u, v in self._edges]}

    def status(self) -> str:
        if self.error:
            return "error"
        if self.sim.done:
            return "finished"
        return "running" if self._running else "paused"

    def _focus_rows(self, d: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rows for the decision's state, with Q-values READ FROM THE AGENT's table right now."""
        state = tuple(d["state"])
        router = self.sim.routers[d["router"]]
        qrow = self.policy.q_table.get(state, {})
        at_dec = {c["hop"]: c for c in d["candidates"]}
        rows = []
        for hop in router.neighbors:
            live_visited = hop in qrow
            live_q = qrow.get(hop)
            obs = self._link_obs(router.id, hop, self.sim.t)
            lu = self._last_upd.get((state, hop))
            if hop == d["selected"]:
                status = "explored" if d["mode"] == "explore" else "selected"
            elif hop in d["down"]:
                status = "down"
            elif hop in d["loop_excluded"]:
                status = "excluded"
            elif hop in at_dec:
                status = "alternative"
            else:
                status = "unavailable"
            c = at_dec.get(hop)
            rows.append({
                "router": router.id, "dst": d["dst"], "level": state[2], "hop": hop, "status": status,
                "visited": live_visited, "q": live_q if live_visited else None, "q_effective": live_q if live_visited else 0.0,
                "q_at_decision": c["q"] if c else None,
                "link_up": obs["up"], "link_delay_ms": obs["delay_ms"], "link_util": obs["util"],
                "last_update": ({"seq": lu["seq"], "t": lu["t"], "q_old": lu["q_old"], "q_new": lu["q_new"],
                                 "trigger": lu["trigger"]} if lu else None),
            })
        return rows

    def _math(self, u: Dict[str, Any]) -> Dict[str, Any]:
        s = tuple(u["state"])
        terminal = u["next_state"] is None
        return {**u,
                "terminal": terminal,
                "formula": "Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s′,a′) − Q(s,a) ]",
                "table_now": self.policy.q_table.get(s, {}).get(u["action"]),
                "note": ("Terminal update (packet delivered or dropped): there is no next state, so the agent uses max Q(s′,a′) = 0."
                         if terminal else None)}

    def snapshot(self, focus: Optional[int] = None, update_focus: Optional[int] = None, limit: int = 40) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        with self.lock:
            sim, pol = self.sim, self.policy
            dec_list = list(self.decisions)
            pending = 0
            if self._cursor is not None:             # step mode: hide decisions that have not been revealed yet
                pending = max(0, self.decision_seq - self._cursor)
                dec_list = [d for d in dec_list if d["seq"] <= self._cursor]
            fd = None
            if dec_list:
                fd = next((d for d in dec_list if d["seq"] == focus), None) if focus is not None else None
                fd = fd or dec_list[-1]
            upd_list = list(self.updates_log)
            if self._cursor is not None and dec_list and dec_list[-1]["update_seq"] is not None:
                upd_list = [u for u in upd_list if u["seq"] <= dec_list[-1]["update_seq"]]
            fu = None
            if upd_list:
                fu = next((u for u in upd_list if u["seq"] == update_focus), None) if update_focus is not None else None
                if fu is None and fd is not None and fd["update_seq"] is not None:
                    fu = next((u for u in upd_list if u["seq"] == fd["update_seq"]), None)
                fu = fu or upd_list[-1]
            rows = self._focus_rows(fd) if fd else []
            valid_q = [r["q_effective"] for r in rows if r["status"] in ("selected", "explored", "alternative")]
            routes = []
            for flow, cur in list(self._route_cur.items())[:4]:
                recent = self._route_recent.get(flow)
                dom = Counter(recent).most_common(1)[0] if recent else None
                routes.append({"flow": f"{flow[0]}→{flow[1]}", "current": cur, "previous": self._route_prev.get(flow),
                               "dominant": {"path": list(dom[0]), "share": dom[1] / len(recent)} if dom else None,
                               "window": len(recent) if recent else 0})
            snap = {
                "active": True, "status": self.status(), "error": self.error, "episode": self.episode, "t": round(sim.t, 2),
                "duration": sim.duration, "speed": self.speed, "reset_info": self.reset_info,
                "params": {"alpha": pol.alpha, "gamma": pol.gamma, "epsilon": pol.epsilon, "initial_epsilon": pol.initial_epsilon,
                           "explore_enabled": pol.explore_enabled, "traffic_multiplier": self.traffic_multiplier},
                "scenario": {"topology": self.cfg.network.topology, "seed": self.cfg.simulation.seed, "n_nodes": len(sim.routers),
                             "sources": list(self.cfg.traffic.sources), "destinations": list(self.cfg.traffic.destinations),
                             "rate_pps": self.cfg.traffic.rate_pps},
                "graph": self._graph(),
                "links": [[1 if sim.links[(u, v)].is_up and sim.links[(v, u)].is_up else 0,
                           round(float(max(sim.links[(u, v)].utilization_ema, sim.links[(v, u)].utilization_ema)), 3)]
                          for u, v in self._edges],
                "routes": routes,
                "focus": ({"decision": fd, "rows": rows} if fd else None),
                "kpis": {
                    "router": fd["router"] if fd else None, "dst": fd["dst"] if fd else None,
                    "selected": fd["selected"] if fd else None, "mode": fd["mode"] if fd else None,
                    "best_valid_q": max(valid_q) if valid_q else None, "epsilon": pol.epsilon,
                    "explore_enabled": pol.explore_enabled, "updates": pol.updates,
                    "route_changes": self.route_changes_traced, "q_entries": sum(len(v) for v in pol.q_table.values()),
                    "q_states": len(pol.q_table), "decisions": self.decision_seq, "pending_decisions": pending, "delivered": self.delivered,
                    "drops": dict(self.drops), "explorations": pol.explorations,
                },
                "math": self._math(fu) if fu else None,
                "decisions": dec_list[-limit:][::-1],
                "updates": [u for u in upd_list[-limit:]][::-1],
                "events": ([{"t": None, "kind": "sim", "text": m} for m in sim.event_log[-20:]][::-1]),
                "controls": list(self.controls_log)[::-1],
                "down_links": [[u, v] for u, v in self._edges if not (sim.links[(u, v)].is_up and sim.links[(v, u)].is_up)],
            }
            return snap

    def qtable_page(self, router: Optional[str] = None, dst: Optional[str] = None, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
        limit = max(1, min(int(limit), QTABLE_PAGE_MAX))
        offset = max(0, int(offset))
        with self.lock:
            rows = []
            for state in sorted(self.policy.q_table, key=lambda s: (str(s[0]), str(s[1]), s[2])):
                if router and state[0] != router:
                    continue
                if dst and state[1] != dst:
                    continue
                for a, q in sorted(self.policy.q_table[state].items()):
                    lu = self._last_upd.get((state, a))
                    link = self.sim.links.get((state[0], a))
                    rows.append({"router": state[0], "dst": state[1], "level": state[2], "hop": a, "q": q,
                                 "link_up": bool(link.is_up) if link else None,
                                 "last_update_t": lu["t"] if lu else None})
            total = len(rows)
            return {"total": total, "offset": offset, "limit": limit, "rows": rows[offset:offset + limit],
                    "routers": sorted({s[0] for s in self.policy.q_table}),
                    "dsts": sorted({s[1] for s in self.policy.q_table})}


CONTROL_ACTIONS = ("play", "pause", "follow", "speed", "step", "reset", "traffic", "fail_link", "restore_link", "explore", "params")
