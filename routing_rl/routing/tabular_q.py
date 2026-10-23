"""Improved tabular Q-Learning with loop avoidance and shaped rewards."""
from __future__ import annotations
import random
from typing import Dict, Optional, Tuple, List, Any
from .base import RoutingPolicy
from routing_rl.network.packet import Packet
from routing_rl.network.router import Router


class TabularQPolicy(RoutingPolicy):
    def __init__(
        self,
        routers,
        links,
        alpha: float = 0.15,
        gamma: float = 0.92,
        epsilon: float = 0.25,
        reward_cfg: Optional[Dict] = None,
        rng=None,
    ):
        super().__init__(routers, links, reward_cfg)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        # ``rng`` is any object with random()/choice(). The default is the ``random`` module itself, so
        # behaviour is identical to before; the live demo passes a private random.Random(seed) so that it
        # stays reproducible even while other simulations run in the same process.
        self.rng = rng if rng is not None else random
        self.explore_enabled = True   # live demo can switch exploration off (greedy only)
        # Optional observer: tracer(event_dict). None (the default) means zero extra work and no behaviour change.
        self.tracer = None
        self._last_sel: Dict[int, Dict[str, Any]] = {}
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9992
        self.q_table: Dict[Tuple[str, str, int], Dict[str, float]] = {}
        self.updates = 0
        self.explorations = 0

        # Reward coefficients
        rc = self.reward_cfg
        self.r_hop = rc.get("hop_penalty", -1.0)
        self.r_queue = rc.get("queue_penalty_scale", -2.5)
        self.r_delivery = rc.get("delivery_bonus", 40.0)
        self.r_drop = rc.get("drop_penalty", -60.0)

    def _get_q(self, state: Tuple, action: str) -> float:
        return self.q_table.get(state, {}).get(action, 0.0)

    def _set_q(self, state: Tuple, action: str, value: float) -> None:
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = value

    def _update_q(
        self,
        state,
        action,
        reward,
        next_state,
        next_actions: List[str],
        now: Optional[float] = None,
        trigger: str = "",
        packet: Optional[Packet] = None,
        reward_parts: Optional[Dict[str, Any]] = None,
    ) -> None:
        existed = action in self.q_table.get(state, {})
        q = self._get_q(state, action)
        max_next = max((self._get_q(next_state, a) for a in next_actions), default=0.0)
        target = reward + self.gamma * max_next
        self._set_q(state, action, q + self.alpha * (target - q))
        self.updates += 1
        if self.tracer is not None:
            next_q = {a: self._get_q(next_state, a) for a in next_actions}
            self.tracer({
                "type": "update", "t": now, "trigger": trigger,
                "packet_id": packet.packet_id if packet is not None else None,
                "state": state, "action": action, "first_visit": not existed,
                "q_old": q, "alpha": self.alpha, "gamma": self.gamma, "reward": reward,
                "next_state": next_state, "next_q": next_q, "max_next": max_next,
                "target": target, "q_new": self._get_q(state, action),   # value read back from the real table
                "reward_parts": reward_parts or {}, "update_index": self.updates,
            })

    def select_next_hop(
        self, packet: Packet, router: Router, now: float
    ) -> Optional[str]:
        valid = [
            n
            for n in router.neighbors
            if self.links.get((router.id, n)) and self.links[(router.id, n)].is_up
        ]
        down = [n for n in router.neighbors if n not in valid]
        # Strong loop avoidance
        unvisited = [a for a in valid if a not in packet.path]
        loop_excluded: List[str] = []
        if unvisited:
            loop_excluded = [a for a in valid if a in packet.path]
            valid = unvisited
        if not valid:
            return None

        state = (router.id, packet.dst, router.queue_level())
        if self.explore_enabled and self.rng.random() < self.epsilon:
            next_hop = self.rng.choice(valid)
            self.explorations += 1
            mode = "explore"
        else:
            q_vals = {a: self._get_q(state, a) for a in valid}
            max_q = max(q_vals.values())
            best = [a for a, q in q_vals.items() if q == max_q]
            next_hop = self.rng.choice(best)
            mode = "greedy" if len(best) == 1 else "greedy_tie"

        packet.pending_state = state
        packet.pending_action = next_hop
        packet.pending_router = router.id
        if self.tracer is not None:
            # Snapshot of exactly what the selection rule saw. It becomes a "decision" only once the
            # packet is really forwarded (on_hop_forward); a selection that is retried is overwritten.
            q_now = {a: self._get_q(state, a) for a in valid}
            top = max(q_now.values())
            self._last_sel[packet.packet_id] = {
                "t": now, "packet_id": packet.packet_id, "priority": packet.is_priority,
                "src": packet.src, "dst": packet.dst, "router": router.id, "state": state,
                "path": list(packet.path), "mode": mode, "selected": next_hop,
                "epsilon": self.epsilon, "explore_enabled": self.explore_enabled,
                "candidates": [{"hop": a, "q": q_now[a], "visited": a in self.q_table.get(state, {})} for a in valid],
                "greedy_set": [a for a, q in q_now.items() if q == top],
                "loop_excluded": loop_excluded, "down": down,
                "queue_len": len(router), "queue_level": state[2],
            }
        return next_hop

    def on_hop_forward(
        self, packet: Packet, router_id: str, next_hop: str, now: float
    ) -> None:
        next_r = self.routers[next_hop]
        next_state = (next_hop, packet.dst, next_r.queue_level())
        next_valid = [
            n
            for n in next_r.neighbors
            if self.links.get((next_hop, n)) and self.links[(next_hop, n)].is_up
        ]
        hop_term = self.r_hop
        queue_term = self.r_queue * next_r.queue_level()
        reward = hop_term + queue_term
        priority_mult = 1.15 if packet.is_priority else 1.0
        if packet.is_priority:
            reward *= 1.15  # slightly stronger signal for priority
        sel = self._last_sel.pop(packet.packet_id, None)
        if self.tracer is not None and sel is not None:
            sel = dict(sel, forwarded_to=next_hop, forwarded_t=now,
                       next_queue_level=next_r.queue_level(), next_queue_len=len(next_r))
            self.tracer({"type": "decision", **sel})
        if packet.pending_state is not None and packet.pending_action is not None:
            self._update_q(
                packet.pending_state,
                packet.pending_action,
                reward,
                next_state,
                next_valid,
                now=now, trigger="hop_forward", packet=packet,
                reward_parts={"hop": hop_term, "queue": queue_term, "priority_multiplier": priority_mult,
                              "next_router": next_hop, "next_queue_level": next_r.queue_level()},
            )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def on_packet_delivered(self, packet: Packet, now: float) -> None:
        if packet.pending_state is not None and packet.pending_action is not None:
            bonus = self.r_delivery
            if packet.is_priority:
                bonus *= 1.3
            self._update_q(
                packet.pending_state,
                packet.pending_action,
                bonus,
                None,
                [],
                now=now, trigger="delivered", packet=packet,
                reward_parts={"delivery_bonus": bonus},
            )
        self._last_sel.pop(packet.packet_id, None)
        if self.tracer is not None:
            self.tracer({"type": "delivered", "t": now, "packet_id": packet.packet_id,
                         "src": packet.src, "dst": packet.dst, "path": list(packet.path),
                         "delay": now - packet.creation_time})

    def on_packet_dropped(self, packet: Packet, reason: str, now: float) -> None:
        if packet.pending_state is not None and packet.pending_action is not None:
            self._update_q(
                packet.pending_state,
                packet.pending_action,
                self.r_drop,
                None,
                [],
                now=now, trigger=f"dropped:{reason}", packet=packet,
                reward_parts={"drop_penalty": self.r_drop, "reason": reason},
            )
        self._last_sel.pop(packet.packet_id, None)
        if self.tracer is not None:
            self.tracer({"type": "dropped", "t": now, "packet_id": packet.packet_id,
                         "src": packet.src, "dst": packet.dst, "reason": reason})

    def q_values(self, state: Tuple, actions: List[str]) -> Dict[str, float]:
        """Current Q(state, a) for each action (0.0, the agent's initial value, for unvisited pairs)."""
        return {a: self._get_q(state, a) for a in actions}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "q_entries": sum(len(v) for v in self.q_table.values()),
            "updates": self.updates,
            "explorations": self.explorations,
            "epsilon": self.epsilon,
        }
