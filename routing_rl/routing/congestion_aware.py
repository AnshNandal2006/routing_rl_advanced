"""Congestion-aware heuristic.

cost(n) = hops(n -> dst) + alpha * ( queue_delay(n) + beta * link_delay(router -> n) )

* hops(n -> dst):   shortest hop count over links that are up (path-length term).
* queue_delay(n):   packets queued at n divided by n's total outgoing capacity, in seconds.
                    Zero when n is the destination (packets are delivered, not queued).
* link_delay(link): per-packet service time at the capacity still free on the link we would use,
                    1 / residual_pps, plus propagation delay. A saturated or low-capacity link
                    therefore looks expensive even if the next router's queue is empty.
Down links, dead ends and routers the packet has already visited are never chosen.
"""
from __future__ import annotations
from typing import Dict, Optional
import networkx as nx
from .base import RoutingPolicy
from routing_rl.network.packet import Packet
from routing_rl.network.router import Router

UNREACHABLE = 10 ** 6


class CongestionAwarePolicy(RoutingPolicy):
    def __init__(self, routers, links, alpha: float = 1.5, reward_cfg=None, beta: float = 1.0):
        super().__init__(routers, links, reward_cfg)
        self.alpha = alpha
        self.beta = beta
        self._hop_dist: Dict[str, Dict[str, int]] = {}
        self._recompute()

    def _recompute(self) -> None:
        G = nx.DiGraph()
        G.add_nodes_from(self.routers)
        G.add_edges_from((s, d) for (s, d), link in self.links.items() if link.is_up)
        self._hop_dist = dict(nx.all_pairs_shortest_path_length(G))

    def on_link_change(self) -> None:
        self._recompute()

    def _hops(self, node: str, dst: str) -> int:
        return self._hop_dist.get(node, {}).get(dst, UNREACHABLE)

    def _queue_delay(self, router: Router) -> float:
        cap = sum(l.capacity_pps for n in router.neighbors
                  if (l := self.links.get((router.id, n))) is not None and l.is_up)
        return float(UNREACHABLE) if cap <= 0 else len(router) / cap

    @staticmethod
    def _link_delay(link) -> float:
        return 1.0 / max(link.available_capacity_pps(), 1.0) + link.propagation_delay

    def candidate_costs(self, packet: Packet, router: Router) -> Dict[str, Dict[str, float]]:
        """Cost breakdown for every usable next hop (useful for tests and debugging)."""
        out: Dict[str, Dict[str, float]] = {}
        for n in router.neighbors:
            link = self.links.get((router.id, n))
            if link is None or not link.is_up or n in packet.path:
                continue
            hops = self._hops(n, packet.dst)
            if hops >= UNREACHABLE:
                continue
            qd = 0.0 if n == packet.dst else self._queue_delay(self.routers[n])
            ld = self._link_delay(link)
            out[n] = {"hops": hops, "queue_delay": qd, "link_delay": ld,
                      "cost": hops + self.alpha * (qd + self.beta * ld)}
        return out

    def select_next_hop(self, packet: Packet, router: Router, now: float) -> Optional[str]:
        costs = self.candidate_costs(packet, router)
        if not costs:
            return None
        return min(costs, key=lambda n: costs[n]["cost"])  # ties keep neighbour order

    def verbose_decision(self, packet: Packet, router: Router, now: float) -> str:
        lines = [f"[t={now:7.2f}] {router.id} q={len(router):3d} → {packet.dst}"]
        costs = self.candidate_costs(packet, router)
        for n in router.neighbors:
            c = costs.get(n)
            lines.append(f"    {n}: unavailable" if c is None else
                         f"    {n}: hops={c['hops']} qdelay={c['queue_delay']*1000:6.1f}ms "
                         f"linkdelay={c['link_delay']*1000:6.1f}ms cost={c['cost']:.3f}")
        return "\n".join(lines)