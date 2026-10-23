"""Static shortest-path routing (minimum hop count over the links that are currently up)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import networkx as nx
from .base import RoutingPolicy
from routing_rl.network.packet import Packet
from routing_rl.network.router import Router


class ShortestPathPolicy(RoutingPolicy):
    def __init__(self, routers, links, reward_cfg=None):
        super().__init__(routers, links, reward_cfg)
        self._next_hop: Dict[Tuple[str, str], str] = {}
        self.unreachable: set = set()
        self._recompute()

    def _recompute(self) -> None:
        """Rebuild the next-hop table. Errors are not swallowed; unreachable pairs are recorded."""
        G = nx.DiGraph()
        G.add_nodes_from(self.routers)
        G.add_edges_from((s, d) for (s, d), link in self.links.items() if link.is_up)
        table: Dict[Tuple[str, str], str] = {}
        for src in self.routers:
            for dest, path in nx.single_source_shortest_path(G, src).items():
                if len(path) >= 2:
                    table[(src, dest)] = path[1]
        self._next_hop = table
        self.unreachable = {(s, d) for s in self.routers for d in self.routers
                            if s != d and (s, d) not in table}

    def on_link_change(self) -> None:
        self._recompute()

    def is_reachable(self, src: str, dst: str) -> bool:
        return src == dst or (src, dst) in self._next_hop

    def select_next_hop(self, packet: Packet, router: Router, now: float) -> Optional[str]:
        """Next hop on the current shortest path, or None when the destination is unreachable."""
        return self._next_hop.get((router.id, packet.dst))