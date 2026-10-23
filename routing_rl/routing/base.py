"""Abstract routing policy interface with rich hooks for RL."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any, List
from routing_rl.network.packet import Packet
from routing_rl.network.link import Link
from routing_rl.network.router import Router


class RoutingPolicy(ABC):
    def __init__(
        self,
        routers: Dict[str, Router],
        links: Dict[Tuple[str, str], Link],
        reward_cfg: Optional[Dict[str, float]] = None,
    ):
        self.routers = routers
        self.links = links
        self.reward_cfg = reward_cfg or {}
        for r in routers.values():
            r.policy = self
        self._decision_log: List[Dict[str, Any]] = []

    @abstractmethod
    def select_next_hop(
        self, packet: Packet, router: Router, now: float
    ) -> Optional[str]:
        ...

    # Lifecycle hooks -------------------------------------------------
    def on_link_change(self) -> None:
        """Called when any link status/capacity changes."""
        pass

    def on_hop_forward(
        self, packet: Packet, router_id: str, next_hop: str, now: float
    ) -> None:
        pass

    def on_packet_delivered(self, packet: Packet, now: float) -> None:
        pass

    def on_packet_dropped(self, packet: Packet, reason: str, now: float) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    # Diagnostics -----------------------------------------------------
    def verbose_decision(self, packet: Packet, router: Router, now: float) -> str:
        return ""

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def reset_stats(self) -> None:
        self._decision_log.clear()
